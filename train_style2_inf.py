import argparse
import math
import random
import os

import pickle
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm

try:
    import wandb

except ImportError:
    wandb = None

from Dataset_inf.InfDataset_mouse import InfDataset
from utils.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from utils.non_leaking import augment, AdaptiveAugment
from utils.inft import get_global_n, get_layer_n, run_inf_img, align_raw_img
from style3.models.stylegan2.op import conv2d_gradfix
from style3.models.stylegan2.model_inf import Generator, Discriminator
from Dataset_inf.config import cfg_mouse

from pathlib import Path

def linspace(start, stop, num):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32).to(start) / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]

    return out

def mixing_gene(gene, noise_dim, mix=False, fix_noise=False, prob=0):
    if mix:
        gbat = gene.shape[0]
        gene = gene[torch.randperm(gbat)]
        gene = (gene[:gbat//2] + gene[gbat//2:]) / 2
    if prob > 0 and random.random() < prob:
        sign = torch.randint_like(gene, -1, 2)
        gene += torch.rand_like(gene) * sign
    if fix_noise:
        noise = torch.randn(noise_dim).to(gene)
        noise = noise.unsqueeze(0).repeat(gene.shape[0], 1)
    else:
        noise = torch.randn(gene.shape[0], noise_dim).to(gene)
    return [noise, gene]


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, check_save):
    noise_dim = args.latent

    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        real_img, real_gene, real_label = next(loader)
        real_img = real_img.to(device)
        noise_gene = torch.rand(real_gene.shape[0], 1, 1, args.gene) 
        real_gene = (real_gene + (noise_gene - 0.5) * 2).to(device)
        real_label = real_label.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_gene(real_gene, noise_dim, prob=args.mixing)
        fake_img, _ = generator(noise, real_gene)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        if idx == 0:
            print(fake_img.shape, real_img.shape, real_gene.shape, real_label.shape)
        fake_pred = discriminator(fake_img, real_label)
        real_pred = discriminator(real_img_aug, real_label)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug, real_label)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_gene(real_gene, noise_dim, prob=args.mixing)
        fake_img, _ = generator(noise, real_gene)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred = discriminator(fake_img, real_label)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            # path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_gene(real_gene, noise_dim, mix=True, prob=args.mixing)
            fake_img, latents = generator(noise, noise[1], return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if (i + 1) in (1, 1000, 2000, 4000, 5000, 10000) or (i + 1) % 50000 == 0:
                # This may only work for mouse, args.batch must be 8
                glob_n = get_global_n(noise_dim, args.batch * 2).to(real_gene)
                layer_n = get_layer_n(g_ema, args.batch * 2, 
                                      device=real_gene.device)
                if args.data == 'CosMx':
                    raw_sz, offset = 160, 512
                    gpths = [f'Data/CosMx/Liver{i + 1}/GeneLabels/GeneLabels_{fl}.npz' for
                             i, fl in enumerate(('F005', 'F009'))]
                elif args.data == 'Xenium':
                    raw_sz, offset = 96, 0
                    gpths = [f'Data/Xenium/Lung{i + 1}/rna/{fl}_rna.npz' for
                             i, fl in enumerate(('16000_20000_32000_36000_15800_20200_31800_36200', 
                                                 '12000_16000_32000_36000_11800_16200_31800_36200'))]
                elif args.data == 'Xenium_breast':
                    raw_sz, offset = 128, 0
                    gpths = [f'Data/Xenium_breast/Breast{i + 1}/rna/{fl}_rna.npz' for
                             i, fl in enumerate(('57344_59392_4096_6144_57216_59520_3968_6272', 
                                                 '71680_73728_38912_40960_71552_73856_38784_41088'))]
                elif args.data == 'Xenium_mouse':
                    raw_sz, offset = 128, 0
                    gpths = [f'Data/Xenium_mouse/rna/{fl}_rna.npz' for
                             i, fl in enumerate(('4096_6144_36864_38912_4032_6208_36800_38976', 
                                                 '10240_12288_45056_47104_10176_12352_44992_47168'))]
                for gid, gpth in enumerate(gpths):
                    save_pth = check_save / 'sample' / \
                                f'{gid + 1}_{str(i + 1).zfill(6)}.png'
                    run_inf_img(g_ema, glob_n, layer_n,
                                gpth, offset, glob_n.shape[0],
                                raw_sz, str(save_pth), False)
                    if i == 0:
                        save_pth = check_save / 'sample' / \
                                f'{gid + 1}_raw.png'
                        align_raw_img(gpth, offset, glob_n.shape[0],
                                      raw_sz, save_pth)

            if (i + 1) % 50000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    str(check_save / 'checkpoint' /
                        '{}.pt'.format(str(i + 1).zfill(6))),
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--data", type=str, help="name of the dataset")
    parser.add_argument("--proc", type=str, choices=('crop', 'tile'), 
                        default='crop', 
                        help="processing method (center-crop or tile with fixed gap)")
    parser.add_argument("--gene", type=int, help="num of the genes")
    parser.add_argument("--gene_use", action="store_true", help="whether to use gene expr")
    parser.add_argument("--cell_label", type=str, default='')
    parser.add_argument("--train_sub", type=str, default='')
    parser.add_argument("--split_scheme", type=str, help="split to subset")
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--img_chn", type=int, default=3)
    parser.add_argument("--latent", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=-1,
        choices=[0, 1, 2, 3, 4, 5, -1],
        help="either train the generator with single channel (chn = 1,...,5) or all the channels (-1)",
    )
    parser.add_argument("--check_save", type=str, help="path to the train output")
    parser.add_argument("--stain", type=str, default='dapi', help="name of the dataset")

    args = parser.parse_args()
    if args.stain == 'dapi':
        assert args.img_chn == 1
    elif args.stain == 'he':
        assert args.img_chn == 3

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.start_iter = 0

    if args.train_sub:
        if args.train_sub == 'Skin':
            sub_pth = [Path('Data/Xenium_mouse/rna/4096_6144_51200_53248_4032_6208_51136_53312_rna.npz'),]
            sm_roi = None
        elif args.train_sub == 'Brain':
            sub_pth = [Path('Data/Xenium_mouse/rna/94208_96256_22528_24576_94144_96320_22464_24640_rna.npz'),]
            sm_roi = None
        elif args.train_sub == 'Colon':
            sub_pth = [Path('Data/Xenium_mouse/rna/32768_34816_36864_38912_32704_34880_36800_38976_rna.npz'),]
            sm_roi = None
        elif args.train_sub == 'Kidney':
            sub_pth = [Path('Data/Xenium_mouse/rna/24576_26624_14336_16384_24512_26688_14272_16448_rna.npz'),]
            sm_roi = None
        elif args.train_sub == 'Liver':
            sub_pth = [Path('Data/Xenium_mouse/rna/34816_36864_14336_16384_34752_36928_14272_16448_rna.npz'),]
            sm_roi = None
        elif args.train_sub == 'Lung':
            sub_pth = [Path('Data/Xenium_mouse/rna/55296_57344_16384_18432_55232_57408_16320_18496_rna.npz'),]
            sm_roi = None
        else:
            subs, frc = args.train_sub, 0
            if 'checkerboard' in subs or 'random' in subs:
                subs, frc = subs.split('-')
            cfg = cfg_mouse(Path(f'Data/{args.data}'), subs, 'organs', float(frc))
            sub_pth, sm_roi = cfg.pths, cfg.sm_roi
    else:
        sub_pth, sm_roi = None, None
    dataset = InfDataset(f'Data/{args.data}', args.gene, args.size, 
                         label=args.cell_label, sub_pth=sub_pth,
                         transform=True, use_labels=True, stain=args.stain,
                         sm_roi=sm_roi, repeat=10**5)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        **{'drop_last': True,
           'num_workers': 8,
           'pin_memory': True}
    )

    gene_num = args.gene if args.gene_use else 0
    generator = Generator(
        args.size, gene_num, args.latent, args.kernel_size, args.n_mlp, channel_multiplier=args.channel_multiplier, img_chn=args.img_chn,
    ).to(device)
    discriminator = Discriminator(
        args.size, dataset._ldim, channel_multiplier=args.channel_multiplier, img_chn=args.img_chn
    ).to(device)
    g_ema = Generator(
        args.size, gene_num, args.latent, args.kernel_size, args.n_mlp, channel_multiplier=args.channel_multiplier, img_chn=args.img_chn,
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")

    check_save = Path(args.check_save)
    check_parm = f'{check_save.name}_{args.gene_use}_{args.mixing}_{args.split_scheme}_{args.kernel_size}_{args.latent}_{args.proc}_{dataset._ldim}'
    if args.train_sub:
        check_parm = f'{check_parm}_{args.train_sub}'
    check_save = check_save.parent / check_parm
    check_save.mkdir(parents=True, exist_ok=True)
    (check_save / 'checkpoint').mkdir(parents=True, exist_ok=True)
    (check_save / 'sample').mkdir(parents=True, exist_ok=True)
    # if args.train_sub:
    #     (check_save / 'pths').mkdir(parents=True, exist_ok=True)
    #     print(args.train_sub)
    #     with open(str(check_save / 'pths'/ 'pths.pickle'), 'wb') as f:
    #         pickle.dump(cfg.pths, f)
    #     with open(str(check_save / 'pths'/ 'pths_arr.pickle'), 'wb') as f:
    #         pickle.dump(cfg.pths_arr, f)
    # print(generator, discriminator)
    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, check_save)
