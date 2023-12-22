import torch
import matplotlib
import dataclasses
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch import nn, autograd
from pathlib import Path
from torch.utils.data import DataLoader

from style3.inversion.models.psp3 import pSp
from style3.models.stylegan2.model import Discriminator

from style3.criteria import moco_loss
from style3.criteria.lpips.lpips import LPIPS

from style3.utils.ranger import Ranger
from style3.utils import common, train_utils
from style3.utils.data_utils import linspace


matplotlib.use('Agg')


class Coach:
    def __init__(self, opts, data):
        self.opts = opts
        self.global_step = 0
        self.device = 'cuda:0'
        self.opts.device = self.device

        # Initialize network
        self.net = pSp(self.opts).to(self.device)

        # Initialize loss
        assert self.opts.id_lambda == 0 and self.opts.w_norm_lambda == 0
        self.mse_loss = nn.MSELoss().to(self.device).eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.opts.moco_lambda > 0:
            self.moco_loss = moco_loss.MocoLoss()
        if self.opts.w_discriminator_lambda > 0:
            self.discriminator = Discriminator(
                opts.output_size, data._n_classes, img_chn=opts.input_nc).to(self.device)
            self.discriminator_optimizer = torch.optim.Adam(list(self.discriminator.parameters()),
                                                            lr=opts.w_discriminator_lr)

        # Initialize optimizer
        self.optimizer = self.set_optimizers()

        # Initialize dataset
        self.train_dataloader = DataLoader(data,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           drop_last=True)
        self.set_test()

        # Initialize log_dir
        self.log_dir = opts.exp_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True, parents=True)

        # Initialize checkpoint dir
        self.checkpoint_dir = opts.exp_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

    def set_test(self, idx=8):
        self.x_test, self.y_test, self.gene_test = [], [], []
        for i, ((x, gene), _, _) in enumerate(self.train_dataloader):
            if i == idx:
                break
            x_test = x.to(self.device)
            y_test = x_test.detach().clone()
            gene_test = gene.to_dense().to(self.device)
            self.x_test.append(x_test)
            self.y_test.append(y_test)
            self.gene_test.append(gene_test)

    def set_optimizers(self):
        for name, parm in self.net.decoder.named_parameters():
            if not self.opts.train_decoder:
                parm.requires_grad = False
            else:
                print(name, parm.requires_grad)

        for name, parm in self.net.encoder.named_parameters():
            if 'clip.' in name:
                print(name)
                parm.requires_grad = False

        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(self.net.parameters(),
                                         lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(self.net.parameters(),
                               lr=self.opts.learning_rate)
        return optimizer

    def perform_train_iteration_on_batch(self, x, y, gene):
        if self.opts.w_discriminator_lambda > 0:
            self.requires_grad(self.discriminator, False)
        self.optimizer.zero_grad()
        y_hat = self.net.forward(x, gene)
        loss, loss_dict, id_logs = self.calc_loss(x, y, y_hat, gene)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return y_hat, loss_dict, id_logs

    def train(self):
        self.net.train()
        while self.global_step < self.opts.max_steps:
            for _, ((x, gene), _, _) in enumerate(self.train_dataloader):
                x = x.to(self.device)
                y = x.detach().clone()
                gene = gene.to_dense().to(self.device)

                y_hat, loss_dict, id_logs = self.perform_train_iteration_on_batch(
                    x, y, gene)
                disc_loss_dict = self.compute_discriminator_loss(
                    x, y_hat, gene)

                loss_dict = {**disc_loss_dict, **loss_dict}
                # store intermediate outputs
                y_hats = {idx: [] for idx in range(x.shape[0])}
                for idx in range(x.shape[0]):
                    y_hats[idx].append([y_hat[idx], None])

                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
                    self.parse_and_log_images(id_logs, x, y, y_hats,
                                              title='images/train')

                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict)
                    else:
                        self.checkpoint_me(loss_dict)

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1

    def perform_val_iteration_on_batch(self, x, y, gene):
        y_hats = {idx: [] for idx in range(x.shape[0])}
        y_hat = self.net.forward(x, gene=gene,
                                 interp=True, randomize_noise=False)

        loss, cur_loss_dict, id_logs = self.calc_loss(x, y, y_hat, gene)
        # store intermediate outputs
        for idx in range(x.shape[0]):
            y_hats[idx].append([y_hat[idx], None])

        return y_hats, cur_loss_dict, id_logs

    def validate(self):
        self.net.eval()
        ndct = {}
        for name, parm in self.net.decoder.named_parameters():
            if 'noise.weight' in name:
                ndct[name] = parm
        print(ndct)

        agg_loss_dict = []
        for i in range(len(self.x_test)):
            with torch.no_grad():
                y_hats, cur_loss_dict, id_logs = self.perform_val_iteration_on_batch(self.x_test[i],
                                                                                     self.y_test[i],
                                                                                     self.gene_test[i])
                agg_loss_dict.append(cur_loss_dict)

            # Logging related
            self.parse_and_log_images(id_logs, self.x_test[i], self.y_test[i],
                                      y_hats, title='images/test',
                                      subscript=f'{i:04d}', display_count=8)

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and i >= 4:
                self.net.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict):
        save_dict = self._get_save_dict()
        checkpoint_path = self.checkpoint_dir / f'{self.global_step}.pt'
        torch.save(save_dict, checkpoint_path)
        with open(self.checkpoint_dir / 'timestamp.txt', 'a') as f:
            f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

    def calc_loss(self, x, y, y_hat, gene):
        loss_dict, loss, id_logs = {}, .0, None
        # Adversarial loss
        if self.is_training_discriminator():
            loss_disc = self.compute_adversarial_loss(y_hat, gene)
            loss_dict['loss_disc'] = float(loss_disc)
            loss += self.opts.w_discriminator_lambda * loss_disc
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
            y_hat = y_hat.repeat(1, 3, 1, 1)
        elif x.shape[1] == 2:
            pad = -torch.ones_like(x[:, 0])[:, None]
            x = torch.cat([pad, x], 1)
            y = torch.cat([pad, y], 1)
            y_hat = torch.cat([pad, y_hat], 1)
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.moco_lambda > 0:
            moco = self.moco_loss(y_hat, y, x)
            loss_moco, sim_improvement, id_logs = moco
            loss_dict['loss_moco'] = float(loss_moco)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_moco * self.opts.moco_lambda
        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def print_metrics(self, metrics_dict, prefix):
        print(f'Metrics for {prefix}, step {self.global_step}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = ', value)

    def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=8):
        display_count = min(display_count, len(x))
        y = linspace(y[0], y[1], y.shape[0])
        im_data = []
        for i in range(display_count):
            if type(y_hat) == dict:
                output_face = [
                    [common.tensor2im(y_hat[i][iter_idx][0]),
                     y_hat[i][iter_idx][1]]
                    for iter_idx in range(len(y_hat[i]))
                ]
            else:
                output_face = [common.tensor2im(y_hat[i])]
            cur_im_data = {
                'input_face': common.tensor2im(x[i]),
                'target_face': common.tensor2im(y[i]),
                'output_face': output_face,
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = train_utils.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = Path(self.log_dir) / name / \
                f'{subscript}_{step:04d}.jpg'
        else:
            path = Path(self.log_dir) / name / f'{step:04d}.jpg'
        path.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(path)
        plt.close(fig)

    def _get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': dataclasses.asdict(self.opts)}
        return save_dict

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Util Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def get_dims_to_discriminate(self):
        deltas_starting_dimensions = self.net.encoder.get_deltas_starting_dimensions()
        return deltas_starting_dimensions[:self.net.encoder.progressive_stage.value + 1]

    def is_progressive_training(self):
        return self.opts.progressive_steps is not None

    def check_for_progressive_training_update(self, is_resume_from_ckpt=False):
        for i in range(len(self.opts.progressive_steps)):
            # Case checkpoint
            if is_resume_from_ckpt and self.global_step >= self.opts.progressive_steps[i]:
                self.net.encoder.set_progressive_stage(ProgressiveStage(i))
            # Case training reached progressive step
            if self.global_step == self.opts.progressive_steps[i]:
                self.net.encoder.set_progressive_stage(ProgressiveStage(i))

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Discriminator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def compute_discriminator_loss(self, x, y_hat, gene):
        disc_loss_dict = {}
        if self.is_training_discriminator():
            disc_loss_dict = self.train_discriminator(x, y_hat, gene)
        return disc_loss_dict


    def compute_adversarial_loss(self, y_hat, gene):
        fake_pred = self.discriminator(y_hat, gene)
        loss_disc = F.softplus(-fake_pred).mean()
        return loss_disc


    def is_training_discriminator(self):
        return self.opts.w_discriminator_lambda > 0

    @staticmethod
    def discriminator_loss(real_pred, fake_pred, loss_dict):
        real_loss = F.softplus(-real_pred).mean()
        fake_loss = F.softplus(fake_pred).mean()
        loss_dict['d_real_loss'] = float(real_loss)
        loss_dict['d_fake_loss'] = float(fake_loss)
        return real_loss + fake_loss

    @staticmethod
    def discriminator_r1_loss(real_pred, real_w):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_w, create_graph=True)
        grad_penalty = grad_real.pow(2).reshape(
            grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty

    def train_discriminator(self, x, y_hat, gene):
        loss_dict = {}
        self.requires_grad(self.discriminator, True)

        real_pred = self.discriminator(x, gene)
        fake_pred = self.discriminator(y_hat, gene)
        loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
        loss_dict['discriminator_loss'] = float(loss)

        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        # r1 regularization
        d_regularize = self.global_step % self.opts.d_reg_every == 0
        if d_regularize:
            real = x.detach()
            real.requires_grad = True
            real_pred = self.discriminator(real, gene)
            r1_loss = self.discriminator_r1_loss(real_pred, real)

            self.discriminator.zero_grad()
            r1_final_loss = self.opts.r1 / 2 * r1_loss * \
                self.opts.d_reg_every + 0 * real_pred[0]
            r1_final_loss.backward()
            self.discriminator_optimizer.step()
            loss_dict['discriminator_r1_loss'] = float(r1_final_loss)

        # Reset to previous state
        self.requires_grad(self.discriminator, False)

        return loss_dict

    def validate_discriminator(self, x):
        with torch.no_grad():
            loss_dict = {}
            real_w, fake_w = self.sample_real_and_fake_latents(x)
            real_pred = self.discriminator(real_w)
            fake_pred = self.discriminator(fake_w)
            loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
            loss_dict['discriminator_loss'] = float(loss)
            return loss_dict

    def sample_real_and_fake_latents(self, x):
        sample_z = torch.randn(self.opts.batch_size,
                               512, device=self.device)
        real_w = self.net.decoder.get_latent(sample_z)
        fake_w = self.net.encoder(x)
        if self.is_progressive_training():  # When progressive training, feed only unique w's
            dims_to_discriminate = self.get_dims_to_discriminate()
            fake_w = fake_w[:, dims_to_discriminate, :]
        if self.opts.use_w_pool:
            real_w = self.real_w_pool.query(real_w)
            fake_w = self.fake_w_pool.query(fake_w)
        if fake_w.ndim == 3:
            fake_w = fake_w[:, 0, :]
        return real_w, fake_w
