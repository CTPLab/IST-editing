import copy
import cv2
import torch
import imageio
import numpy as np
import torch.nn.functional as F
import torchvision.utils as tv_utils

from math import ceil
from PIL import Image
from skimage.util.shape import view_as_windows
from utils.common import run_GAN, run_GANI, prep_input, prep_outim, add_bbx, prep_inp_mat, init_img_noise, set_img_noise, post_output


def run_one_demo(analysis, decoder,
                 model, img, gene, noise=None, shift=None):
    r"""
    Run the GAN (Inversion) model once to generate one video frame

    Args:
        analysis: The model used for analysis: GAN or GAN Inversion
        decoder: StyleGAN2 generator
        model: GAN or GAN Inversion
        img: morphological image
        gene: one-dim gene expression vector
        noise: Gaussian noise fed to the mapping of StyleGAN2
        shift: The morphological transition coefficient,
               fully transformed to the reference cell morphology 
               if shift == 1. Only for GAN Inversion
    """

    with torch.inference_mode():
        if analysis == 'GAN':
            out = run_GAN(decoder, model, gene, noise,
                          randomize_noise=False)
        elif 'GANI' in analysis:
            out = run_GANI(decoder, model, gene, img,
                           randomize_noise=False,
                           shift=shift)
    return out.clamp(-1, 1)


def run_demo_edit(args, path,
                  dload, eigen, model,
                  gene_idx, step=0.01, target_only=False):
    r"""
    Create cell image gallary for GAN (Inversion) model.

    Args:
        args: The parameters from args.py
        path: Path to the save folder
        dload: Dataloader
        eigen: List of eigen{vector, value} of 
               compared cell (sub)populations
        model: GAN (Inversion) model
        gene_idx: Indices of targeted gene expressions
        step: the step length of morph transition for each video frame
        target_only: set non-targeted genes to 0 if True (not very useful)
    """

    noise_fix = torch.randn((args.n_eval, 512))
    for i,  ((img, gene), _, _) in enumerate(dload):
        wrt = imageio.get_writer(str(path / f'{str(i).zfill(3)}.mp4'),
                                 fps=24)
        with torch.inference_mode():
            img, gene = prep_input(img, gene, args.gene_num,
                                   is_GAN=args.analysis == 'GAN')
            out = run_one_demo(args.analysis, args.decoder,
                               model, img, gene, noise_fix,
                               shift=0)
            ovid = torch.cat(out.unbind(0), -1)
            ivid = torch.cat(img.unbind(0), -1)
            gene_all = edit_gene(gene, eigen[0], eigen[1])
            gene_sub = edit_gene(gene, eigen[0], eigen[1], idx=gene_idx)

            for shf in np.arange(0, 1 + step, step):
                shf_all = (1 - shf) * gene + shf * gene_all
                shf_sub = (1 - shf) * gene + shf * gene_sub
                if target_only:
                    msk = torch.zeros(shf_sub.shape[-1]).to(gene)
                    msk[gene_idx] = 1
                    shf_sub = msk * shf_sub
                out_all = run_one_demo(args.analysis, args.decoder,
                                       model, img, shf_all, noise_fix,
                                       shift=shf)
                out_sub = run_one_demo(args.analysis, args.decoder,
                                       model, img, shf_sub, noise_fix,
                                       shift=shf)

                if shf == 1:
                    out_img = (torch.cat([img, out, out_sub, out_all]) + 1) / 2
                    if out_img.shape[1] == 2:
                        out_img = torch.cat((torch.zeros_like(out_img[:, 0])[:, None],
                                             out_img), dim=1)
                    tv_utils.save_image(out_img,
                                        str(path / f'{str(i).zfill(3)}.png'),
                                        nrow=args.n_eval,
                                        padding=2)

                output = torch.cat([ivid,
                                    ovid,
                                    torch.cat(out_sub.unbind(0), -1),
                                    torch.cat(out_all.unbind(0), -1)], 1)
                if output.shape[0] == 2:
                    output = torch.cat((-torch.ones_like(output[0])[None],
                                        output))
                output = output.transpose(0, 2).transpose(0, 1)
                output = prep_outim(output).cpu().numpy()
                wrt.append_data(output.astype('uint8'))
            wrt.close()
            return


def run_fov_edit(args, path, size,
                 fov, cat, bdy,
                 dload, eigen, model,
                 gene_idx, step=0.01, target_only=False):
    r"""
    Create ROI video demo for GAN Inversion model, which
    bdy parameters are customized for the selected ROI image.

    Args:
        args: The parameters from args.py
        path: Path to the save folder
        fov: Morpholocial image of region of interest
        cat: Cluster annotation image of ROI
        bdy: Boundary of the image to be cropped for demo
        dload: Dataloader
        eigen: list of eigen{vector, value} of 
               compared cell (sub)populations
        model: GAN Inversion model
        gene_idx: Indices of targeted gene expressions
        step: the step length of morph transition for each video frame
        target_only: set non-targeted genes to 0 if True (not very useful)
    """

    noise_fix = torch.randn((args.n_eval, 512))
    wrt = imageio.get_writer(str(path / 'fov_demo.mp4'),
                             fps=24)
    for shf in np.arange(0, 1 + step, step):
        print(shf)
        fov_edit = copy.deepcopy(fov).astype(np.float32)
        msk_edit = np.zeros_like(fov)
        if shf == 0:
            fov_bbx = copy.deepcopy(fov)
        for i,  ((img, gene), _, meta) in enumerate(dload):
            with torch.inference_mode():
                img, gene = prep_input(img, gene, args.gene_num,
                                       is_GAN=args.analysis == 'GAN')
                gene_sub = edit_gene(gene, eigen[0], eigen[1], idx=gene_idx)
                shf_sub = (1 - shf) * gene + shf * gene_sub
                if target_only:
                    msk = torch.zeros(shf_sub.shape[-1]).to(gene)
                    msk[gene_idx] = 1
                    shf_sub = msk * shf_sub
                out_sub = run_one_demo(args.analysis, args.decoder,
                                       model, img, shf_sub, noise_fix,
                                       shift=shf)
                out_sub = F.interpolate(out_sub, size,
                                        mode='bicubic', antialias=True)
                out_sub = prep_outim(out_sub)
                if out_sub.shape[1] == 2:
                    out_sub = torch.cat((torch.zeros_like(out_sub[:, 0])[:, None],
                                         out_sub), dim=1)
                out_sub = out_sub.transpose(1, 3).transpose(1, 2)
                for b in range(img.shape[0]):
                    if (b % 8 != i % 8 and args.data_name == 'Xenium') or \
                       (b % 8 != 1 and args.data_name == 'CosMx'):
                        continue
                    r = slice(meta[0][b] - size // 2,
                              meta[0][b] + size // 2)
                    c = slice(meta[1][b] - size // 2,
                              meta[1][b] + size // 2)
                    if (msk_edit[r, c] == 0).all():
                        msk_edit[r, c] = 1
                        img_crop = out_sub[b].cpu().numpy()
                        if args.data_name == 'Xenium':
                            img_crop = add_bbx(
                                np.repeat(img_crop, 3, -1))[:, :, 0]
                        else:
                            img_crop = add_bbx(img_crop)
                        fov_edit[r, c] = img_crop

                        if shf == 0:
                            cat[r, c] = add_bbx(cat[r, c])
                            if args.data_name == 'Xenium':
                                fov_bbx[r, c] = add_bbx(
                                    np.repeat(fov_bbx[r, c][:, :, None], 3, -1))[:, :, 0]
                            else:
                                fov_bbx[r, c] = add_bbx(fov_bbx[r, c])
        fov_edit = fov_edit[bdy[0]:bdy[1], bdy[0]:bdy[1]].astype('uint8')
        if args.data_name == 'Xenium':
            # This is configured for the selected lung ROI image
            fov_edit = fov_edit[-1024:, -1024:]
        wrt.append_data(fov_edit)
        if shf == 0 or shf == 1:
            im_nm = 'rec' if shf == 0 else 'edit'
            Image.fromarray(fov_edit).save(str(path / f'{im_nm}.png'))
            if shf == 0:
                Image.fromarray(fov_bbx[bdy[0]:bdy[1], bdy[0]:bdy[1]]).\
                    save(str(path / 'crop_bbx.png'))
                Image.fromarray(cat[bdy[0]:bdy[1], bdy[0]:bdy[1]]).\
                    save(str(path / 'cat_bbx.png'))
    wrt.close()
    return


def run_fov_edit1(args, path, num, size,
                  fov, cat, bdy,
                  dload, eigen, model,
                  gene_idx, step=0.5):
    if args.decoder == 'style2':
        noise_img = init_img_noise(model.decoder, 1, num)

    noise_fix = torch.randn((args.n_eval, 512))
    wrt = imageio.get_writer(str(path / 'fov_demo.mp4'),
                             fps=24)
    for shf in np.arange(0, 1 + step, step):
        print(shf)
        fov_edit = np.zeros_like(fov).astype(np.float32)
        msk_edit = np.zeros_like(fov).astype(np.float32)
        if shf == 0:
            fov_bbx = copy.deepcopy(fov)
        for i,  ((img, gene, msk), _, meta) in enumerate(dload):
            with torch.inference_mode():
                if args.decoder == 'style2':
                    set_img_noise(model.decoder, len(img),
                                  [n[i * args.n_eval:(i + 1) * args.n_eval] for n in noise_img[0]])
                img, gene = prep_input(img, gene, args.gene_num,
                                       is_GAN=args.analysis == 'GAN')
                gene_sub = edit_gene(gene, eigen[0], eigen[1], idx=gene_idx)
                shf_sub = (1 - shf) * gene + shf * gene_sub
                out_sub = run_one_demo(args.analysis, args.decoder,
                                       model, img, shf_sub, noise_fix,
                                       shift=shf)
                out_sub = F.interpolate(out_sub, size,
                                        mode='bicubic', antialias=True)
                out_sub = prep_outim(out_sub)
                if out_sub.shape[1] == 2:
                    out_sub = torch.cat((torch.zeros_like(out_sub[:, 0])[:, None],
                                         out_sub), dim=1)
                out_sub = out_sub.transpose(1, 3).transpose(1, 2)
                for b in range(img.shape[0]):
                    r = slice(meta[0][b] - size // 2,
                              meta[0][b] + size // 2)
                    c = slice(meta[1][b] - size // 2,
                              meta[1][b] + size // 2)
                    msk_one = msk[b].cpu().numpy()
                    msk_one = msk_one.transpose((1, 2, 0))
                    out_one = out_sub[b].squeeze().cpu().numpy()
                    fov_edit[r, c] = fov_edit[r, c] * (1 - msk_one) + \
                        out_one * msk_one

        fov_edit = np.clip(fov_edit, 0, 255)
        fov_edit = fov_edit[bdy[0]:bdy[1], bdy[0]:bdy[1]].astype('uint8')
        wrt.append_data(fov_edit)
        if shf == 0 or shf == 1:
            im_nm = 'rec' if shf == 0 else 'edit'
            Image.fromarray(fov_edit).save(str(path / f'{im_nm}.png'))
            if shf == 0:
                Image.fromarray(fov_bbx[bdy[0]:bdy[1], bdy[0]:bdy[1]]).\
                    save(str(path / 'crop_bbx.png'))
                Image.fromarray(cat[bdy[0]:bdy[1], bdy[0]:bdy[1]]).\
                    save(str(path / 'cat_bbx.png'))
    wrt.close()
    return


def edit_gene(x, eigx, eigr,
              wei=None, topk=1, idx=None):
    r"""
    The core function of in silico editing by matching 
    the SCM of gene expressions of a give cell population to another

    Args:
        x: The collection of gene expressions of 
           a given population
        eigx: Eigenvector and eigenvalue of x
        eigr: Eigenvector and eigenvalue of gene expressions
              from another cell population
        wei: Scaled weights added to the 
             transformation (not very useful)
        topk: The top k-th eigenvalues to be transformed, 
              topk = 1 by default
        idx: Indices of gene expressions to be transformed,
             transform all genes if ids=None 
    """

    if eigx is None:
        return x
    else:
        x = x.detach().clone()
        vecx, valx = eigx
        vecr, valr = eigr
        xedt = x @ vecx.float().to(x)
        if wei is None:
            wei = torch.ones_like(valx).float()
            wei[:topk] = (valr[:topk] / valx[:topk]).sqrt().float()
        xedt = wei[None].to(x) * xedt
        xedt = xedt @ vecr.T.float().to(x)
        if idx is not None:
            x[:, idx] = xedt[:, idx].clone()
            xedt = x
        xedt[xedt < 0] = 0
        return xedt


def windows_idx(h, w, sz, gap):
    # h: height (row)
    # w: width (col)
    # sz: size of the tile
    # gap: gap between tiles

    # Basic index unit
    unit = [True] * sz + [False] * gap
    out = []
    for dim in (h, w):
        # Calc the amount of tiles and remaining dims
        num, end = dim // (sz + gap), dim % (sz + gap)
        # Add the last tile if remaining > sz
        idx = [True] * sz + [False] * \
            (end - sz) if end >= sz else [False] * end
        idx = unit * num + idx
        out.append(idx)
    return out


def to_windows(img, cel, rna,
               sz, gap):
    # Assume img (H, W, C), cel (H, W), gene (H, W, G)
    c = img.shape[-1]
    h, w, g = rna.shape
    hid, wid = windows_idx(h, w, sz, gap)

    img = view_as_windows(img, (sz, sz, c), (sz + gap, sz + gap, c))
    img = np.squeeze(img, axis=2)
    hn, wn, ht, wt = img.shape[:4]
    # Convert img to (hn, wn, C, ht, wt)
    img = img.transpose(0, 1, 4, 2, 3)

    cel = view_as_windows(cel, (sz, sz), (sz + gap, sz + gap))
    cel = cel.sum((-2, -1))

    # Efficiently reduce_sum sparse rna
    rna = rna[hid][:, wid][None, :, None]
    rna = rna.reshape((hn, ht, wn, wt, g)).sum((1, 3)).todense()
    return img, cel, rna, hid, wid


def run_mat_edit(args, path, name, cond, size,
                 fov, cel, rna, cat,
                 dload, eigen, gan, mat,
                 gene_idx, step=0.5, size_o=128):
    r"""
    Create ROI video demo for GAN Inversion model, which
    bdy parameters are customized for the selected ROI image.

    Args:
        args: The parameters from args.py
        path: Path to the save folder
        fov: Morpholocial image of region of interest
        cat: Cluster annotation image of ROI
        bdy: Boundary of the image to be cropped for demo
        dload: Dataloader
        eigen: list of eigen{vector, value} of 
               compared cell (sub)populations
        model: GAN Inversion model
        gene_idx: Indices of targeted gene expressions
        step: the step length of morph transition for each video frame
        target_only: set non-targeted genes to 0 if True (not very useful)
    """

    # Get the dim info for the input data
    gap, cond = size // 4, F.one_hot(cond, 2)[None]
    fov_t, cel_t, rna_t, hid, wid = to_windows(fov, cel, rna,
                                               size, gap)
    row, col, chn = fov_t.shape[:3]

    # Get the dim info for the output data (resized)
    h_o, w_o = row * int(size_o * 1.25), col * int(size_o * 1.25)
    chn_o, gap_o = chn, size_o // 4
    if args.data_name == 'CosMx':
        chn_o -= 1
    hid_o, wid_o = windows_idx(h_o, w_o, size_o, gap_o)

    print(fov_t.shape, cel_t.shape, rna_t.shape, cond.shape, cond)

    # TODO: Noise alignment, need to be careful
    noise_fix = torch.randn((fov_t.shape[0], fov_t.shape[1], 512)).cuda()
    noise_mat = torch.randn((row, col, 512 * 4)).cuda()
    wrt = imageio.get_writer(str(path / f'{name}_demo.mp4'),
                             fps=24)
    if args.decoder == 'style2':
        noise_img = init_img_noise(gan, fov_t.shape[0], fov_t.shape[1])
    for shf in np.arange(0, 1 + step, step):
        print(shf)
        fov_edit = 255 * np.ones_like(fov).astype(np.float32)
        mat_edit = -torch.ones((chn_o, h_o, w_o)).float()
        msk = 255 * torch.ones((1, h_o, w_o))
        # fov_edit = copy.deepcopy(fov).astype(np.float32)
        if shf == 0:
            fov_bbx = copy.deepcopy(fov)
        with torch.inference_mode():
            for r in range(row):
                if args.decoder == 'style2':
                    set_img_noise(gan, fov_t.shape[1],
                                  noise_img[r])

                img, gene = prep_inp_mat(fov_t[r], rna_t[r],
                                         args.data_name, size_o)
                gene_sub = edit_gene(gene, eigen[0],
                                     eigen[1], idx=gene_idx)
                shf_sub = (1 - shf) * gene + shf * gene_sub
                out_sml = run_one_demo('GAN', args.decoder,
                                       gan, img, shf_sub, noise_fix[r],
                                       shift=shf)
                # # with bbox without gap
                # img_crop = post_output(out_sml, size).cpu().numpy()
                # if args.data_name == 'Xenium':
                #     img_crop = add_bbx(np.repeat(img_crop, 3, 1))[:, 0]
                # else:
                #     img_crop = add_bbx(img_crop)
                # img_c = img_crop.transpose((2, 0, 3, 1))
                # img_c = img_c.reshape((img_c.shape[0], -1, img_c.shape[-1]))
                # st = r * (size + gap)
                # fov_edit[st:st + size, wid] = img_c

                # collect all the tiles for inpainting
                img_o = out_sml.permute((1, 2, 0, 3))
                img_o = img_o.reshape((img_o.shape[0], img_o.shape[1], -1))
                st_o = r * int(size_o * 1.25)
                mat_edit[:, st_o:st_o+size_o, wid_o] = img_o.clamp(-1, 1).cpu()
                # msk[:, st_o:st_o+size_o, wid_o] = 0

            for r in range(row - 3):
                for c in range(col - 3):
                    sz_o = size_o * 4 + gap_o * 3
                    st_o = (r * int(size_o * 1.25), c * int(size_o * 1.25))
                    inp = mat_edit[:, st_o[0]:st_o[0] + sz_o,
                                   st_o[1]:st_o[1] + sz_o]
                    out = mat(inp[None].cuda(), None, noise_mat[r, c].unsqueeze(0).cuda(),
                              cond.cuda(), noise_mode='const')
                    out = post_output(out, int(size * 2.25)).cpu().numpy()

                    sz = size * 2 + gap
                    st = (r * int(size * 1.25), c * int(size * 1.25))
                    fov_edit[st[0]:st[0] + sz,
                             st[1]:st[1] + sz] = out[0].transpose((1, 2, 0))
                    # print(r, c, out.min(), out.max())
                # if shf == 0:
                #     # cat[r, wid] = add_bbx(cat[r, wid])
                #     if args.data_name == 'Xenium':
                #         fov_bbx[r, wid] = add_bbx(
                #             np.repeat(fov_bbx[r, c][:, :, None], 3, -1))[:, :, 0]
                #     else:
                #         fov_bbx[r, wid] = add_bbx(fov_bbx[r, c])

        fov_edit = fov_edit.astype('uint8')
        wrt.append_data(fov_edit)
        if shf == 0 or shf == 1:
            im_nm = 'rec' if shf == 0 else 'edit'
            mat_edit = ((mat_edit + 1) * 127.5).round().clamp(0, 255)
            if args.data_name == 'CosMx':
                mat_edit = torch.cat((torch.zeros(1, h_o, w_o),
                                      mat_edit))
            mat_edit = mat_edit.numpy().transpose((1, 2, 0)).astype('uint8')
            # msk = msk.numpy().transpose((1, 2, 0))[:, :, 0]
            # mat_edit = cv2.inpaint(mat_edit.astype('uint8'), msk.astype('uint8'), 3, cv2.INPAINT_TELEA)

            Image.fromarray(mat_edit).save(
                str(path / f'{name}_{im_nm}_out.jpg'))
            Image.fromarray(fov_edit).save(str(path / f'{name}_{im_nm}.jpg'))
            # if shf == 0:
            #     Image.fromarray(fov_bbx).\
            #         save(str(path / 'crop_bbx.png'))
            #     Image.fromarray(cat).\
            #         save(str(path / 'cat_bbx.png'))
    wrt.close()
    return
