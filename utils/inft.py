import cv2
import torch
import sparse
import imageio
import argparse
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from math import ceil
from torchvision import utils

from PIL import ImageFont, ImageDraw, Image
from statannotations.Annotator import Annotator
from utils.metric import run_one_moment, post_moment
ITALIC_FONT = ImageFont.truetype('utils/arialitalic.ttf', 128)


def load_model(pth, model):
    r"""
    Generate global noise

    Args:
        pth: Path to the checkpoint 
        model: Initialized GAN model
    """

    if not pth.is_file():
        print(f'{pth} is not a ckpt file, ignore')
        return False
    else:
        ckpt = torch.load(pth, map_location='cpu')
        model.load_state_dict(ckpt['g_ema'], strict=True)
        model.cuda().eval()
        return True


def get_global_n(noise_dim, batch):
    r"""
    Generate global noise

    Args:
        noise_dim: Dimension of the noise 
        batch: Batch size
    """

    # Get the global feature (noise)
    return torch.randn(1, noise_dim).repeat(batch, 1)


def get_layer_n(model, batch, cid=None,
                row_ovlp=None, col_ovlp=None,
                device='cuda', seed=None):
    r"""
    Generate layer-wise noise

    Args:
        model: Trained GAN model 
        batch: Batch size
        cid: Column id
        row_ovlp: The buffer dict storing the intermediate 
                  The 'width' amount of n-layer noise, update in each row iter
        col_ovlp: The buffer dict storing the intermediate 
                  n-layer noise, update in each width iter
        device: CPU or CUDA
        seed: Random seed for fixing the generated noise
    """

    if seed is not None:
        torch.manual_seed(seed)
    # Get the overlapped layer-wise noise for inf generation
    layer_n, locl_sz = [], model.locl_sz
    for n in range(model.num_layers):
        if n % 2 == 1:
            locl_sz = 2 * locl_sz - 5
        sz = (batch - 1) * (locl_sz - 5) + locl_sz
        ly_n = torch.randn(1, sz, sz)
        if cid is not None:
            assert cid >= 0 and isinstance(cid, int)
            if col_ovlp[n] is not None:
                ly_n[:, :, :5] = col_ovlp[n].detach().clone()
            if row_ovlp[cid][n] is not None:
                ly_n[:, :5] = row_ovlp[cid][n].detach().clone()
            col_ovlp[n] = ly_n[:, :, -5:].detach().clone()
            row_ovlp[cid][n] = ly_n[:, -5:].detach().clone()
        ly_n = ly_n.unfold(1, locl_sz, (locl_sz - 5)).\
            unfold(2, locl_sz, (locl_sz - 5))
        ly_n = ly_n.permute((1, 2, 0, 3, 4))
        # print(n, ly_n.shape)
        layer_n.append(ly_n.to(device))
    return layer_n


def run_inf_gene(gpth, gdim,
                 batch, offset,
                 roi, raw_sz,
                 seed=None, **edit_dct):
    r"""
    Process the sparse gene array, note that the editing is usually excuted on the 
    8x8-summed gene grid rather than the input 32x32-summed gene grid 

    Args:
        gpth: Either path to the sparse gene array or the sparse gene array itself
        gdim: The dimension (n-plex) of gene array
        batch: Batch size
        offset: The offset of the roi, = 0 if mouse and = 512 if CosMx lung (depracted in IST-editing)
        roi: The spatial size of the gene expression array         
        raw_size: The training tile (cell) size (=128)
        seed: Random seed for fixing the generated noise of noie and logo experiment
    """

    if gpth is None:
        gene = np.random.randn(roi, roi, gdim)
    else:
        if isinstance(gpth, str):
            gene = sparse.load_npz(gpth)
        else:
            gene = gpth
        # The first tile of each row (if exists)
        # has smaller image size, need to shift the
        # sparse pixels to create large img
        if gene.shape[1] != roi:
            print(gene.shape, roi)
            assert gene.shape[1] < roi
            crds = gene.coords
            crds[1] = crds[1] + (roi - gene.shape[1])
            gene = sparse.COO(crds, gene.data,
                              shape=(roi, roi, gene.shape[2]))
        gene = gene[offset: offset + roi,
                    offset: offset + roi, :gdim]

    gene_sz = raw_sz // 4
    # This workaround is meant for creating clearer gene visualization
    gene_s = gene[None, :, None].reshape((16 * (batch + 1), gene_sz // 4,
                                          16 * (batch + 1), gene_sz // 4, -1))
    gene_s = gene_s.sum((1, 3)) if gpth is None else \
        gene_s.sum((1, 3)).todense()
    if edit_dct:
        cof = edit_dct['cof']
        if edit_dct['edit'] == 'scale':
            gidx, shft = edit_dct['gidx'], edit_dct['shft']
            print('scale', gidx, shft, gene_s.shape[-1])
            gcoef = np.zeros(gene_s.shape[-1]) + shft
            gcoef[gidx] = cof
            gene_s = gene_s * gcoef
        elif edit_dct['edit'] == 'eigen' and edit_dct['cof'] != 1:
            eig = edit_dct['eig']
            gene_s = gene_s.reshape(((16 * (batch + 1)) ** 2, -1))
            gene_s = edit_gene_inf(torch.from_numpy(gene_s).float().to(eig[0]),
                                   eig, torch.as_tensor(cof), 100)
            gene_s = gene_s.reshape((16 * (batch + 1), 16 * (batch + 1), -1))
            gene_s = gene_s.cpu().numpy()
        elif edit_dct['edit'] in ('noise', 'logo') and edit_dct['cof'] != 1:
            if seed is not None:
                np.random.seed(seed)
            gene_n = np.random.rand(*gene_s.shape)
            gene_m = np.random.rand(16 * (batch + 1),
                                    16 * (batch + 1))
            gene_m[gene_m < 0.8] = 0
            gene_m[gene_m != 0] = 1
            if edit_dct['edit'] == 'logo':
                logo = np.array(logo.resize((logo.width * 2, logo.height * 2)))
                logo[logo == 255] = 0
                logo = logo.sum((-1))
                logo[logo != 0] = 1
                g_sz = 16 * (batch + 1)
                assert g_sz >= logo.shape[0] and g_sz >= logo.shape[1]
                logo = np.pad(logo, ((g_sz - logo.shape[0], 0),
                                     (g_sz - logo.shape[1], 0)))
                gene_m = gene_m * logo
            gene_n = gene_n * gene_m[:, :, None]
            gene_s = gene_n * (1 - cof) + gene_s * cof

    # Here, we always assume 8 X 8 gene tiles with img_sz // 4 resolution
    gene = gene_s[:, None, :, None].reshape((4 * (batch + 1), 4,
                                             4 * (batch + 1), 4, -1))
    gene = gene.sum((1, 3))
    # # Here, we always assume 8 X 8 gene tiles with img_sz // 4 resolution
    # gene = gene[None, :, None].reshape((4 * (batch + 1), gene_sz,
    #                                     4 * (batch + 1), gene_sz, -1))
    # gene = gene.sum((1, 3)) if gpth is None else gene.sum((1, 3)).todense()
    # assert (gene_s == gene).all()
    gene = torch.from_numpy(gene.transpose((2, 0, 1)))
    gene = gene.unfold(1, 8, 4).unfold(2, 8, 4)
    gene = gene.permute((1, 2, 3, 4, 0)).contiguous()
    return gene_s, gene


def run_inf_img(model, glob_n, layer_n,
                gpth, offset, batch,
                raw_sz, save_pth, debug,
                value_range=(-1, 1),
                seed=None, **edit_dct):
    r"""
    Generate a cropped bioimage, usually with a 2048x2048 resolution in the mouse study 

    Args:
        model: The trained GAN model
        glob_n: The global noise
        layer_n: The layer-wise noise
        gpth: Either path to the sparse gene array or the sparse gene array itself
        offset: The offset of the roi, = 0 if mouse and = 512 if CosMx lung (depracted in IST-editing)
        batch: Batch size
        raw_sz: The training tile (cell) size (=128)
        save_pth: Path to the saved output
        debug: Check the boundary consistency if True else
               run the roi image generation
        value_range: The value range of output image
        seed: random Seed for fixing the generated noise of noie and logo experiment
    """

    roi = raw_sz * (batch + 1)
    gene_s, gene = run_inf_gene(gpth, model.gene_dim,
                                batch, offset,
                                roi, raw_sz,
                                seed, **edit_dct)
    # Sometimes we require double precision
    # for data and model to check the overlap
    gene = gene.type(glob_n.dtype).to(glob_n)

    with torch.inference_mode():
        sample = []
        for b in range(batch):
            noise = [ln[b].contiguous() for ln in layer_n]
            smp = model([glob_n], gene[b], noise=noise,
                        randomize_noise=False, debug=debug)[0]
            sample.append(smp)
        sample = torch.cat(sample, 0)
        if sample.shape[1] == 2:
            sample = torch.cat([-torch.ones_like(sample[:, 0])[:, None],
                                sample], 1)
    if debug:
        _h, _w = sample.shape[-2:]
        sample = sample.reshape((batch, batch, -1, _h, _w))
        for b in range(1, batch):
            print(b, (sample[b, :, :, :5] ==
                      sample[b - 1, :, :, -5:]).all())
            print(b, (sample[:, b, :, :, :5] ==
                      sample[:, b - 1, :, :, -5:]).all())
    else:
        utils.save_image(
            sample, save_pth,
            nrow=batch,
            normalize=True,
            value_range=value_range,
            padding=0)
        return gene_s[batch // 2:-batch // 2, batch // 2:-batch // 2]


def align_raw_img(gpth, offset, batch, raw_sz,
                  save_pth, resolution=128):
    r"""
    Output the aligned raw image to the generated one 

    Args:
        gpth: Path to the sparse gene array
        offset: The offset of the roi, 0 if mouse and 512 if CosMx lung (depracted in IST-editing)
        batch: Batch size
        raw_sz: The training tile (cell) size (=128 in IST-eidting)
        save_pth: Path to the saved output
        resolution: The input image resolution
    """

    if 'CosMx' in gpth:
        img_pth = gpth.replace('GeneLabels', 'CellComposite')
        img_pth = img_pth.replace('.npz', '.jpg')
    elif 'Xenium' in gpth:
        img_pth = gpth.replace('_rna.npz', '.jpg')
        img_sty = 'hne' if 'Breast' in gpth else 'dapi'
        img_pth = img_pth.replace('rna', img_sty)
    st = offset + raw_sz // 2
    ed = raw_sz * batch + st
    img = Image.open(img_pth).crop((st, st, ed, ed))
    if raw_sz != resolution:
        img = img.resize((resolution * batch,
                          resolution * batch))
    img.save(save_pth)


def run_inf_metric(args, path, dload,
                   model, mfeat, glob_fix,
                   total, repeat=4):
    r"""
    Compute the cell-level metrics (deprecated)

    Args:
        args: Arguments that are implemented in args.py file
        path: Path to the intermediate output image
        dload: Dataloader (deprecated, may not be compatible to the latest dataset impl)
        model: Trained GAN model
        mfeat: The dict of feature extractor for calc FID
        glob_fix: Whether to fix the global noise
        total: The total amount of cell-level data
        repeat: The amount of times for repeating the calc
    """

    for rid in range(repeat):
        tot = 0
        avg = dict({m: [0,] for m in mfeat})
        scm = dict({m: [0,] for m in mfeat})

        if glob_fix:
            glob_n = get_global_n(model.style_dim, args.n_eval).cuda()

        for lid, (img, gene, label) in enumerate(dload):
            if tot >= total:
                print(f'Metric calc done for {args.decoder} GAN.')
                break

            if not glob_fix:
                glob_n = get_global_n(model.style_dim, img.shape[0]).cuda()

            # layer_n = get_layer_n(model, img.shape[0])
            # layer_n = [ln.cuda() for ln in layer_n]
            with torch.inference_mode():
                out = model([glob_n], gene.cuda(),
                            # noise=layer_n,
                            randomize_noise=True)[0]

                # Create fake images
                if lid < 5:
                    outi = (torch.cat([img.cuda(), out]) + 1) / 2
                    if outi.shape[1] == 2:
                        outi = torch.cat((torch.zeros_like(outi[:, 0])[:, None], outi),
                                         dim=1)
                    utils.save_image(outi,
                                     str(path / f'{rid}_{lid}.png'),
                                     nrow=args.n_eval,
                                     padding=2)

                if (lid + 1) % 1000 == 0:
                    print(lid)

                # Prepare image for stats calc
                b = min(total - tot,  gene.shape[0])
                run_one_moment([out], mfeat,
                               avg, scm)
                tot += b

        assert tot == total
        post_moment(rid, path, total,
                    avg, scm)
    return


def run_inf_vid(pth, subs, task,
                nam, coefs, org_dct,
                idx=0):
    r"""
    Generate the ensemble of cell-, tissue- and animal-level of mouse image with or without 
    gene expression image

    Args:
        pth: Root to the image materials required to be assembled 
             and to the saved output image
        subs: Subset of generated mouse image tiles
        task: Task name for generating different parts of figure 1 and 2
        nam: Folder name of required image materials (usually 'Merg')
        coefs: The scaling factors for different editing strategies
        org_dct: The organ dict determined in the 'config.py'
        idx: The index of gene to be illustrated in the output image
    """

    clt, ckp = pth.parent.name.split('_')[-2], pth.stem
    wsi_crd, wsi_sz = '1_53_0_26', 512 * 6
    wsi_nam = f'{subs}_{task}_{wsi_crd}_{ckp}_{clt}'

    if 'all' in org_dct:
        del org_dct['all']
    for cid, cof in enumerate(coefs):
        save_pth = pth / f'{cid}_VID'
        save_pth.mkdir(parents=True, exist_ok=True)
        wsi_pth = pth / f'{nam}_{cid}_D'
        wsi_img = cv2.imread(str(wsi_pth / f'{wsi_nam}.jpg'))
        frac = wsi_img.shape[0] / wsi_sz
        wsi_img = cv2.resize(wsi_img,
                             (ceil(wsi_img.shape[1] / frac), wsi_sz))
        print(cid, cof, wsi_img.shape)

        d_pth = pth / f'{nam}_{cid}_D'
        g_pth = pth / f'{nam}_{cid}_G'
        img_pth = [(d_pth, '_cells'), (d_pth, '_bbox'), (g_pth, '')]
        lft, rgt = [], []
        for org, ovl in org_dct.items():
            o_h, o_w = ovl['h'], ovl['w']
            roi_crd = f'{o_h[0]}_{o_h[1]}_{o_w[0]}_{o_w[1]}'
            roi_nam = f'{subs}_{task}_{roi_crd}_{ckp}_{clt}'
            img_lst = []
            for p in img_pth:
                img = cv2.imread(str(p[0] / f'{roi_nam}{p[1]}.jpg'))
                # Add gene name to the left corner of the image
                if not p[1]:
                    img_pil = Image.fromarray(img)
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((32, 32), ovl['glst'][idx],
                              font=ITALIC_FONT, fill=(255, 255, 255))
                    img = np.array(img_pil)
                img_lst.append(img)
            if ovl['vpos'] == 'right':
                img_lst = list(reversed(img_lst))
            img = np.concatenate(img_lst, axis=1)

            if ovl['vpos'] == 'left':
                lft.append(img)
            elif ovl['vpos'] == 'right':
                rgt.append(img)
            cv2.imwrite(str(save_pth / f'{org}.jpg'), img)

        lft = np.concatenate(lft, axis=0)
        cv2.imwrite(str(save_pth / 'left.jpg'), lft)
        rgt = np.concatenate(rgt, axis=0)
        cv2.imwrite(str(save_pth / 'right.jpg'), rgt)
        out = np.concatenate((lft, wsi_img, rgt), axis=1)
        cv2.imwrite(str(save_pth / 'out.jpg'), out)

    # The code snippet for video generation,
    # which requires smaller steps to generate coefs list

    # print('Create video')
    # wrt = imageio.get_writer(str(pth / 'out.mp4'),
    #                          fps=24)
    # for cid, cof in enumerate(coefs):
    #     print(cid, cof)

    #     save_pth = pth / f'{cid}_VID'
    #     wrt.append_data(np.array(Image.open(str(save_pth / 'out.jpg'))))
    # wrt.close()


def run_raw_vid(pth, subs, task, org_dct, idx=0):
    r"""
    Generate the ensemble of cell-, tissue- and animal-level of raw mouse image with or without 
    gene expression image

    Args:
        pth: Root to the image materials required to be assembled 
             and to the saved output image
        subs: Subset of generated mouse image tiles
        task: Task name for generating different parts of figure 1 and 2
        org_dct: The organ dict determined in the 'config.py'
        idx: The index of gene to be illustrated in the output image
    """

    wsi_crd, wsi_sz = '1_53_0_26', 512 * 6
    wsi_nam = f'{subs}_{task}_{wsi_crd}_raw_11'

    if 'all' in org_dct:
        del org_dct['all']
    save_pth = pth / 'VID'
    save_pth.mkdir(parents=True, exist_ok=True)
    wsi_pth = pth / 'MERG'
    wsi_img = cv2.imread(str(wsi_pth / f'{wsi_nam}.jpg'))
    frac = wsi_img.shape[0] / wsi_sz
    wsi_img = cv2.resize(wsi_img,
                         (ceil(wsi_img.shape[1] / frac), wsi_sz))
    print(wsi_img.shape)

    d_pth = pth / 'MERG'
    img_pth = [(d_pth, '_cells'), (d_pth, '_bbox')]
    # This part need to get from vid generation
    g_pth = pth / 'MERG_G'
    if g_pth.is_dir():
        img_pth += [(g_pth, '')]
    lft, rgt = [], []
    for org, ovl in org_dct.items():
        o_h, o_w = ovl['h'], ovl['w']
        roi_crd = f'{o_h[0]}_{o_h[1]}_{o_w[0]}_{o_w[1]}'
        roi_nam = f'{subs}_{task}_{roi_crd}_raw_11'
        img_lst = []
        for p in img_pth:
            img = cv2.imread(str(p[0] / f'{roi_nam}{p[1]}.jpg'))
            # # Add gene name to the left corner of the image
            if not p[1]:
                img_pil = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pil)
                draw.text((32, 32), ovl['glst'][idx],
                          font=ITALIC_FONT, fill=(255, 255, 255))
                img = np.array(img_pil)
            img_lst.append(img)
        if ovl['vpos'] == 'right':
            img_lst = list(reversed(img_lst))
        img = np.concatenate(img_lst, axis=1)

        if ovl['vpos'] == 'left':
            lft.append(img)
        elif ovl['vpos'] == 'right':
            rgt.append(img)
        cv2.imwrite(str(save_pth / f'{org}.jpg'), img)

    lft = np.concatenate(lft, axis=0)
    cv2.imwrite(str(save_pth / 'left.jpg'), lft)
    rgt = np.concatenate(rgt, axis=0)
    cv2.imwrite(str(save_pth / 'right.jpg'), rgt)
    out = np.concatenate((lft, wsi_img, rgt), axis=1)
    cv2.imwrite(str(save_pth / 'out.jpg'), out)


def edit_gene_inf(x, eig, coef,
                  topk=1, idx=None):
    r"""
    Get edited gene expressions using eigenvalue manipulation

    Args:
        x: Gene expression array to be edited
        eig: Eigenbasis and eigenvalue of the scm 
        coef: Scaling factor
        topk: Top k eigenvalues to be changed
        org_dct: The organ dict determined in the 'config.py'
        idx: The index list of genes to be changed, 
             while the rest remain unchanged
    """

    if eig is None or coef == 1:
        return x
    else:
        x = x.detach().clone()
        eigvec, eigval = eig
        xedt = x @ eigvec.float().to(x)
        wei = torch.ones_like(eigval).float()
        wei[:topk] = torch.sqrt(coef)
        xedt = wei[None].to(x) * xedt
        xedt = xedt @ eigvec.T.float().to(x)
        if idx is not None:
            x[:, idx] = xedt[:, idx].clone()
            xedt = x
        xedt[xedt < 0] = 0
    return xedt


def run_cellpose(pth, cmp=[-1, 0, 1, 2], mtype='cyto'):
    r"""
    Calc cellular morphological features using cellpose

    Args:
        pth: Path to the output csv file
        cmp: The compared roi images: e.g., -1 usually is ground truth.
             0, 1, 2 are rather folder names than real scaling factors.
        mtype: The cell segmentation model
    """

    from cellpose import plot as cplt
    from cellpose import models as cmodels
    cpose = cmodels.Cellpose(gpu=True, model_type=mtype)

    org = ('colon', 'hair', 'kidney', 'lung')
    name = ['Nuclei count', 'size', 'Nuclei ratio', 'DAPI expr']
    (pth / mtype).mkdir(parents=True, exist_ok=True)
    for o in org:
        cell_outs = {}
        for c in cmp:
            cell_outs[c] = {n: [] for n in name}
            for i in range(2):
                img = pth / str(c) / f'{o}_{i}.jpg'
                img = cv2.imread(str(img))
                if len(img.shape) == 3:
                    img = img[:, :, 0]
                img = img[1024:3072, 1024:3072]
                masks, flows, styles, diams = cpose.eval(img.astype(np.float32) / 255.,
                                                         diameter=None, normalize=False, channels=[0, 0])
                fig = plt.figure(figsize=(12, 5))
                cplt.show_segmentation(fig, img, masks,
                                       flows[0], channels=[0, 0])
                plt.tight_layout()
                plt.savefig(str(pth / mtype / f'{o}_{i}_{c}.png'),
                            bbox_inches='tight', dpi=200)
                plt.close()

                cid, cnt = np.unique(masks, return_counts=True)
                cnt = cnt[1:]
                cell_outs[c]['Nuclei count'].append(len(cnt))
                cell_outs[c]['size'] += cnt.tolist()
                cell_outs[c]['Nuclei ratio'].\
                    append(masks[masks != 0].size / masks.size)
                cell_outs[c]['DAPI expr'].append(img.mean())
                print(masks.shape, len(cid), len(cnt), cid[:4], cnt[:4])

        df = pd.DataFrame()
        for c in cmp:
            rdct = {'coef': c}
            for nm in name:
                val = cell_outs[c]
                if nm == 'Nuclei count':
                    cell_outs[c][nm] = np.sum(val[nm])
                elif nm == 'size':
                    cell_outs[c]['Nuclei size'] = np.mean(val['size'])
                    cell_outs[c]['Nuclei size (std)'] = np.std(val['size'])
                    if c != -1:
                        cell_outs[c]['Nuclei size'] /= cell_outs[-1]['Nuclei size']
                        cell_outs[c]['Nuclei size (std)'] /= cell_outs[-1]['Nuclei size (std)']
                else:
                    cell_outs[c][nm] = np.mean(val[nm])
                if c != -1 and nm != 'size':
                    cell_outs[c][nm] /= cell_outs[-1][nm]

                if nm != 'size':
                    rdct[nm] = cell_outs[c][nm]
                else:
                    for nam in ('Nuclei size', 'Nuclei size (std)'):
                        rdct[nam] = cell_outs[c][nam]
            df = df.append(rdct,
                           ignore_index=True)
        df.to_csv(str(pth / mtype / f'{o}.csv'))


def edit_plot_inf(path, glst,
                  prefix,
                  gene_all,
                  gene_sub,
                  dpi=6, hue_order=['0.5', '1 (GT)', '2']):
    r"""
    Generate the plot of gene expressions before and after editing

    Args:
        path: Path to the save folder
        glst: The list of edited genes
        prefix: Subset name added to the output plot name (if exists)
        gene_all: List of all gene expressions
        gene_sub: List of genes that output plots
        dpi: Image output resolution
        hue_order: The hue name list (Decpreated, 
                   the hue names are added manully in the macos numbers)
    """

    font = {'weight': 'bold',
            'size': 32}
    plt.rc('font', **font)

    for gnm in gene_sub:
        gid = gene_all.index(gnm)
        expr, cat = [], []
        for i, gn in enumerate(glst):
            g = gn[:, gid]
            gmin, glow, ghigh, gmax = np.quantile(g, [0, 0.05, 0.95, 1])
            expr.append(g[(g >= glow) & (g <= ghigh)])
            print(i, gmin, glow, ghigh, gmax,
                  g.shape, expr[-1].shape)
            cat += [hue_order[i]] * len(expr[-1])
        x = [gnm] * len(cat)
        expr = np.concatenate(expr)
        df = pd.DataFrame({'x': x, 'cat': cat,
                           'Transcript counts': expr})

        print(df.head())
        _, ax = plt.subplots(1, 1, figsize=(5, 4.5), dpi=dpi * 100)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        flierprops = dict(markerfacecolor='purple',
                          markeredgecolor='purple', alpha=0.15,
                          markersize=8,
                          linestyle='none')
        bx = sns.boxplot(ax=ax, data=df, x='x', y='Transcript counts',
                         hue='cat', hue_order=hue_order,
                         palette=sns.color_palette("rocket"),
                         linewidth=4, flierprops=flierprops)
        bx.set_xticklabels(bx.get_xticklabels())
        for lab in bx.get_xticklabels():
            # Here it sets all italic.
            lab.set_style("italic")

        if 'eigen' in prefix:
            plt.ylim(top=20)
        elif gnm in ('Calm4', 'Sbsn') and 'naive' in prefix:
            plt.ylim(top=10)

        pair = [((gnm, hue_order[0]), (gnm, hue_order[1])),
                ((gnm, hue_order[1]), (gnm, hue_order[2]))]
        annot = Annotator(ax, pair, data=df, x='x', y='Transcript counts',
                          hue='cat',
                          hue_order=hue_order)
        annot.configure(test='Mann-Whitney',
                        comparisons_correction=None,
                        text_format='star',
                        loc='outside',
                        verbose=2)
        annot.apply_test()
        annot.annotate()
        plt.xlabel(None)
        plt.legend([], [], frameon=False)
        plt.savefig(str(path / f'{prefix}_{gnm}.png'), bbox_inches='tight')
        plt.close()


def gen_heatmap(pth, df, is_full, id):
    r"""
    Output the heatmap plot for gene expressions

    Args:
        pth: Path to the save folder
        df: The dataframe for creating the heatmap
        is_full: Output the full list of gene expression if True else
                 partial list of high expressed genes
        id: The id of the gene list, 
            which is meant for the sperated full gene list
    """

    font = {'weight': 'bold',
            'size': 16}
    plt.rc('font', **font)
    plt.figure(figsize=(5, 25 if is_full else 35))
    ht = sns.heatmap(df,
                     yticklabels=True, xticklabels=True)
    ht.set_yticklabels(ht.get_yticklabels())
    for lab in ht.get_yticklabels():
        # Here it sets all italic.
        lab.set_style('italic')
        lab.set_size(8 if is_full else 16)
        lab.set_weight('bold')
    plt.savefig(str(pth / f'heat_{id}.png'),
                bbox_inches='tight', dpi=600)
    plt.close()


if __name__ == '__main__':
    from style3.models.stylegan2.model_inf import Generator
    parser = argparse.ArgumentParser(description='StyleGAN2 trainer')
    parser.add_argument('--data', type=str, help='name of the dataset')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    noise_dim, batch = 512, 4

    if args.data == 'CosMx':
        gene_num, img_chn = 1000, 2
        raw_sz, offset = 160, 512
    elif args.data == 'Xenium':
        gene_num, img_chn = 392, 1
        raw_sz, offset = 96, 0
    elif args.data == 'Xenium_brain':
        gene_num, img_chn = 319, 1
        raw_sz, offset = 128, 0
    elif args.data == 'Xenium_breast':
        gene_num, img_chn = 280, 3
        raw_sz, offset = 128, 0

    model = Generator(128, gene_num, noise_dim, 3, 8, img_chn=img_chn)
    model.eval().cuda().double()
    glob_n = get_global_n(noise_dim, batch).cuda().double()
    layer_n = get_layer_n(model, batch)
    layer_n = [ln.cuda().double() for ln in layer_n]
    run_inf_img(model, glob_n, layer_n,
                None, offset,
                batch, raw_sz, f'./0_{raw_sz}_merge.jpg',
                debug=args.debug, value_range=(0, 255))
