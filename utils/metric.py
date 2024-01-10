import torch
import cellpose
import numpy as np
import pandas as pd
import seaborn as sns
import cellpose.io as cio
import matplotlib.pyplot as plt
import torchvision.utils as tv_utils

from cellpose import plot as cplt
from torch.linalg import svdvals, eigvals, svd

from criteria.psnr import PSNR
from criteria.ms_ssim import MS_SSIM

from utils.edit import edit_gene
from utils.feat import fn_resize
from utils.common import run_GAN, run_GANI, prep_input, prep_outim


def get_eigen(scm):
    r"""
    Calc the eigenvalue and eigenvector of a given SCM

    Args:
        scm: Sample covariance matrix of gene expressions 
    """
    return svd(scm, full_matrices=False)[:2]


def run_one_GAN(analysis, decoder, eigen,
                model, img, gene, gene_idx):
    r"""
    Run the GAN (Inversion) model once to generate 
    one batch of generated and edited morphological images

    Args:
        analysis: The model used for analysis: GAN or GAN Inversion
        decoder: StyleGAN2 generator
        eigen: list of gene eigen{vector, value} of 
               compared cell (sub)populations
        model: GAN or GAN Inversion
        img: morphological image
        gene: one-dim gene expression vector
        gene_idx: Indices of targeted gene expressions
    """

    with torch.inference_mode():
        if analysis == 'GAN':
            noise = torch.randn((gene.shape[0], 512)).to(gene)
            out = run_GAN(decoder, model, gene, noise)
            out1 = run_GAN(decoder, model,
                           edit_gene(gene, eigen[0],
                                     eigen[1], idx=gene_idx),
                           noise)
        elif 'GANI' in analysis:
            out = run_GANI(decoder, model, gene, img)
            if analysis == 'GANIM':
                out1 = out
            else:
                out1 = run_GANI(decoder, model,
                                edit_gene(gene, eigen[0],
                                          eigen[1], idx=gene_idx),
                                img, shift=1)
        out, out1 = out.clamp(-1, 1), out1.clamp(-1, 1)
    return out, out1


def run_one_moment(out, mfeat,
                   avg, scm):
    r"""
    Calc first and second statistical moments for 
    one batch of generated and edited morphological images

    Args:
        out: The list of generated and edited image batch
        mfeat: Feature extractor for calc d_FID
        avg: The dict of summed mean values without normalization
        scm: The dict of summed SCM without normalization
    """

    with torch.inference_mode():
        for i, o in enumerate(out):
            if len(o.shape) == 3:
                o = o[None]
            if o.shape[1] == 2:
                o = torch.cat((-torch.ones_like(o[:, 0])[:, None], o), dim=1)
            for mn, mod in mfeat.items():
                oim = prep_outim(o)
                feat = mod(fn_resize(oim,
                                     299 if mn == 'clean' else 224,
                                     mn)).double()
                avg[mn][i] += feat.sum(0)
                scm[mn][i] += feat.T @ feat


def post_moment(i, path, total,
                avg, scm):
    r"""
    Postprocess first and second statistical moments for 
    one batch of generated and edited morphological images,
    then we obtain the mean, cov and eigenvalue

    Args:
        i: The i-th repeat of computation
        path: Path to the save folder
        total: The total amount of data for normalization
        avg: The list of summed mean values without normalization
        scm: The list of summed SCM without normalization
    """

    for mn in avg:
        stat = f'{i}_{mn}'
        for edt in range(len(avg[mn])):
            if edt == 1:
                stat += '_edit'
            m, s, eigval = get_cmp_stat(avg[mn][edt] / total,
                                        scm[mn][edt] / total)
            torch.save((m, s, eigval), str(path / f'{stat}.pt'))


def run_metric(args, path, dload, eigen,
               model, cpose, mfeat,
               total, repeat=4, gene_idx=None):
    r"""
    Calc the quantification results for generated and reconstructed
    images driven by non-edited and edited gene expressions.

    Args:
        args: The parameters from args.py
        path: Path to the save folder
        dload: Dataloader
        eigen: List of gene eigen{vector, value} of 
               compared cell (sub)populations
        model: GAN (Inversion) model
        mfeat: Feature extractor for d_FID calc
        total: The total amount of data
        repeat: The repeat time of d_FID calc
        gene_idx: Indices of targeted gene expressions
    """

    if args.analysis == 'GANIM':
        met_fuc, met_all = prep_met(met_chn=args.input_nc)
    else:
        met_fuc, met_all = None, None
        cell_outs = {i: [] for i in range(3)}

    for rid in range(repeat):
        tot = 0
        avg = dict({m: [0, 0] for m in mfeat})
        scm = dict({m: [0, 0] for m in mfeat})
        for lid, ((img, gene), _, _) in enumerate(dload):
            if tot >= total:
                print(f'Metric calc done for {args.decoder} GAN.')
                break

            img, gene = prep_input(img, gene, args.gene_num,
                                   is_GAN=args.analysis == 'GAN')

            out = run_one_GAN(args.analysis, args.decoder, eigen,
                              model, img, gene, gene_idx)

            # Create fake images
            if lid < 5:
                outi = (torch.cat([img, out[0], out[1]]) + 1) / 2
                if outi.shape[1] == 2:
                    outi = torch.cat((torch.zeros_like(outi[:, 0])[:, None], outi),
                                     dim=1)
                tv_utils.save_image(outi,
                                    str(path / f'{rid}_{lid}.png'),
                                    nrow=args.n_eval,
                                    padding=2)

            if (lid + 1) % 1000 == 0:
                print(lid)

            # Prepare image for stats calc
            b = min(total - tot,  gene.shape[0])
            if args.analysis == 'GANIM':
                run_one_met(img, out[0],
                            met_fuc, met_all)
            else:
                run_one_moment(out, mfeat,
                               avg, scm)
                cell_metric(cpose, [img, out[0], out[1]],
                            path, rid, lid, cell_outs)
            tot += b

        assert tot == total

        if args.analysis == 'GANIM':
            post_met(path / '.pt', met_all)
        else:
            post_moment(rid, path, total,
                        avg, scm)
            post_cell(cell_outs, path)

    return


def get_freq(array, exclude=0):
    count = np.bincount(array[array != exclude])
    if count.size == 0:
        return exclude
    else:
        return np.argmax(count)


def cell_metric(cpose, imgs, path,
                rid, lid, cell_outs,
                sz=16):
    for iid, img in enumerate(imgs):
        img = prep_outim(img)
        cell = img[:, -1]
        b, h, w = cell.shape
        # Convert batch of cell images to a larger image
        cell = cell[:, None].reshape((b // 4, 4, h, w))
        cell = cell.permute((0, 2, 1, 3)).reshape((-1, 4 * w))
        cell = cell.cpu().numpy()
        masks, flows, styles, diams = cpose.eval(cell,
                                                 diameter=None, channels=[0, 0])
        # # Convert the larger image back to batch of cell images
        # sbmsk = masks[None, :, None].copy()
        # sbmsk = sbmsk.reshape((b // 4, h, 4, w))
        # sbmsk = sbmsk.transpose((0, 2, 1, 3)).reshape((b, h, w))
        # sbmsk = sbmsk[:,
        #               h // 2 - sz:h // 2 + sz,
        #               w // 2 - sz:w // 2 + sz]
        # # Get the cell id that centered in the image
        # ccd = np.apply_along_axis(lambda x: get_freq(x),
        #                           axis=-1, arr=sbmsk.reshape((b, -1)))
        # masks[~np.isin(masks, ccd)] = 0
        # cnvex, solid, cmpct = cellpose.utils.get_mask_stats(masks)
        area = masks[masks != 0].size / masks.size
        cid, cnt = np.unique(masks, return_counts=True)
        if cid[0] == 0:
            cnt = cnt[1:]
        exprd = float(img[:, -1].mean().cpu().numpy())
        if len(cnt) == 0:
            out = [0, 0, exprd / 255.]
        else:
            out = [area, np.std(cnt), exprd / 255.]
        if img.shape[1] == 2:
            exprb = float(img[:, 0].mean().cpu().numpy())
            out.append(exprb / 255.)
        cell_outs[iid].append(out)
        if lid < 10:
            print(iid, out)
            fig = plt.figure(figsize=(12, 5))
            cplt.show_segmentation(fig, cell, masks, flows[0], channels=[0, 0])
            plt.tight_layout()
            plt.savefig(str(path / f'{rid}_{lid}_{iid}.png'))
            plt.close()


def post_cell(outs, path,
              name=['nuclei area', 'nuclei variance', 'DAPI expression', 'CD298/B2M expression']):
    df = pd.DataFrame()
    for key, val in outs.items():
        stt = np.array(val).mean(0)
        dct = {name[i]: stt[i] for i in range(len(stt))}
        df = df.append(dct, ignore_index=True)
    df.to_csv(str(path / 'cell.csv'))


def prep_met(met_key=('psnr', 'ssim'), met_chn=3):
    r"""
    Initialize the PSNR and SSIM metrics for quantifying
    image recon quality

    Args:
        met_key: The key (name) of metrics
        met_chn: The amount of image channels  
    """

    met_fuc, met_all = dict(), dict()
    mchn = 1 if met_chn == 1 else met_chn + 1
    for met in met_key:
        met = met.lower()
        if met == 'psnr':
            met_fuc[met] = PSNR()
        elif met == 'ssim':
            # kernel size = 7 for image res 128x128
            met_fuc[met] = MS_SSIM(1., False, 7, channel=met_chn)
        else:
            raise NameError(f'{met} is not a valid metric.')

        # here we should use list comprehension to avoid
        # list memory sharing issue
        met_all[met] = [[] for _ in range(mchn)]
    return met_fuc, met_all


def run_one_met(input,
                output,
                met_fuc,
                met_all):
    r"""
    Calc the PSNR and SSIM metrics for one batch of 
    reconstructed images

    Args:
        input: Input image batch
        output: Output image batch
        met_fuc: The PSNR and SSIM func dict
        met_all: The dict of metric results
    """

    input = prep_outim(input)
    output = prep_outim(output)
    # print(input.shape, output.shape)
    for mkey, mfuc in met_fuc.items():
        mchn = len(met_all[mkey])
        for chn in range(mchn):
            cn = chn
            if chn == mchn - 1 and chn > 0:
                cn = list(range(chn))
            metric = mfuc(input[:, cn],
                          output[:, cn]).detach()
            metric = metric.cpu().numpy().tolist()
            met_all[mkey][chn].extend(metric)


def post_met(pth, met_all):
    r"""
    Save the PSNR and SSIM results

    Args:
        pth: Path to the save folder
        met_all: The dict of metric results
    """

    for mkey, mlst in met_all.items():
        for mchn in range(len(mlst)):
            stat_pth = str(pth).replace('.pt', f'{mkey}_{mchn}.pt')
            torch.save(torch.tensor(mlst[mchn]), stat_pth)


def calc_img_dist(path, refs):
    r"""
    Calc several stats including d_FID score

    Args:
        path: Path to folder storing non_edit and edited stats
        refs: Stats of the reference cell image collection
    """

    for rid, ref in refs.items():
        stats = list(path.glob(f'*{rid}*.pt'))
        if not stats:
            print(f'stat files are missing in {path}')
            continue

        with open(str(path / f'stats_{rid}.txt'), 'w') as f:
            for snm in ('non_edit', 'edit'):
                if snm == 'non_edit':
                    pts = [str(s) for s in stats if 'edit' not in str(s.stem)]
                else:
                    pts = [str(s) for s in stats if 'edit' in str(s.stem)]
                if not pts:
                    continue
                for (prfx, rf) in (('same', ref[0]), ('comp', ref[1])):
                    mr, sr, eigr = rf
                    # image stats
                    l_eigs, d_fids, d_eigs, d_leas = [], [], [], []
                    # message stats
                    m_fids, m_eigs, m_leas = '', '', ''
                    for p in pts:
                        m, s, eig = torch.load(p)
                        l_eigs.append(float(eig[0].cpu().numpy()))
                        d_fids.append(calc_d_fid(m, mr, s, sr))
                        m_fids += f'{d_fids[-1].cpu().numpy():.2f}, '
                        d_eigs.append(calc_d_eig(eig, eigr) * 10)
                        m_eigs += f'{d_eigs[-1].cpu().numpy():.2f}, '
                        d_leas.append(calc_d_lea(eig, eigr) * 100)
                        m_leas += f'{d_leas[-1].cpu().numpy():.2f}, '

                    stdi, meani = torch.std_mean(torch.FloatTensor(l_eigs))
                    stdf, meanf = torch.std_mean(torch.FloatTensor(d_fids))
                    stde, meane = torch.std_mean(torch.FloatTensor(d_eigs))
                    stdl, meanl = torch.std_mean(torch.FloatTensor(d_leas))

                    prfx = f'{snm}_{prfx}_'
                    l_eigs = [f'{l:.2f}' for l in l_eigs]
                    f.write(f'{prfx}leig: {l_eigs}, leig_ref: {eigr[0]:.2f}, '
                            f'm:{meani:.2f}, s: {stdi:.2f} \n')
                    f.write(f'{prfx}fid:{m_fids}m:{meanf:.2f}, s:{stdf:.2f}\n')
                    f.write(f'{prfx}eig:{m_eigs}m:{meane:.2f}, s:{stde:.2f}\n')
                    f.write(f'{prfx}lea:{m_leas}m:{meanl:.2f}, s:{stdl:.2f}\n')
                    f.write('\n')
                f.write('\n\n\n')
    return


def calc_img_reco(path):
    r"""
    Calc the std and mean of 
    PSNR and SSIM of reconstructed images

    Args:
        path: Path to the PNSR and SSIM results tensor
    """

    with open(str(path / 'psnr.txt'), 'w') as f:
        pts = path.glob('psnr_*.pt')
        for p in pts:
            chn = p.stem.split('_')[-1]
            psnr = torch.load(str(p))
            stdp, meanp = torch.std_mean(torch.FloatTensor(psnr))
            ssim = torch.load(str(path / f'ssim_{chn}.pt'))
            stds, means = torch.std_mean(torch.FloatTensor(ssim))
            f.write(f'(psnr_{chn}) mean:{meanp:.2f}, std:{stdp:.2f}\n')
            f.write(f'(ssim_{chn}) mean:{means:.2f}, std:{stds:.2f}\n')
            f.write('\n')
    return


def get_cmp_stat(m, scm):
    r"""
    Compute the mean, cov and eigenvalue

    Args:
        m: Mean
        scm: Sample covariance matrix
    """

    s = scm - m[:, None] @ m[:, None].T
    eigval = svdvals(scm)
    eigval[eigval < 0] = 0
    return m, s, eigval


def _d_novel(sigma1, sigma2):
    r"""
    The core and more efficient impl of d_FID

    Args:
        sigma1: Covariance of one image collection
        sigma2: Covariance of compared image collection
    """

    eigval = eigvals(sigma1 @ sigma2)
    eigval = eigval.real
    eigval[eigval < 0] = 0
    return 2 * eigval.sqrt().sum()


def calc_d_fid(mu1, mu2, sigma1, sigma2):
    r"""
    The Function of d_FID calc

    Args:
        mu1: Mean of one image feat collection
        mu2: Mean of compared image feat collection
        sigma1: Covariance of one image feat collection
        sigma2: Covariance of compared image feat collection
    """

    mu1 = torch.atleast_1d(mu1)
    mu2 = torch.atleast_1d(mu2)

    sigma1 = torch.atleast_2d(sigma1)
    sigma2 = torch.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    fid_easy = diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2)
    fid_hard = _d_novel(sigma1, sigma2)
    fid = fid_easy - fid_hard
    return fid


def calc_d_eig(eig0, eig1, topk=5):
    r"""
    The Function of d_Eig calc:
    sorted eigenvalue comparison

    Args:
        eig0: Sorted eigenvalues of one image feat collection
        eig1: Sorted eigenvalues of compared image feat collection
        topk: The top k eigenvalues used for d_Eig calc
    """

    assert topk > 0 and isinstance(topk, int)
    dif = (eig0.sqrt() - eig1.sqrt())[:topk]
    return dif.dot(dif)


def calc_d_lea(eig0, eig1, topk=5):
    r"""
    The Function of d_LEA calc:
    relative sorted eigenvalue comparison

    Args:
        eig0: sorted eigenvalues of one image feat collection
        eig1: sorted eigenvalues of compared image feat collection
        topk: The top k eigenvalues used for d_LEA calc
    """

    assert topk > 0 and isinstance(topk, int)
    dif = (eig0.sqrt() - eig1.sqrt())[:topk]
    dif /= eig1.sqrt()[:topk]
    return dif.dot(dif)


def run_gene_counts(path, gene_expr, gene_name, gene='', plot=''):
    print(gene_expr.shape, len(gene_name), gene)
    if not gene:
        tot_trans = gene_expr.astype(np.int).sum(1)
        tot_genes = np.count_nonzero(gene_expr, 1)
    else:
        idx = gene_name.index(gene)
        tot_trans = gene_expr[:, idx].astype(np.int)
        tot_genes = np.count_nonzero(tot_trans[:, None], 1)

    gnm = '' if not gene else gene + ' '
    _, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 6))
    x0 = f'Number of {gnm}genes per cell'
    g0 = sns.histplot(
        data=pd.DataFrame({x0: tot_genes}),
        x=x0, binwidth=5 if not gene else 0.2,
        ax=ax0)
    ax0.set_ylabel('Number of cells (in thousands)')
    ylabels = ['{:,.1f}'.format(x) for x in g0.get_yticks() / 1000]
    g0.set_yticklabels(ylabels)

    x1 = f'Number of {gnm}transcripts per cell'
    g1 = sns.histplot(
        data=pd.DataFrame({x1: tot_trans[tot_trans < 10000]}),
        x=x1, binwidth=5 if not gene else 2,
        ax=ax1)
    ax1.set_ylabel('')
    ylabels = ['{:,.1f}'.format(x) for x in g1.get_yticks() / 1000]
    g1.set_yticklabels(ylabels)

    plt.tight_layout()
    plt.savefig(str(path / f'{plot}.png'),
                dpi=300)
    plt.close()


if __name__ == '__main__':
    for crop in ('tile', ):
        pth = f'stats_inf/xenium_mouse_{crop}'
        m_all, s_all = torch.load(f'{pth}_clean.pt')[:2]
        for i in (14, 12, 1, 3, 5, 8):
            for dat in ('trn', 'val'):
                m, s = torch.load(f'{pth}_{i}_{dat}_clean.pt')[:2]
                print(f'{i}_{dat}:', calc_d_fid(m_all, m, s_all, s))
