import os
import sys
import cv2
import torch
import random
import sparse
import pickle
import pyvips
import numpy as np

from PIL import Image
from pathlib import Path

from args import parse_args
from style3.models.stylegan2.model_inf import Generator

from utils.common import add_bbx
from Dataset_inf.config import cfg_mouse
from utils.inft import get_global_n, get_layer_n, run_inf_img, \
    load_model, run_raw_vid, run_inf_vid


sys.path.append('.')


def setup_seed(seed):
    r"""
    Args:
        seed: Seed for reproducible randomization.

    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def proc_raw(img, pstem, roi):
    r"""
    Process the raw roi image such that overlapped boundaries
      are removed, e.g., 2176 X 2176 --> 2048 X 2048

    Args:
        img: Tile image
        pstem: Name of the tile including boundary coord info
        roi: Size of the tile image
    """

    crd = pstem.split('_')
    shfh = int(crd[0]) - int(crd[4])
    shfw = int(crd[2]) - int(crd[6])
    return img[shfh:shfh + roi:, shfw:shfw + roi]


def append_img(lst, pth):
    r"""
    Append the raw or generated tile image to the list

    Args:
        lst: The list of tiled images
        pth: Path to the tile image to be appended
    """

    lst.append(pyvips.Image.new_from_file(str(pth),
                                          access='sequential'))
    return


def run_raw_crop(raw_pth, out_pth, bbx_clr,
                 roi, txt,
                 add_info=False):
    r"""
    Generate the processed raw tile image, while keeping 
    image pixel value unchanged. This step is meant for the 
    following up raw WSI stitching

    Args:
        raw_pth: Path to the raw tile image
        out_pth: Path to the outputted raw tile image for WSI stitching
        bbx_clr: bounding box colors of regions that are zoomed in for 
                 better visualization 
        roi: Size of outputted tile image
        txt: Text that is added to the raw tile image
        add_info: Add text if True
    """

    if not out_pth.is_file():
        raw = cv2.imread(str(raw_pth))
        raw = proc_raw(raw, raw_pth.stem, roi)
        if bbx_clr is not None:
            # 4 parms of bbx_clr means that we attach the small bbox
            # corresponding to checkerboard-12
            if len(bbx_clr) == 4:
                _h, _w = raw.shape[:2]
                slc = (slice(_h // 2 - bbx_clr[-1], _h // 2 + bbx_clr[-1]),) + \
                      (slice(_w // 2 - bbx_clr[-1], _w // 2 + bbx_clr[-1]),)
                raw[slc] = add_bbx(raw[slc], bbx_clr[0],
                                   bbx_clr[1], bbx_clr[2])
            else:
                raw = add_bbx(raw, bbx_clr[0], bbx_clr[1], bbx_clr[2])
        if add_info:
            assert raw.shape[0] == roi and raw.shape[1] == roi
            # assign pos to each tile for determining val data
            cv2.putText(raw, txt,
                        (0, roi // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        20, (0, 0, 255), 16)
        cv2.imwrite(str(out_pth), raw)


def run_gen_crop(out_pth, bbx_clr, sm_roi,
                 args, cfg,
                 model, glob_n, layer_n,
                 gene, gene_lst,
                 seed=None, **edit_dct):
    r"""
    Generate a cropped tile image, usually with a 2048x2048 resolution in the mouse study 

    Args:
        out_pth: Path to the saved output
        bbx_clr: bounding box colors of regions that are zoomed in for 
                 better visualization
        sm_roi: The size of checkboarder tile image, especifically for checkerboard-12 
                with sm_roi = 512 (true size is sm_roi * 2)
        args: Arguments that are implemented in args.py file
        cfg: Arguments that are implemented in config.py file
        model: The trained GAN model
        glob_n: The global noise
        layer_n: The layer-wise noise
        gene: The sparse gene array
        gene_lst: The highly expressed gene list for regions highlighted with bbox
        seed: random Seed for fixing the generated noise
    """

    if not out_pth.is_file():
        gene_s = run_inf_img(model, glob_n, layer_n,
                             gene, cfg.offset, args.n_eval,
                             cfg.raw_sz, str(out_pth), False,
                             seed=seed, **edit_dct)

        if not (bbx_clr is None and sm_roi is None):
            img = cv2.imread(str(out_pth))
            if gene_lst is not None:
                gene_s = gene_s[:, :, edit_dct['gidx'][0]]
                gene_s = np.clip(gene_s, 0, 2) / 2
                gene_s = np.stack((gene_s * bbx_clr[0][0],
                                   gene_s * bbx_clr[0][1],
                                   gene_s * bbx_clr[0][2]), -1)
                gene_s = add_bbx(gene_s, cfg.neu_clr,
                                 bbx_clr[1] // 8, bbx_clr[2])
                gene_pth = str(out_pth).replace('_D', '_G')
                cv2.imwrite(gene_pth, gene_s.astype(np.uint8))
            if bbx_clr is not None:
                img = add_bbx(img, bbx_clr[0], bbx_clr[1], bbx_clr[2])
            if sm_roi is not None:
                _h, _w = img.shape[:2]
                slc = (slice(_h // 2 - sm_roi[0], _h // 2 + sm_roi[0]),) + \
                    (slice(_w // 2 - sm_roi[0], _w // 2 + sm_roi[0]),)
                img[slc] = add_bbx(img[slc], sm_roi[1], sm_roi[2], sm_roi[3])
            cv2.imwrite(str(out_pth), img)


def run_one_crop(args, cfg, pth,
                 cid, h, w,
                 glob_n, model, save_pth,
                 row_ovlp, col_ovlp, gene_lst,
                 draw_bbx=True, **edit_dct):
    r"""
    Output a generated and raw tile image, usually with a 2048x2048 resolution in the mouse study 

    Args:
        args: Arguments that are implemented in args.py file
        cfg: Arguments that are implemented in config.py file
        pth: Path to the gene array file
        cid: The id for the scaling factor,
        h: Row id
        w: Column id 
        glob_n: The global noise
        model: The trained GAN model
        save_pth: Path to the saved output
        row_ovlp: The buffer dict storing the intermediate 
                  The 'width' amount of n-layer noise, update in each row iter
        col_ovlp: The buffer dict storing the intermediate 
                  n-layer noise, update in each width iter
        gene_lst: The highly expressed gene list for regions highlighted with bbox
        draw_bbx: Draw the bbox and ready for generating cell- tissue- and animal-levle assembled image if True
    """

    layer_n = get_layer_n(model, args.n_eval, w,
                          row_ovlp, col_ovlp,
                          seed=h * 26 + w)

    bbox, raw_clr = cfg.pths_arr['vis'][h][w], None
    if isinstance(pth, Path):
        # Process gene
        gene = sparse.load_npz(str(pth))[:, :, :args.gene_num]
        # Output raw and gene roi images
        assert args.n_eval == 16
        out_pth = str(save_pth / '{}' / f'{h}_{w}.jpg')
        if bbox is not None:
            # Neutral color for input gene or image
            if draw_bbx:
                raw_clr = (cfg.neu_clr, bbox['blen'], bbox['bpos'])
            raw_pth = pth.parent.parent / args.stain / f'{pth.stem[:-4]}.jpg'
            if cid == 0:
                run_raw_crop(raw_pth, Path(out_pth.format('CROP_D')),
                             raw_clr, cfg.roi, f'{h} {w}', False)

        # Output all generated images
        glst = None
        gen_clr, sml_clr, trn_clr = None, None, cfg.pths_arr['trn_clr'][h][w]
        if bbox is not None:
            glst = [gene_lst.index(g) for g in bbox['gene']]
            if draw_bbx:
                gen_clr = (bbox['clr'], bbox['blen'], bbox['bpos'])
        # if trn_clr is not None:
            # if cfg.sm_roi is not None:
            #     sml_clr = (cfg.sm_roi, trn_clr[0], trn_clr[1], trn_clr[2])
            # else:
            #     gen_clr = trn_clr

        run_gen_crop(Path(out_pth.format(f'CROP_{cid}_D')),
                     gen_clr, sml_clr, args, cfg,
                     model, glob_n, layer_n,
                     gene, glst, seed=h + w * 52,
                     **edit_dct)


def post_crop(crop_pth, strp_pth, blak_pth,
              w_st, w_ed, h, org):
    r"""
    Output the intermediate stripe image for a give row 

    Args:
        crop_pth: Path to the generated tile images
        strp_pth: Path to the output stripe image
        blak_pth: Path to the empty image that fills in the tile 
                  image without any gene expressions
        w_st: The starting column id
        w_ed: The ending column id
        h: The row id
        org: The name of organs (e.g., 'all')
    """

    crops = []
    for w in range(w_st, w_ed):
        pth = crop_pth / f'{h}_{w}.jpg'
        if pth.is_file():
            append_img(crops, pth)
        else:
            append_img(crops, blak_pth)
    wsi_crop = pyvips.Image.arrayjoin(crops,
                                      across=w_ed - w_st)
    wsi_crop.write_to_file(str(strp_pth / f'{h}_{org}.tif'),
                           compression='jpeg', tile=True)


def post_strp(args, org, o_h, o_w,
              roi, pth, out_pth,
              subs, task, nitr, frac,
              vis_sz=None, cid=None, clr=None):
    r"""
    Output the generated WSI

    Args:
        args: Arguments that are implemented in args.py file
        org: The name of organs (e.g., 'all')
        o_h: List of start and end row id
        o_w: List of start and end col id
        roi: The size of tile image (deprecated)
        pth: Path to the intermediate strip images
        out_pth: Path to the generated WSI
        subs: Name of the subset such as checkerboader-n
        task: Name of running task for outputing raw or generated WSI
        nitr: Number of iterations
        frac: Fraction of training data
        vis_sz: Size of cell-level visualization (256)
        cid: List of zoomed in cellular regions
        clr: List of color
    """

    print('start wsi')
    strp = [pyvips.Image.new_from_file(str(pth / f'{h}_{org}.tif'),
                                       access='sequential')
            for h in range(o_h[0], o_h[1])]
    wsi = pyvips.Image.arrayjoin(strp, across=1)
    print('write to file')

    wsi_nam = f'{subs}_{task}_{o_h[0]}_{o_h[1]}_{o_w[0]}_{o_w[1]}'
    if 'video' in task:
        clt = Path(args.ckpt_path).parent.name.split('_')[-1]
        wsi_nam = f'{wsi_nam}_{nitr[:-3]}_{clt}'
    else:
        wsi_nam = f'{wsi_nam}_raw_11'

    # Get the raw tif image with multiple pages
    # This is for QuPath examination
    wsi_pth = str(out_pth / wsi_nam)
    wsi.write_to_file(f'{wsi_pth}.tif',
                      pyramid=True, bigtiff=True, tile=True,
                      tile_height=256, tile_width=256,
                      compression='jpeg')

    # Get downscaled jpg for stitching the comphrehensive image
    print('write to jpg')
    page = int(np.log2(frac)) if '_G' not in out_pth.stem else 0
    wsi = pyvips.Image.new_from_file(f'{wsi_pth}.tif',
                                     page=page)
    wsi.write_to_file(f'{wsi_pth}.jpg')

    # Get the raw organ (tissue) image with 'jpg', this is for visual comparison
    if 'bbox' not in task and org != 'all':
        # don't generate checker-12 raw images without bbox
        if subs == 'all' or 'raw' not in task:
            tis = pyvips.Image.new_from_file(f'{wsi_pth}.tif',
                                             page=0)
            tis_pth = str(out_pth / f'{org}')
            tis.write_to_file(f'{tis_pth}.jpg')

    if cid is not None and '_G' not in out_pth.stem:
        print('write to bbox and cell')
        cells = []
        img_l = cv2.imread(f'{wsi_pth}.tif')
        img_s = cv2.imread(f'{wsi_pth}.jpg')
        for (h, w) in cid:
            slc_l = (slice(h * vis_sz, (h + 1) * vis_sz),) + \
                (slice(w * vis_sz, (w + 1) * vis_sz),)

            if clr is not None:
                img_l[slc_l] = add_bbx(img_l[slc_l], clr, 8)
                slc_s = (slice(h * vis_sz // frac, (h + 1) * vis_sz // frac),) + \
                    (slice(w * vis_sz // frac, (w + 1) * vis_sz // frac),)
                img_s[slc_s] = add_bbx(img_s[slc_s], clr, 4)

            cells.append(img_l[slc_l])

        if clr is not None:
            cv2.imwrite(f'{wsi_pth}_bbox.tif', img_l)
            cv2.imwrite(f'{wsi_pth}_bbox.jpg', img_s)

        assert len(cells) == 8
        cells = np.stack(cells)
        bat, hsz, wsz, chn = cells.shape
        cells = cells[None].reshape((bat // 2, 2, hsz, wsz, chn))
        cells = cells.transpose((0, 2, 1, 3, 4))
        cells = cells.reshape((-1, 2 * wsz, chn))
        cv2.imwrite(f'{wsi_pth}_cell.jpg', cells)
        cells = Image.fromarray(cells)
        (cel_w, cel_h) = (cells.width // 2, cells.height // 2)
        # always use cv2 to save image for keeping chn consistency
        cv2.imwrite(f'{wsi_pth}_cells.jpg',
                    np.array(cells.resize((cel_w, cel_h))))


def init_analysis(args, step=1):
    r"""
    Initialize the analysis parms

    Args:
        args: Arguments that are implemented in args.py file
        step: Step for generating the video frames (unused)
    """

    assert args.data_name == 'Xenium_mouse'
    root = Path(f'Data/Xenium_mouse')
    subs, orgs, task, crop = args.task.split('_')
    # Init gene list
    with open(str(root / 'GAN/crop/transcripts.pickle'), 'rb') as fp:
        gene_lst = pickle.load(fp)
    # Gene coefficent (only for video)
    coefs = list(np.arange(0, 2 + step, step))

    # Init the region of interest for all tasks
    frc = 0
    if 'checkerboard' in subs or 'random' in subs:
        subs, frc = subs.split('-')
    cfg = cfg_mouse(root, subs, orgs, float(frc), 0)

    # Init the model, iterations for metric and video tasks
    model, i_iter, glob_n = None, None, None
    if task in ('metric', 'feat') or 'video' in task:
        model = Generator(128, args.gene_num, 512, 3, 8, 2,
                          img_chn=args.input_nc)
        i_iter = list(range(100000, 900000, 100000)) if args.n_iter is None \
            else [args.n_iter]
        i_iter = [f'{str(i).zfill(6)}.pt' for i in i_iter]
        if 'video' in task:
            assert args.n_iter is not None
            i_iter = i_iter[0]
        if task == 'feat' or 'video' in task:
            # Create global feat (raw, video)
            glob_n = get_global_n(model.style_dim, args.n_eval).cuda()
    else:
        assert 'raw' in task
        # assert task in (
        #     'cmp', 'feat-out') or 'gene' in task or 'raw' in task or 'radar' in task

    if 'raw' in task or 'video' in task:
        if args.input_nc == 1:
            empty = np.zeros((cfg.roi, cfg.roi, 3), dtype=np.uint8)
        else:
            empty = np.ones((cfg.roi, cfg.roi, 3), dtype=np.uint8) * 255
        # Create a empty tile (with 3 channels to avoid pyvips stitch issue)
        cv2.imwrite(str(args.save_path / 'empty.jpg'), empty)

    tsk_parm = (root, subs, frc, orgs, task, crop, cfg)
    ifr_parm = (model, i_iter, glob_n, gene_lst, coefs)
    return tsk_parm, ifr_parm


def main(args):
    r"""
    Main function for generating the WSI results
    """

    tsk_parm, ifr_parm = init_analysis(args)
    root, subs, frc, orgs, task, crop, cfg = tsk_parm
    model, i_iter, glob_n, gene_lst, coefs = ifr_parm
    print(tsk_parm[:5])
    # print(i_iter, glob_n.shape, len(gene_lst), coefs)

    if 'raw' in task:
        save_pth, org = args.save_path, 'all'
        for subf in ('CROP', 'STRP', 'MERG'):
            (save_pth / f'{subf}').mkdir(parents=True, exist_ok=True)

        # del cfg.org_dct['all']
        for org, oval in cfg.org_dct.items():
            o_h, o_w = oval['h'], oval['w']
            for h in range(o_h[0], o_h[1]):
                for w in range(o_w[0], o_w[1]):
                    pth = cfg.pths_arr['wsi'][h][w]
                    if pth is None:
                        continue
                    print(h, w, pth)

                    raw_pth = pth.parent.parent / \
                        args.stain / f'{pth.stem[:-4]}.jpg'
                    out_pth = save_pth / 'CROP' / f'{h}_{w}.jpg'
                    if subs == 'all':
                        clr = None
                        if 'bbox' in task:
                            clr = cfg.pths_arr['vis'][h][w]
                            if clr is not None:
                                clr = [cfg.neu_clr, clr['blen'], clr['bpos']]
                        run_raw_crop(raw_pth, out_pth, clr,
                                     cfg.roi, f'{h} {w}', False)
                    else:
                        clr = cfg.pths_arr['trn_clr'][h][w]
                        if clr is not None:
                            clr = [clr[0], clr[1], clr[2]]
                            if cfg.sm_roi is not None:
                                clr += [cfg.sm_roi]
                            run_raw_crop(raw_pth, out_pth, clr,
                                         cfg.roi, f'{h} {w}', False)

                post_crop(save_pth / 'CROP',
                          save_pth / 'STRP',
                          args.save_path / 'empty.jpg',
                          o_w[0], o_w[1], h, org)

            o_frc = 16 if org == 'all' else 8
            if org != 'all' and subs == 'all' and 'bbox' in task:
                vis_sz = cfg.vis_sz
                o_cid = oval['cid']
                o_clr = cfg.neu_clr
            else:
                vis_sz, o_cid, o_clr = None, None, None
            post_strp(args, org, o_h, o_w, cfg.roi,
                      save_pth / 'STRP',
                      save_pth / 'MERG',
                      subs, task, i_iter, o_frc,
                      vis_sz, o_cid, o_clr)

        # To include gene plot, obtain it from
        # the editing func when coef = 1
        if subs == 'all' and 'bbox' in task:
            run_raw_vid(save_pth, subs, task, cfg.org_dct)
    elif 'video' in task:
        assert load_model(args.ckpt_path / i_iter, model)
        save_pth = args.save_path / i_iter
        draw_bbx = True if 'bbox' in task else False
        # gene_idx may not work well
        edit_mtd, gene_num = task.split('-')[1].lower(), 4
        if 'tif' in task:
            # only need to output generated tif
            coefs = [1, ]
        elif edit_mtd == 'noise':
            coefs = [0, 0.5, 1]
        elif edit_mtd == 'scale':
            coefs, shft = [0.5, 1, 2], 0
        elif edit_mtd == 'eigen':
            coefs = [0.1, 0.5, 1]
            eig_dct = torch.load('stats_inf/organ_eig.pt')
        if 'tif' not in task:
            # make sure that outputted tif don't have bbox
            if not draw_bbx or edit_mtd in ('eigen', 'scale'):
                del cfg.org_dct['all']
        for subf in ('CROP', 'STRP', 'MERG'):
            (save_pth / f'{subf}_D').mkdir(parents=True, exist_ok=True)

        for cid, cof in enumerate(coefs):
            for subf in (f'CROP_{cid}', f'STRP_{cid}', f'MERG_{cid}'):
                (save_pth / f'{subf}_D').mkdir(parents=True, exist_ok=True)
                (save_pth / f'{subf}_G').mkdir(parents=True, exist_ok=True)

            for org, ovl in cfg.org_dct.items():
                print(cof, org, ovl)
                o_h, o_w = ovl['h'], ovl['w']
                row_dct = {f'{cid}_D': True,
                           f'{cid}_G': org != 'all',
                           'D': org != 'all' and cid == 0}
                row_ovlp = {w: {n: None for n in range(model.num_layers)}
                            for w in range(o_w[0], o_w[1])}
                for h in range(o_h[0], o_h[1]):
                    print(h)
                    col_ovlp = {n: None for n in range(model.num_layers)}
                    for w in range(o_w[0], o_w[1]):
                        # Only gen the tile when run the wsi
                        # or run the rois
                        if org == 'all' or 'all' not in cfg.org_dct:
                            # Note that when run through only roi,
                            # the boundary noise may not match
                            edit_dct = {'edit': edit_mtd, 'cof': cof}
                            vis_dct = cfg.pths_arr['vis'][h][w]
                            if vis_dct is not None:
                                gnm = vis_dct['gene']
                                glst = [gene_lst.index(g) for g in gnm]
                                edit_dct['gidx'] = glst[:gene_num]
                            crop_pth = cfg.pths_arr['wsi'][h][w]
                            if edit_mtd == 'scale':
                                edit_dct['shft'] = shft
                            elif edit_mtd == 'eigen':
                                edit_dct['eig'] = eig_dct[org[:-2]]
                            run_one_crop(args, cfg, crop_pth,
                                         cid, h, w,
                                         glob_n, model, save_pth,
                                         row_ovlp, col_ovlp, gene_lst,
                                         draw_bbx, **edit_dct)

                    for nm, is_valid in row_dct.items():
                        if is_valid:
                            post_crop(save_pth / f'CROP_{nm}',
                                      save_pth / f'STRP_{nm}',
                                      args.save_path / 'empty.jpg',
                                      o_w[0], o_w[1], h, org)

                o_cid = ovl['cid']
                o_frc = 16 if org == 'all' else 8
                for (nm, is_valid) in row_dct.items():
                    if is_valid:
                        o_clr = ovl['clr'] if nm == f'{cid}_D' else cfg.neu_clr
                        post_strp(args, org, o_h, o_w, cfg.roi,
                                  save_pth / f'STRP_{nm}',
                                  save_pth / f'MERG_{nm}',
                                  subs, task, i_iter, o_frc,
                                  cfg.vis_sz, o_cid, o_clr)
        if 'bbox' in task and edit_mtd not in ('eigen', 'scale'):
            run_inf_vid(save_pth, subs, task, 'MERG', coefs,
                        cfg.org_dct, 0)


if __name__ == '__main__':
    args = parse_args()
    setup_seed(args.seed)
    main(args)
