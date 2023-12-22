import sys
import cv2
import math
import zarr
import pyvips
import pickle
import random
import sparse
import shutil
import argparse
import colorsys
import itertools
import multiprocessing

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scanpy as sc
import pyarrow.compute as pc
import pyarrow.parquet as pq

from math import ceil
from pathlib import Path
from random import shuffle
from PIL import Image, ImageFile
from tifffile import imread, imwrite


from Dataset_inf.utils_mouse import _um_to_pixel, _df_to_roi

ImageFile.LOAD_TRUNCATED_IMAGES = True
# suppress chained assignment warning
pd.options.mode.chained_assignment = None
sys.path.append('.')


def _random_color(size, num=128):
    r"""
    Obtain a list of randomised color for visual testing 
        the processed data 
    
    Args:
        size: The amount of color to be selected
        num: The amount of candidate color

    """

    h = np.random.rand(num)
    l = 0.4 + np.random.rand(num) / 5.0
    s = 0.5 + np.random.rand(num) / 2.0
    rl, gl, bl = list(), list(), list()
    for n in range(num):
        r, g, b = colorsys.hls_to_rgb(h[n], l[n], s[n])
        rl.append(max(int(255 * r), 100))
        gl.append(max(int(255 * g), 100))
        bl.append(max(int(255 * b), 100))
    rn = np.array(random.choices(rl, k=size))
    rn[0] = 0  # assign black to loc without cell_id
    gn = np.array(random.choices(gl, k=size))
    gn[0] = 0
    bn = np.array(random.choices(bl, k=size))
    bn[0] = 0
    return rn, gn, bn


def _roi_to_coord(h, w,
                  hei, wid,
                  roi, ovlp=200):
    r"""
    Get the coord list of the full-size image and the one
        without overlapped boundaries
    
    Args:
        h: Row id
        w: Column id
        hei: Height of the WSI
        wid: Width of the WSI
        roi: Size of the tile image without overlapped boundaries
        ovlp: Size of the overlapped boundaries

    """

    crd = [h * roi, (h + 1) * roi,
           w * roi, (w + 1) * roi]
    crd = np.clip(crd,
                  [0, 0, 0, 0],
                  [hei, hei, wid, wid])

    crdo = [h * roi - ovlp, (h + 1) * roi + ovlp,
            w * roi - ovlp, (w + 1) * roi + ovlp]
    crdo = np.clip(crdo,
                   [0, 0, 0, 0],
                   [hei, hei, wid, wid])
    return crd.tolist(), crdo.tolist()


def prep_roi_img(img_pth, out_pth, roi, ovlp=200):
    r"""
    Get the tile images with overlap from the WSI
    
    Args:
        img_pth: Path to the WSI
        out_pth: Path to outputted tile images
        roi: Size of the tile image
        ovlp: Size of the overlapped boundaries

    """

    # (108670, 53991)
    img = imread(str(img_pth))
    # high = np.max(img) 11061
    extm = [0, 11061]
    bdry = [0, 11061]

    is_rgb = len(img.shape) == 3
    hei, wid = img.shape[-2], img.shape[-1]
    h_num, w_num = ceil(hei / roi), ceil(wid / roi)
    print(h_num, w_num)
    for h in range(h_num):
        for w in range(w_num):
            print(h, w)
            crd, crdo = _roi_to_coord(h, w, hei, wid, roi, ovlp)
            print(crd, crdo, crd+crdo)
            if is_rgb:
                if img.shape[0] == 4:
                    # This occurs because the merge of dapi and he WSI after
                    # alignment using qupath Java script
                    img_c = img[1:, crdo[0]: crdo[1], crdo[2]: crdo[3]]
                elif img.shape[0] == 3:
                    img_c = img[:, crdo[0]: crdo[1], crdo[2]: crdo[3]]
                img_c = img_c.transpose((1, 2, 0))
                img_c = cv2.cvtColor(img_c, cv2.COLOR_RGB2BGR)
            else:
                img_c = img[crdo[0]: crdo[1], crdo[2]: crdo[3]]

                img_c = np.clip(img_c.astype(np.float16), extm[0], extm[1])
                img_c = (img_c - bdry[0]) / (bdry[1] - bdry[0])
                img_c = (img_c * 255).astype(np.uint8)
            roi_nm = '_'.join(map(str, crd + crdo))
            cv2.imwrite(str(out_pth / f'{roi_nm}.jpg'),
                        img_c)


def prep_roi_logo(img_pth, out_pth, roi, ovlp=200, frac=16):
    r"""
    Get the tile images with overlap from the CTP logo image
    created with macos keynote
    
    Args:
        img_pth: Path to the WSI
        out_pth: Path to outputted tile images
        roi: Size of the tile image
        ovlp: Size of the overlapped boundaries
        frac: Down-scale factor when cropping the tile image,
            as the createdlogo image is smaller than WSI

    """


    # (108670, 53991)
    img = imread(str(img_pth))[:, :, :3]
    print(img.shape)
    # high = np.max(img) 11061

    hei, wid = img.shape[0] * frac, img.shape[1] * frac
    print(img.shape, hei, wid)
    h_num, w_num = ceil(hei / roi), ceil(wid / roi)
    print(h_num, w_num)
    for h in range(h_num):
        for w in range(w_num):
            print(h, w)
            crd, crdo = _roi_to_coord(h, w, hei, wid, roi, ovlp)
            print(crd, crdo, crd+crdo)
            img_c = img[crdo[0] // frac: crdo[1] // frac,
                        crdo[2] // frac: crdo[3] // frac]
            img_c = cv2.cvtColor(img_c, cv2.COLOR_RGB2BGR)
            roi_nm = '_'.join(map(str, crd + crdo))
            cv2.imwrite(str(out_pth / f'{roi_nm}.jpg'),
                        img_c)


def prep_roi_rna(df, bid, masks, crd, crdo,
                 out_pth, rna_axs, color=None):
    r"""
    Get the tile images with overlap from the gene expression array
        with the same gigapixel resolution as the WSI
    
    Args:
        bid: The full list of cell id derived from 
            the Barcode in the raw table
        masks: The gigapixel raw array stored the nucleis and cell masks
        crd: The coord list of the tile image without overlap
        crdo: The coord list of the full resolution tile image
        out_pth: Path to the outputted sparse arrays
        rna_axs: The dict with gene name as the key and id as the value
        color: Output the cell and nucleus masks for visual examination if not None

    """

    roi_nm = '_'.join(map(str, crd + crdo))
    df = _df_to_roi(df, crdo, crdo, '{}_location')
    nucl_roi = masks[0][crdo[0]: crdo[1], crdo[2]: crdo[3]]
    cell_roi = masks[1][crdo[0]: crdo[1], crdo[2]: crdo[3]]

    if df.empty:
        if not (np.all(nucl_roi == 0) and np.all(cell_roi == 0)):
            print(f'none gene expression but exist cells for {roi_nm}')
        return

    df.feature_name = df.feature_name.map(rna_axs)
    df = df.groupby(list(df.columns), as_index=False).size()
    y = df.y_location.values
    x = df.x_location.values
    z = df.feature_name.values
    rna_coo = sparse.COO((y, x, z), df['size'].values,
                         shape=list(nucl_roi.shape) + [len(rna_axs)])

    if color is not None:
        img_n = color[0][nucl_roi]
        img_c = color[1][cell_roi]
        rna_np = rna_coo.sum(axis=-1).todense()
        img_rna = color[2][rna_np]

        cv2.imwrite(str(out_pth / f'{roi_nm}.jpg'),
                    np.stack([img_n, np.zeros_like(img_n), img_rna], axis=-1))
        # cv2.imwrite(str(out_pth / f'{roi_nm}_n.jpg'), img_n)
        # cv2.imwrite(str(out_pth / f'{roi_nm}_c.jpg'), img_c)
        # cv2.imwrite(str(out_pth / f'{roi_nm}_rna.jpg'), img_rna)
    else:
        msk_np = np.stack((nucl_roi, cell_roi), axis=-1)
        cell_id = np.unique(msk_np)
        assert len(cell_id) > 1
        assert cell_id[0] == 0 and (cell_id[1:] != 0).all()
        # fliter out the cells not in the cluster table
        diff = np.setdiff1d(cell_id[1:], bid)
        if len(diff) != 0:
            # remove all the cells without labels
            msk_np = np.where(np.isin(msk_np, diff), 0, msk_np)
        msk_coo = sparse.COO.from_numpy(msk_np)
        sparse.save_npz(str(out_pth / f'{roi_nm}_msk'), msk_coo)
        sparse.save_npz(str(out_pth / f'{roi_nm}_rna'), rna_coo)
        print(len(diff), roi_nm, len(df), rna_coo.shape)


def test_roi_cell(pth, cid, bid):
    cell = sparse.load_npz(pth).data
    cell_id = np.unique(cell)
    diff0 = np.setdiff1d(cell_id, cid)
    diff1 = np.setdiff1d(cell_id, bid)
    if len(diff0) != 0 or len(diff1) != 0:
        print(set(diff0) == set(diff1), len(diff0) - len(diff1), pth)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Crop single-cell images out of the large NanoString image.')
    parser.add_argument('--root',
                        type=Path,
                        default=Path('Data/Xenium_mouse'),
                        help='Path to NanoString dataset.')
    parser.add_argument('--core',
                        type=int,
                        default=8,
                        help='Number of cores used for image processing.')
    parser.add_argument('--roi_size',
                        type=int,
                        default=2048,
                        help='Size of cropped image region.')
    parser.add_argument('--roi_ovlp',
                        type=int,
                        default=64,
                        help='Overlap of cropped image region.')
    parser.add_argument('--prep_roi_img',
                        action='store_true',
                        help='crop image region with interest')
    parser.add_argument('--prep_roi_logo',
                        action='store_true',
                        help='crop image region with interest')
    parser.add_argument('--prep_roi_rna',
                        action='store_true',
                        help='Prepare roi of gene expr')
    parser.add_argument('--test_roi_cell',
                        action='store_true',
                        help='Prepare roi of gene expr')
    parser.add_argument('--debug',
                        action='store_true')
    parser.add_argument('--stain',
                        type=str,
                        default='dapi',
                        help='Overlap of cropped image region.')

    args = parser.parse_args()

    if args.prep_roi_img:
        if args.stain == 'dapi':
            img_pth = args.root / 'outs' / 'morphology_mip.ome.tif'
        else:
            img_pth = args.root / 'mouse_align.ome.tif'
        out_pth = args.root / args.stain
        out_pth.mkdir(parents=True, exist_ok=True)
        prep_roi_img(img_pth, out_pth,
                     args.roi_size, args.roi_ovlp)

    if args.prep_roi_logo:
        img_pth = args.root / 'outs' / 'logo.tiff'
        out_pth = args.root / 'logo'
        out_pth.mkdir(parents=True, exist_ok=True)
        prep_roi_logo(img_pth, out_pth,
                      args.roi_size, args.roi_ovlp)

    if args.prep_roi_rna:
        rna_col = ['cell_id', 'y_location', 'x_location', 'feature_name']
        with multiprocessing.Pool(processes=args.core) as pool:
            prep_args = list()
            meta_pth = args.root / 'outs'
            if args.debug:
                out_pth = args.root / 'debug_rna'
            else:
                out_pth = args.root / 'rna'

            out_pth.mkdir(parents=True, exist_ok=True)

            # all the cluster should have the same cell ids
            clt_pth = args.root / 'outs' / 'analysis' / 'clustering' / \
                f'gene_expression_graphclust'
            dfc = pd.read_csv(str(clt_pth / 'clusters.csv'))
            dfm = pq.read_table(str(args.root / 'outs' / 'nucleus_boundaries.parquet'),
                                columns=['cell_id', 'vertex_y', 'vertex_x']).to_pandas()
            # calc the centroid of each cell based on nucleus boundary
            # minimum 7 vertices
            dfm = dfm.groupby(['cell_id'],
                              as_index=False)[['vertex_y', 'vertex_x']].mean()
            print('all cells', len(dfm))
            _dct = dict(zip(dfm.cell_id, dfm.index.values + 1))
            dfc.Barcode = dfc.Barcode.map(_dct)
            bid = dfc.Barcode.values
            print('all cells with label', len(dfc))
            del dfm

            with zarr.ZipStore(str(meta_pth / 'cells.zarr.zip'), mode='r') as cstore:
                adata = sc.read_10x_h5(filename=str(meta_pth / 'cell_feature_matrix.h5'),
                                       gex_only=False)
                rna_dct = dict(adata.var['feature_types'])
                rna_axs = {k: i for i, k in enumerate(rna_dct)}
                del adata

                df_rna = pq.read_table(str(meta_pth / 'transcripts.parquet'),
                                       filters=[('cell_id', '!=', 'UNASSIGNED'),
                                                ('qv', '>=', 20)],
                                       columns=rna_col).to_pandas()
                # During the training, we use the labels stored in cell.parquet
                # instead of the conflicted cell_id and overlaps* stored in transcripts.parquet
                df_rna = df_rna.drop(columns=['cell_id'])
                _um_to_pixel(df_rna, '{}_location')

                masks = zarr.group(store=cstore, overwrite=False).masks
                hei, wid = masks[0].shape[-2], masks[0].shape[-1]
                h_num = ceil(hei / args.roi_size)
                w_num = ceil(wid / args.roi_size)
                print(h_num, w_num)

                color = None
                if args.debug:
                    # cell num < 1000000
                    color = _random_color(1000000)

                for h in range(h_num):
                    for w in range(w_num):
                        crd, crdo = _roi_to_coord(h, w, hei, wid,
                                                  args.roi_size, args.roi_ovlp)
                        if crdo[1] - crdo[0] < args.roi_ovlp * 4 or \
                                crdo[3] - crdo[2] < args.roi_ovlp * 4:
                            print('Ignore', '_'.join(map(str, crd + crdo)))
                        else:
                            prep_args.append([df_rna, bid, masks, crd, crdo,
                                              out_pth, rna_axs, color])
            pool.starmap(prep_roi_rna, prep_args)

    if args.test_roi_cell:
        with multiprocessing.Pool(processes=args.core) as pool:
            prep_args = list()
            bar = None
            for clt in (range(11)):
                if clt == 0:
                    clt_nam = 'graphclust'
                if clt == 1:
                    continue
                if clt > 1:
                    clt_nam = f'kmeans_{clt}_clusters'

                clt_pth = args.root / 'outs' / 'analysis' / 'clustering' / \
                    f'gene_expression_{clt_nam}'
                df_raw = pd.read_csv(str(clt_pth / 'clusters.csv'))
                bar1 = df_raw
                if bar is not None:
                    assert (bar.Barcode == bar1.Barcode).all()
                else:
                    bar = bar1
                print(clt_nam, len(df_raw))

            dfm = pq.read_table(str(args.root / 'outs' / 'nucleus_boundaries.parquet'),
                                columns=['cell_id', 'vertex_y', 'vertex_x']).to_pandas()

            # calc the centroid of each cell based on nucleus boundary
            # minimum 7 vertices
            dfm = dfm.groupby(['cell_id'],
                              as_index=False)[['vertex_y', 'vertex_x']].mean()
            print(len(dfm))
            _dct = dict(zip(dfm.cell_id, dfm.index.values + 1))
            bar.Barcode = bar.Barcode.map(_dct)
            bid = bar.Barcode.values

            cell_pth = list((args.root / 'rna').rglob('*_msk.npz'))
            df = pd.read_csv(str(Path(args.root) / 'GAN/crop/metadata.csv'),
                             index_col=0)
            cid = df.num_id.values
            for pth in cell_pth:
                prep_args.append((str(pth), cid, bid))
            pool.starmap(test_roi_cell, prep_args)

    # dapi_pth = list((args.root / 'dapi').rglob('*.jpg'))
    # for pth in dapi_pth:
    #     rna_pth = str(pth).replace('dapi', 'rna')
    #     if Path(rna_pth.replace('.jpg', '_rna.npz')).is_file():
    #         fdir = 'dapi0'
    #     else:
    #         fdir = 'dapi1'
    #     shutil.copyfile(str(pth),
    #                     str(pth).replace('dapi', fdir))

    # # Test the new component in prep_roi_rna that exclude cells without label
    # # all the cluster should have the same cell ids
    # clt_pth = args.root / 'outs' / 'analysis' / 'clustering' / \
    #     f'gene_expression_graphclust'
    # dfc = pd.read_csv(str(clt_pth / 'clusters.csv'))
    # dfm = pq.read_table(str(args.root / 'outs' / 'nucleus_boundaries.parquet'),
    #                     columns=['cell_id', 'vertex_y', 'vertex_x']).to_pandas()
    # # calc the centroid of each cell based on nucleus boundary
    # # minimum 7 vertices
    # dfm = dfm.groupby(['cell_id'],
    #                   as_index=False)[['vertex_y', 'vertex_x']].mean()
    # print('all cells', len(dfm))
    # _dct = dict(zip(dfm.cell_id, dfm.index.values + 1))
    # dfc.Barcode = dfc.Barcode.map(_dct)
    # bid = dfc.Barcode.values
    # print('all cells with label', len(dfc))

    # msk_pth = list((args.root / 'rna').rglob('*_msk.npz'))
    # for pid, pth in enumerate(msk_pth):
    #     pth_old = str(pth).replace('rna/', 'rna_okay/')
    #     msk = sparse.load_npz(str(pth)).todense()
    #     msk_old = sparse.load_npz(str(pth_old)).todense()
    #     diff = msk_old[msk_old != msk]
    #     if len(diff) != 0:
    #         diff = np.unique(diff)
    #         assert all(d not in bid for d in diff)
    #     if pid % 100 == 0:
    #         print(pid)
