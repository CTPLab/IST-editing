import sparse
import pickle
import tiledb
import tiledbsoma
import pyvips
import itertools
import multiprocessing

import numpy as np
import pandas as pd
import scanpy as sc

from scipy import stats
from pathlib import Path
from cleanfid import fid
from random import shuffle
import pyarrow.parquet as pq

# The PX_SZ here is slightly different from Xenium (Lung) (Dataset/utils)
# which is determined in the 'cells.zarr.zip'
PX_SZ = 4.705882
pd.options.mode.chained_assignment = None


def _um_to_pixel(df, crd_nm):
    for axis in ('x', 'y'):
        # astype(float) could slightly improve the inconsistency on dgx
        df[crd_nm.format(axis)] = df[crd_nm.format(axis)].astype(float) * PX_SZ
        # this gives us the least error rate compared to use round()
        df[crd_nm.format(axis)] = df[crd_nm.format(axis)].astype(int)


def _df_to_roi(df, roi_crop, roi_trns,
               crd_nm, crop=True):
    if crop:
        is_roi_h = (df[crd_nm.format('y')] >= roi_crop[0]) & \
            (df[crd_nm.format('y')] < roi_crop[1])
        is_roi_w = (df[crd_nm.format('x')] >= roi_crop[2]) & \
            (df[crd_nm.format('x')] < roi_crop[3])
        df = df[(is_roi_h) & (is_roi_w)]
    # covert the global- to local- coordinate for the roi image
    # hei_st = crd[0], wid_st = crd[2]
    df[crd_nm.format('y')] -= roi_trns[0]
    df[crd_nm.format('x')] -= roi_trns[2]
    return df


def _df_add_col(df, crd, crdo,
                rna_pth, raw_sz):
    df['img_id'] = '_'.join(map(str, crd + crdo))
    df['height'] = crdo[1] - crdo[0]
    df['width'] = crdo[3] - crdo[2]
    for sz in (256, 512):
        img_sz, hlim, wlim = _get_sm_roi(str(rna_pth), raw_sz, sz)
        assert crdo[1] - crdo[0] == img_sz[0]
        assert crdo[3] - crdo[2] == img_sz[1]
        df[str(sz)] = _df_roi_fltr(df, 'vertex_{}',
                                   hlim[0], hlim[1], wlim[0], wlim[1])
    return df


def _df_roi_fltr(df, crd_nm,
                 h_st, h_ed, w_st, w_ed):
    # Only the cropped image that completely in
    # the training region is considered training data
    is_roi_h = (df[crd_nm.format('y')] >= h_st) & \
        (df[crd_nm.format('y')] < h_ed)
    is_roi_w = (df[crd_nm.format('x')] >= w_st) & \
        (df[crd_nm.format('x')] < w_ed)
    return (is_roi_h) & (is_roi_w)


def _get_sm_roi(pth, raw_sz, sm_roi=None):
    img_pth = pth.replace('_rna.npz', '.jpg')
    img_pth = img_pth.replace('rna', 'dapi')
    img = pyvips.Image.new_from_file(img_pth)
    hei, wid = img.height, img.width
    h_st, h_ed = 0, hei - raw_sz * 2
    w_st, w_ed = 0, wid - raw_sz * 2
    if sm_roi is not None:
        h_mid, w_mid = hei // 2, wid // 2
        h_st = h_mid - sm_roi - raw_sz // 2
        h_ed = h_mid + sm_roi - raw_sz * 3 // 2
        w_st = w_mid - sm_roi - raw_sz // 2
        w_ed = w_mid + sm_roi - raw_sz * 3 // 2
    # The above code is equivalent to the one in InfDataset_mouse,
    # where h_st, h_ed, w_st, w_ed correspond to left, top coords
    # Then we add raw_sz to align with centroid vertex
    return (hei, wid), (h_st + raw_sz, h_ed + raw_sz), (w_st + raw_sz, w_ed + raw_sz)


def prep_Xenium_meta(data='Xenium', raw_sz=128):
    in_pth, out_pth = Path(f'Data/{data}'), Path(f'Data/{data}/GAN')

    dfm = pq.read_table(str(in_pth / 'outs' / 'nucleus_boundaries.parquet'),
                        columns=['cell_id', 'vertex_y', 'vertex_x']).to_pandas()

    # calc the centroid of each cell based on nucleus boundary
    # minimum 7 vertices
    dfm = dfm.groupby(['cell_id'],
                      as_index=False)[['vertex_y', 'vertex_x']].mean()
    print(dfm.head())
    _dct = dict(zip(dfm.cell_id, dfm.index.values + 1))
    _y = dict(zip(dfm.cell_id, dfm.vertex_y))
    _x = dict(zip(dfm.cell_id, dfm.vertex_x))

    # prepare the meta data
    df_cell = pq.read_table(str(in_pth / 'outs' / 'cells.parquet')).to_pandas()
    df_cell = df_cell.drop(columns=['cell_area', 'nucleus_area',
                                    'x_centroid', 'y_centroid'])
    df_cell['vertex_y'] = df_cell.cell_id.map(_y)
    df_cell['vertex_x'] = df_cell.cell_id.map(_x)
    _um_to_pixel(df_cell, 'vertex_{}')
    print(df_cell.head(10))
    roi_pth = in_pth / 'rna'
    roi_lst = list(roi_pth.glob('*rna.npz'))
    df = list()
    for rid, roi in enumerate(roi_lst):
        crd_all = list(map(int, roi.stem.split('_')[:-1]))
        crd, crdo = crd_all[:4], crd_all[4:]
        df_sub = _df_to_roi(df_cell, crd, crdo, 'vertex_{}')
        df_sub = _df_add_col(df_sub, crd, crdo, roi, raw_sz)
        df.append(df_sub)
        if rid % 100 == 0:
            print(rid)
    df = pd.concat(df, ignore_index=True)

    # Here, one corner case anmhjhof-1 with cell id 80329 + 1
    # which only lies on the boundary of each roi
    df_corner = df_cell[~df_cell.cell_id.isin(df.cell_id)]
    print(df_corner)
    for rid, roi in enumerate(roi_lst):
        crd_all = list(map(int, roi.stem.split('_')[:-1]))
        crd, crdo = crd_all[:4], crd_all[4:]
        # We add the corner case back by considering crdo
        dfc = _df_to_roi(df_corner, crdo, crdo, 'vertex_{}')
        if not dfc.empty:
            print(rid, roi)
            dfc = _df_add_col(dfc, crd, crdo, roi, raw_sz)
            df = pd.concat([dfc, df], ignore_index=True)
            break
    print(df.head())
    print(len(df), len(df_cell))

    for clt in range(11):
        if clt == 0:
            clt_nam = 'graphclust'
        if clt == 1:
            continue
        if clt > 1:
            clt_nam = f'kmeans_{clt}_clusters'
        print(clt_nam)

        clt_pth = in_pth / 'outs' / 'analysis' / 'clustering' / \
            f'gene_expression_{clt_nam}'
        df_raw = pd.read_csv(str(clt_pth / 'clusters.csv'))
        print(len(df_raw))
        # bar_max = df_raw.Barcode.max()
        # assert df_raw.Barcode.is_monotonic_increasing
        # df_raw = df_raw.set_index('Barcode').reindex(range(1, bar_max + 1),
        #                                              fill_value=0)
        # df_raw = df_raw.reset_index().astype(int)
        raw_dct = dict(zip(df_raw.Barcode, df_raw.Cluster))
        df[clt_nam] = df.cell_id.map(raw_dct)
    df_meta = df[df.graphclust.notna()]
    # convert last 10 cols of cluster to int
    df_meta.iloc[:, -10:] = df_meta.iloc[:, -10:].astype(int)
    df_meta['num_id'] = df_meta.cell_id.map(_dct)
    assert not df_meta.isnull().values.any()
    assert not df_meta.isna().values.any()
    print(df_meta.head())
    df_meta.to_csv(str(out_pth / 'crop' / 'metadata.csv'), index=False)


def align_xenium_meta(data='Xenium'):
    pth = Path(f'Data/{data}')

    dfm = pd.read_csv(str(pth / 'GAN/crop/metadata.csv'))
    print(dfm.head())
    adata = sc.read_10x_h5(filename=str(pth / 'outs/cell_feature_matrix.h5'),
                           gex_only=True)
    gene_dct = dict(adata.var['feature_types'])
    df_gene = pq.read_table(str(pth / 'outs/transcripts.parquet'),
                            filters=[('cell_id', '!=', 'UNASSIGNED'),
                                     # https://www.10xgenomics.com/cn/resources/analysis-guides/performing-3d-nucleus-segmentation-with-cellpose-and-generating-a-feature-cell-matrix
                                     # critical QV thres >= 20
                                     ('qv', '>=', 20)],
                            columns=['cell_id', 'feature_name']).to_pandas()
    # print(df_gene.head())

    # the following steps can be very slow
    print('start counting the exprs')
    # compute the counts for Gene expr, neg ctr prob,
    # neg ctr cod and blank code (can be slow)
    df_gene = df_gene.groupby(['cell_id', 'feature_name'],
                              as_index=False).size()

    print('start transposing the df')
    # transpose the df format so it is similar to df_cell
    df_gene = df_gene.pivot(index='cell_id', columns='feature_name',
                            values='size').reset_index()
    df_gene.columns.name = None
    df_gene = df_gene.set_index('cell_id')
    df_gene = df_gene.fillna(0).astype(int)
    df_gene_col = df_gene.columns.isin(list(gene_dct.keys()))
    df_gene = df_gene.loc[:, df_gene_col]
    df_gene = df_gene.reindex(dfm.cell_id)
    print(df_gene.head())
    print(len(df_gene), len(dfm))
    df_gene.to_csv(str(pth / 'GAN/crop/metadata_cell.csv'))


def save_transcript_list(data):
    print(data)
    if data == 'Visium':
        lst = ['SLC2A1', 'CCN1', 'ATP1A1', 'S100A1', 'NES', 'SLC4A5', 'PAX3', 'MLPH', 'SEMA3B', 'WNT5A',
               'MITF', 'ROPN1B', 'SLIT2', 'SLC45A2', 'TGFBI', 'GFRA3', 'PDGFRB', 'ABCB5', 'AQP1', 'EGFR',
               'TMEM176B', 'GFRA2', 'LOXL2', 'MLANA', 'TYRP1', 'TNC', 'VIM', 'LOXL4', 'PLEKHB1', 'RAB38',
               'TYR', 'SLC2A3', 'PMEL', 'CDK2', 'ERBB3', 'NT5DC3', 'POSTN', 'SLC22A17', 'SERPINA3', 'AKT1',
               'CAPN3', 'CDH1', 'CDH13', 'NGFR', 'SOX9', 'CDH2', 'TCF4', 'BCL2', 'CDH19', 'MBP', 'MIA',
               'AXL', 'BIRC7', 'S100B', 'PRAME', 'SOX10', 'GPR143', 'GPM6B', 'PIR', 'GJB1', 'BGN']
    elif data == 'Xenium_mouse':
        adata = sc.read_10x_h5(filename=f'Data/{data}/outs/cell_feature_matrix.h5',
                               gex_only=True)
        gene_dct = dict(adata.var['feature_types'])
        lst = [k for k in gene_dct]
        print(lst, len(lst))
    elif data == 'Xenium':
        adata = sc.read_10x_h5(filename=f'Data/{data}/Lung1/outs/cell_feature_matrix.h5',
                               gex_only=True)
        gene_dct = dict(adata.var['feature_types'])
        lst = [k for k in gene_dct]

        adata2 = sc.read_10x_h5(filename=f'Data/{data}/Lung2/outs/cell_feature_matrix.h5',
                                gex_only=True)
        gene_dct2 = dict(adata2.var['feature_types'])
        assert lst == list(gene_dct2.keys())
    elif data == 'Xenium_brain':
        adata = sc.read_10x_h5(filename=f'Data/{data}/Brain1/outs/cell_feature_matrix.h5',
                               gex_only=True)
        gene_dct = dict(adata.var['feature_types'])
        lst = [k for k in gene_dct]

        adata2 = sc.read_10x_h5(filename=f'Data/{data}/Brain2/outs/cell_feature_matrix.h5',
                                gex_only=True)
        gene_dct2 = dict(adata2.var['feature_types'])

        # We exclude the Alzheimers slide due to batch effect and different gene collection
        # adata3 = sc.read_10x_h5(filename=f'Data/{data}/Alzheimers/outs/cell_feature_matrix.h5',
        #                         gex_only=True)
        # gene_dct3 = dict(adata3.var['feature_types'])

        assert lst == list(gene_dct2.keys())
    elif data == 'CosMx':
        df_expr = pd.read_csv(f'Data/{data}/Liver1/exprMat_file.csv',
                              index_col=0)
        lst = [e for e in df_expr.columns]

        df_expr2 = pd.read_csv(f'Data/{data}/Liver2/exprMat_file.csv',
                               index_col=0)
        assert lst == df_expr2.columns.values.tolist()
    else:
        raise NameError('unrecognized data name {data}')

    with open(f'Data/{data}/GAN/crop/transcripts.pickle', 'wb') as fp:
        pickle.dump(lst, fp)


def prep_one_tile(pth, roi, gene_num,
                  hei, wid,
                  hst, wst, raw_sz=128):
    rna = sparse.load_npz(str(pth))
    assert rna.shape[0] == hei and rna.shape[1] == wid
    rna = rna[hst:hst+roi, wst:wst+roi, :gene_num]
    msk_pth = str(pth).replace('rna.npz', 'msk.npz')
    msk = sparse.load_npz(msk_pth)
    msk = msk[hst:hst+roi, wst:wst+roi, 1]

    num = roi // raw_sz
    gene = rna[None, :, None].reshape((num, raw_sz,
                                       num, raw_sz, -1))
    gene = gene.sum((1, 3)).todense().astype(np.uint16)
    mask = msk[None, :, None].reshape((num, raw_sz,
                                       num, raw_sz))
    mask = mask.transpose((0, 2, 1, 3))
    # 'cell_id', 'transcript_counts', 'control_probe_counts',
    #    'control_codeword_counts', 'unassigned_codeword_counts',
    #    'deprecated_codeword_counts', 'total_counts', 'vertex_y', 'vertex_x',
    #    'img_id', 'height', 'width', 'graphclust', 'kmeans_2_clusters',
    #    'kmeans_3_clusters', 'kmeans_4_clusters', 'kmeans_5_clusters',
    #    'kmeans_6_clusters', 'kmeans_7_clusters', 'kmeans_8_clusters',
    #    'kmeans_9_clusters', 'kmeans_10_clusters', 'num_id'

    # Test snippet
    meta, meta_cell = [], []
    for r in range(num):
        for c in range(num):
            dat = mask[r, c].data
            if len(dat) != 0:
                clab = stats.mode(dat, keepdims=False)[0]
                assert clab != 0, f'wrong {pth.stem[:-4]}'
                vy = int((r + 0.5) * raw_sz) + hst
                vx = int((c + 0.5) * raw_sz) + wst
                sm_roi, ctry, ctrx = [], hei // 2, wid // 2
                for sz in (256, 512):
                    shft = sz - raw_sz // 2
                    sm_roi.append((ctry - shft <= vy < ctry + shft) and \
                                  (ctrx - shft <= vx < ctrx + shft))
                meta.append([hei, wid,
                             vy, vx, sm_roi[0], sm_roi[1],
                             pth.stem[:-4], gene[r, c].sum(), clab])
                meta_cell.append(gene[r, c])
            # row = slice(r * raw_sz, (r + 1) * raw_sz)
            # col = slice(c * raw_sz, (c + 1) * raw_sz)
            # gn = rna[row, col].sum((0, 1))
            # mk = msk[row, col]
            # assert (gn.todense() == gene[r, c]).all()
            # assert (mk.todense() == mask[r, c].todense()).all()
    print(gene.shape, msk.shape, pth.stem[:-4])
    return meta, meta_cell


def prep_Xenium_tile(core, roi=2048,
                     gene=379, raw_sz=128):
    save_pth = Path('Data/Xenium_mouse/GAN/crop_tile')
    save_pth.mkdir(parents=True, exist_ok=True)
    with multiprocessing.Pool(processes=core) as pool:
        prep_args = list()

        rna_pths = list(Path('Data/Xenium_mouse/rna').glob('*_rna.npz'))
        # shuffle(rna_pths)
        for pid, p in enumerate(rna_pths):
            crd = p.stem.split('_')
            if int(crd[1]) - int(crd[0]) == int(crd[3]) - int(crd[2]) == roi:
                assert int(crd[0]) - int(crd[4]) == raw_sz // 2
                assert int(crd[2]) - int(crd[6]) in (0, raw_sz // 2)
                hei = int(crd[5]) - int(crd[4])
                wid = int(crd[7]) - int(crd[6])
                prep_args.append((p, roi, gene,
                                  hei, wid,
                                  int(crd[0]) - int(crd[4]),
                                  int(crd[2]) - int(crd[6])))
            # if pid == 50:
            #     break
        print(len(prep_args))
        res = pool.starmap(prep_one_tile, prep_args)
        meta, meta_cell = list(zip(*res))
        # Merge sublist collected from each calc to one list
        meta = list(itertools.chain.from_iterable(meta))
        h_sz, w_sz, v_y, v_x, sm0, sm1, i_id, cnt, n_id = list(zip(*meta))
        meta_cell = list(itertools.chain.from_iterable(meta_cell))
        print(len(v_y), len(v_x), len(i_id), len(n_id), len(meta_cell))
        print(len(meta[0]), len(meta_cell[0]))
        dfm = pd.read_csv('Data/Xenium_mouse/GAN/crop/metadata.csv')
        # assert (dfm['total_counts'] == (dfm['transcript_counts'] + \
        #                                 dfm['control_probe_counts'] + \
        #                                 dfm['control_codeword_counts'] + \
        #                                 dfm['unassigned_codeword_counts'] + \
        #                                 dfm['deprecated_codeword_counts'])).all()
        # assert (dfm['total_counts'] >= dfm['transcript_counts']).all()
        dfm = dfm.set_index('num_id')
        dfm = dfm.reindex(n_id)
        dfm = dfm.reset_index()
        dfm['height'] = h_sz
        dfm['width'] = w_sz
        dfm['vertex_y'] = v_y
        dfm['vertex_x'] = v_x
        dfm['img_id'] = i_id
        dfm['total_counts'] = cnt
        dfm['transcript_counts'] = cnt
        dfm['256'] = sm0
        dfm['512'] = sm1
        for col in ['control_probe_counts',
                    'control_codeword_counts',
                    'unassigned_codeword_counts',
                    'deprecated_codeword_counts']:
            dfm[col] = 0
        cols = list(dfm.columns)
        cols = cols[1:] + [cols[0]]
        dfm = dfm[cols]
        print(dfm.head(), len(dfm))
        dfm.to_csv(str(save_pth / 'metadata.csv'), index=False)

        gnm = pd.read_csv('Data/Xenium_mouse/GAN/crop/metadata_cell.csv',
                          nrows=0).columns.tolist()
        dfc = pd.DataFrame(meta_cell,
                           columns=gnm[1:])
        dfc.insert(0, column='cell_id', value=dfm.cell_id)
        print(dfc.head(), len(dfc))
        dfc.to_csv(str(save_pth / 'metadata_cell.csv'), index=False)


if __name__ == '__main__':
    save_transcript_list('Xenium_mouse')
    prep_Xenium_meta('Xenium_mouse')
    align_xenium_meta('Xenium_mouse')
    prep_Xenium_tile(8, 2048, 379)
