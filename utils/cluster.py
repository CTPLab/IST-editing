import cv2
import sparse
import argparse
import multiprocessing

import numpy as np
import pandas as pd
import colorcet as cc
import seaborn as sns
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw


def sort_df(df, col='cell_id', pad=0):
    imax = df[col].max() + 1 + pad
    df = df.set_index(col).reindex(range(imax), fill_value=0)
    assert df.equals(df.sort_index())
    # df = df.reset_index().astype(int)
    return df


def prep_type_legend(data, sub_lst, pth, hei=16, wid=30, shf=8):

    clr = sns.color_palette(cc.glasbey, n_colors=len(sub_lst))
    clr_dct = {sub_lst[i]: (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
               for i, c in enumerate(clr)}

    mark = np.ones([(hei+shf) * (len(clr)), wid * 16, 3]) * 255
    for key, tpy in enumerate(clr_dct):
        mark[(hei+shf)*key:(hei+shf)*key + hei, :wid] = clr_dct[tpy]

    mark = Image.fromarray(mark.astype(np.uint8))
    font = ImageFont.truetype('arial.ttf', 16)
    draw = ImageDraw.Draw(mark)
    for key, typ in enumerate(clr_dct):
        draw.text((wid + 4, (hei+shf)*key), str(typ),
                  (0, 0, 0), font=font)
    if not (pth / 'legend.png').is_file():
        mark.save(str(pth / 'legend.png'))
    if data == 'Xenium':
        clr_dct[0] = (0, 0, 0)
    return clr_dct


def visual_subtype(data, df, msk_pth, out_pth):
    if data == 'CosMx':
        fov = int(msk_pth.stem[-3:])
        df = df[df['fov'] == fov]
        if df.empty:
            print(f'{msk_pth} has no cells')
            return
        df = df.drop('fov', axis=1)
        df = sort_df(df, pad=500)
        msk = cv2.imread(str(msk_pth), flags=cv2.IMREAD_UNCHANGED)
        comp_pth = str(msk_pth).replace('CellLabels', 'CompartmentLabels')
        comp = cv2.imread(str(comp_pth), flags=cv2.IMREAD_UNCHANGED)
        msk[comp != 1] = 0
    elif data == 'Xenium':
        msk = sparse.load_npz(str(msk_pth))
        msk = msk[:, :, 0].todense()

    out = np.stack([(df.r.values)[msk],
                    (df.g.values)[msk],
                    (df.b.values)[msk]], axis=-1)
    pth = out_pth / f'{msk_pth.stem}.png'
    Image.fromarray(out.astype(np.uint8)).save(str(pth))
    print(pth, 'done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Draw cell subtypes.py')
    parser.add_argument('--path',
                        type=Path,
                        default=Path('Data/'),
                        help='Path to the ST dataset.')
    parser.add_argument('--data',
                        type=str,
                        help='Name of the ST dataset.')
    parser.add_argument('--core',
                        type=int,
                        default=8,
                        help='Number of cores used for image processing.')
    parser.add_argument('--cluster',
                        type=str,
                        help='column name for the cluster')
    args = parser.parse_args()
    meta_pth = args.path / f'{args.data}/GAN/crop/metadata.csv'
    save_pth = Path(f'Experiment/{args.data}/cluster/{args.cluster}')
    save_pth.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(meta_pth)
    assert args.cluster in df
    sub_lst = np.unique(df[args.cluster])
    clr_dct = prep_type_legend(args.data, sub_lst, save_pth)
    print(sub_lst)

    with multiprocessing.Pool(processes=args.core) as pool:
        sub_args, col_lst = list(), ['r', 'g', 'b']
        if args.data == 'CosMx':
            col, col_flter = 'slide_ID_numeric', ['cell_id', 'fov']
        elif args.data == 'Xenium':
            col, col_flter = 'slide_ID_numeric', ['cell_id', ]
            df1 = pd.read_csv(str(meta_pth).replace('.csv', '_bar.csv'))

        for i in (1, 2):
            (save_pth / str(i)).mkdir(parents=True, exist_ok=True)

            dfs = df[df[col] == i].copy()
            if args.data == 'CosMx':
                dfs['cell_id'] = dfs.path.map(lambda x: int(x.split('_')[-2]))
            elif args.data == 'Xenium':
                dfb = df1[df[col] == i]
                dfs['cell_id'] = dfb[dfb.columns[0]].map(lambda x: int(x.split('_')[2]))

            dfs[args.cluster] = dfs[args.cluster].map(clr_dct)
            for n, c in enumerate(col_lst):
                dfs[c] = dfs[args.cluster].apply(lambda x: x[n])
            dfs = dfs[col_flter + col_lst]

            if args.data == 'Xenium':
                dfs = sort_df(dfs, pad=0)
            # elif args.data == 'CosMx':
            #     dfs = dfs[dfs['fov'] == 24]
            #     dfs = dfs.drop('fov', axis=1)
            #     dfs = sort_df(dfs, pad=500)
            #     print(dfs)

            #     print(len(dfs))
            #     print(dfs.head())

            if args.data == 'CosMx':
                msk_pth = (args.path / f'{args.data}/Liver{i}/CellLabels').\
                    glob('CellLabels_F*.tif')
            elif args.data == 'Xenium':
                msk_pth = (args.path / f'{args.data}/Lung{i}/rna').\
                    glob('*_msk.npz')

            for mid, msk in enumerate(msk_pth):
                sub_args.append((args.data, dfs, msk, save_pth / str(i)))
        pool.starmap(visual_subtype, sub_args)

    # for r in ('Rep1', 'Rep2'):
    #     clt_pth = args.path / r / 'outs' / 'analysis' / 'clustering' / \
    #         f'gene_expression_{args.cluster}'
    #     df_raw = pd.read_csv(str(clt_pth / 'clusters.csv'))
    #     bar_max = df_raw.Barcode.max()
    #     assert df_raw.Barcode.is_monotonic_increasing
    #     df = df_raw.set_index('Barcode').reindex(range(1, bar_max + 1),
    #                                              fill_value=0)
    #     df = df.reset_index().astype(int)
    #     # df = df[df.Cluster != 0].reset_index(drop=True)
    #     # assert df_raw.equals(df)

    #     # split cluster color to three columns
    #     df.Cluster = df.Cluster.map(clr_dct)
    #     for n, col in enumerate(col_lst):
    #         df[col] = df.Cluster.apply(lambda x: x[n])
    #     df = df.drop(columns=['Cluster'])
    #     # insert (top) 0 list for assigning black to background
    #     df.loc[-1] = [0, 0, 0, 0]
    #     df.index = df.index + 1
    #     df = df.sort_index()
    #     assert df.Barcode.is_monotonic_increasing
    #     print(df, '\n')

    #     out_pth = args.path / r / f'type_{args.cluster}'
    #     out_pth.mkdir(parents=True, exist_ok=True)
    #     msk_pth = (args.path / r / 'rna').glob('*_msk.npz')
    #     for msk in msk_pth:
    #         sub_args.append((df, msk, out_pth))
    #         # break
    # pool.starmap(visual_subtype, sub_args)
