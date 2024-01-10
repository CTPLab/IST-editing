import cv2
import torch
import pyvips
import random
import sparse
import itertools
import numpy as np
import pandas as pd
import torchvision.transforms.functional as F
import torchvision.transforms.functional_tensor as Ft

from PIL import Image
from scipy import stats
from pathlib import Path
from random import shuffle
from Dataset_inf.Dataset import Dataset


class InfDataset(Dataset):
    def __init__(self,
                 path,  # Path to directory or zip.
                 gene,
                 resolution=128,
                 transform=False,
                 label='kmeans_10_clusters',
                 sub_pth=None,
                 crop=False,
                 debug=False,
                 sm_roi=None,
                 repeat=None,
                 stain='dapi',
                 **super_kwargs):
        self._path = path
        self._gene = gene
        self._type = 'dir'
        self._zipfile = None
        self.transform = transform
        self.crop = crop
        self.debug = debug
        self.sm_roi = sm_roi
        self.raw_sz = resolution
        self.stain = stain

        df_dir = 'crop_tile' if self.crop == 'tile' else 'crop'
        df = pd.read_csv(str(Path(self._path) / f'GAN/{df_dir}/metadata.csv'),
                         index_col=0)
        if self.crop:
            df, is_valid = self._fltr_crop(df, resolution)
            if label:
                assert label in df
                self.labc = df[label].values.tolist()

        if label:
            assert label in df
            if 'kmeans' in label:
                # Add additional 0 cat of black tile w/t gene
                self._ldim = int(label.split('_')[1]) + 1
            elif label == 'graphclust':
                # 67 categories with additional 0 cat of black tile w/t gene
                self._ldim = 68
            self._lab = dict(zip(df.num_id.values, df[label].values))
        else:
            # Without label for discriminator training
            self._ldim = 0

        # To make sure that an input image always has the gene expr,
        # we load gene expr npz first
        self._image_fnames = list(Path(path).glob('rna/*_rna.npz'))
        if sub_pth:
            self._image_fnames = sub_pth
        if repeat is not None:
            _fnm = list(itertools.repeat(self._image_fnames, repeat))
            self._image_fnames = list(itertools.chain(*_fnm))
        if self.debug or self.crop:
            self._x = df.vertex_x.values.tolist()
            self._y = df.vertex_y.values.tolist()
            self._img = df.img_id.values.tolist()
            self._image_fnames = self._img

            if self.debug:
                self._nid = df.num_id.values.tolist()
                df_cell = pd.read_csv(str(Path(self._path) / f'GAN/{df_dir}/metadata_cell.csv'),
                                      index_col=0)
                if self.crop:
                    df_cell = df_cell[is_valid]
                    self._cnt = df.transcript_counts.values.tolist()
                self.expr_cell = df_cell.astype(np.int16).to_numpy()
                print('expr lens', len(self.expr_cell))
        print(len(self._image_fnames))
        raw_shape = [len(self._image_fnames),
                     1, resolution, resolution]
        super().__init__(name='Xenium_mouse', raw_shape=raw_shape, **super_kwargs)

    def _load_input(self, raw_idx):
        pth = str(self._image_fnames[raw_idx])
        msk_pth = pth.replace('rna.npz', 'msk.npz')
        msk = sparse.load_npz(msk_pth)
        img_pth = pth.replace('_rna.npz', '.jpg')
        img_pth = img_pth.replace('rna', self.stain)
        img = pyvips.Image.new_from_file(img_pth)

        # Get coord for the randomly cropped tile
        h, w = self._get_sm_roi(img.height, img.width)
        for cnt in range(4):
            top = random.randint(h[0], h[1])
            left = random.randint(w[0], w[1])
            # Sample cell mask util we get the non-black tile or max 4 times
            cell = msk[top + self.raw_sz // 2: top + 3 * (self.raw_sz // 2),
                       left + self.raw_sz // 2: left + 3 * (self.raw_sz // 2), 1]
            cell = cell.data
            # TODO: could also assign propotional label or weighted sampling
            if (len(cell) != 0):
                if self._ldim > 0:
                    clab = stats.mode(cell, keepdims=False)[0]
                    label = self._load_labels(self._lab[clab])
                else:
                    label = np.zeros(1)
                break
            elif cnt == 3:
                label = self._load_labels(0) if self._ldim > 0 else np.zeros(1)
        # The tile should center on the large image
        til = img.crop(left + self.raw_sz // 2,
                       top + self.raw_sz // 2,
                       self.raw_sz,
                       self.raw_sz)

        tiln = np.ndarray(buffer=til.write_to_memory(),
                          dtype=np.uint8,
                          shape=[til.height, til.width, til.bands])
        # # Test the cropping
        # img1 = np.array(Image.open(img_pth))
        # tiln1 = img1[top: top + self.raw_sz * 2,
        #              left: left + self.raw_sz * 2]
        # Image.fromarray(np.squeeze(tiln)).save(f'Experiment_nm/test/{top}_{left}_small.jpg')
        # Image.fromarray(tiln1).save(f'Experiment_nm/test/{top}_{left}_large.jpg')
        tiln = np.ascontiguousarray(tiln.transpose((2, 0, 1)))
        if 'CosMx' in self._name:
            tiln = tiln[1:]

        rna = sparse.load_npz(pth)
        # need neighboring gene expr as the local input feat
        rna = rna[top: top + self.raw_sz * 2,
                  left: left + self.raw_sz * 2, :self._gene]
        gene = rna[None, :, None].reshape((8, self.raw_sz // 4,
                                           8, self.raw_sz // 4, -1))
        gene = gene.sum((1, 3)).todense()
        # self._gene_test(rna, gene)
        return tiln, gene, label, left, top

    def _fltr_crop(self, df, res):
        print('total', len(df))
        valid_h = (df['vertex_y'] >= res) & \
            (df['vertex_y'] < df['height'] - res)
        valid_w = (df['vertex_x'] >= res) & \
            (df['vertex_x'] < df['width'] - res)
        is_valid = (valid_h) & (valid_w)
        df = df[is_valid]
        return df, is_valid

    def _get_coord(self, idx, sz):
        row = slice(self._y[idx] - sz,
                    self._y[idx] + sz)
        col = slice(self._x[idx] - sz,
                    self._x[idx] + sz)
        return row, col

    def _get_sm_roi(self, hei, wid):
        h_st, h_ed = 0, hei - self.raw_sz * 2
        w_st, w_ed = 0, wid - self.raw_sz * 2
        if self.sm_roi is not None:
            h_mid, w_mid = hei // 2, wid // 2
            h_st = h_mid - self.sm_roi - self.raw_sz // 2
            h_ed = h_mid + self.sm_roi - self.raw_sz * 3 // 2
            w_st = w_mid - self.sm_roi - self.raw_sz // 2
            w_ed = w_mid + self.sm_roi - self.raw_sz * 3 // 2
        return (h_st, h_ed), (w_st, w_ed)

    def _load_crop(self, idx):
        rna_pth = Path(self._path) / 'rna' / f'{self._img[idx]}_rna.npz'
        rna = sparse.load_npz(str(rna_pth))[:, :, :self._gene]
        row, col = self._get_coord(idx, self.raw_sz)
        gene = rna[row, col]
        h, w = gene.shape[:2]
        if h != self.raw_sz * 2 or w != self.raw_sz * 2:
            print(self._img[idx], h, w)
        gene = gene[None, :, None].reshape((8, self.raw_sz // 4,
                                            8, self.raw_sz // 4, -1))
        gene = gene.sum((1, 3)).todense()
        # self._cell_test(idx, rna, gene)

        img_pth = Path(self._path) / self.stain / f'{self._img[idx]}.jpg'
        img = cv2.imread(str(img_pth))
        row, col = self._get_coord(idx, self.raw_sz // 2)
        img = img.transpose((2, 0, 1))
        if self.stain == 'dapi': 
            img = img[0, row, col]
            img = img[None]
        img = torch.from_numpy(img).float()

        img = img / 127.5 - 1
        gene = torch.from_numpy(gene).float()
        if self._ldim > 0:
            label = self._load_labels(self.labc[idx])
        else:
            label = np.zeros(1)

        return img, gene, label

    def _load_labels(self, label):
        onehot = np.zeros(self._ldim, dtype=np.float32)
        onehot[label] = 1
        return onehot.copy()

    def __getitem__(self, idx):
        if self.debug:
            err = self._meta_test(idx)
            return err

        if self.crop:
            img, gene, label = self._load_crop(idx)
            return img, gene, label

        image, gene, label, left, top = self._load_input(self._raw_idx[idx])

        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8

        img = torch.from_numpy(image.copy()).float()
        if self.resolution != self.raw_sz:
            img = Ft.resize(img, self.resolution,
                            'bicubic', True).round().clamp(0, 255)
        gene = torch.from_numpy(gene).float()
        # self._trans_test(img, gene, left, top)
        if self.transform:
            img, gene = self._trans(img, gene)
        return img / 127.5 - 1, gene, label

    def _trans(self, img, gene, p=0.5):
        gene = gene.permute((2, 0, 1))
        n = random.randint(0, 3)
        if n > 0:
            img = torch.rot90(img, n, [1, 2])
            gene = torch.rot90(gene, n, [1, 2])
        if torch.rand(1) < p:
            img = F.hflip(img)
            gene = F.hflip(gene)
        return img, gene.permute((1, 2, 0))

    def _trans_test(self, img, gene, left, top):
        if img.shape[0] == 2:
            img = torch.cat((torch.zeros_like(img[0])[None], img))
        # Here we test the transform consistency
        cv2.imwrite(f'Experiment_nm/test/{top}_{left}_img0.jpg',
                    img.permute((1, 2, 0)).numpy().astype(np.uint8))
        cv2.imwrite(f'Experiment_nm/test/{top}_{left}_gene0.jpg',
                    (gene).sum(-1).numpy().astype(np.uint8))
        gene = gene.permute((2, 0, 1))
        imgf = F.hflip(img)
        genef = F.hflip(gene)
        cv2.imwrite(f'Experiment_nm/test/{top}_{left}_imgf.jpg',
                    imgf.permute((1, 2, 0)).numpy().astype(np.uint8))
        cv2.imwrite(f'Experiment_nm/test/{top}_{left}_genef.jpg',
                    (genef).sum(0).numpy().astype(np.uint8))
        n = random.randint(1, 3)
        img = torch.rot90(img, n, [1, 2])
        gene = torch.rot90(gene, n, [1, 2])
        cv2.imwrite(f'Experiment_nm/test/{top}_{left}_img{n}.jpg',
                    img.permute((1, 2, 0)).numpy().astype(np.uint8))
        cv2.imwrite(f'Experiment_nm/test/{top}_{left}_gene{n}.jpg',
                    (gene).sum(0).numpy().astype(np.uint8))

    def _gene_test(self, rna, gene):
        # Here rna is the spatial resolved raw mrna count data
        # gene is the processed n x n x plex expression table
        for i in range(8):
            for j in range(8):
                row = slice(i * self.raw_sz // 4,
                            (i + 1) * self.raw_sz // 4)
                col = slice(j * self.raw_sz // 4,
                            (j + 1) * self.raw_sz // 4)
                gn = rna[row, col, :].sum((0, 1)).todense()
                assert (gn == gene[i, j]).all()

    def _cell_test(self, idx, rna, gene):
        assert self._y[idx] - self.raw_sz >= 0 and \
            self._y[idx] + self.raw_sz < rna.shape[0]
        assert self._x[idx] - self.raw_sz >= 0 and \
            self._x[idx] + self.raw_sz < rna.shape[1]

        rna = rna[self._y[idx] - self.raw_sz:self._y[idx] + self.raw_sz,
                  self._x[idx] - self.raw_sz:self._x[idx] + self.raw_sz,
                  :self._gene]
        # Here rna is the spatial resolved raw mrna count data
        # gene is the processed n x n x plex expression table
        for i in range(8):
            for j in range(8):
                row = slice(i * self.raw_sz // 4,
                            (i + 1) * self.raw_sz // 4)
                col = slice(j * self.raw_sz // 4,
                            (j + 1) * self.raw_sz // 4)
                gn = rna[row, col].sum((0, 1)).todense()
                assert (gn == gene[i, j]).all()

    def _meta_test(self, idx, sz=256):
        if self.crop == 'tile':
            sz = 64
        rna_pth = Path(self._path) / 'rna' / f'{self._img[idx]}_rna.npz'
        rna = sparse.load_npz(str(rna_pth))[:, :, :self._gene]
        row = slice(max(self._y[idx] - sz, 0),
                    min(self._y[idx] + sz, rna.shape[0]))
        col = slice(max(self._x[idx] - sz, 0),
                    min(self._x[idx] + sz, rna.shape[1]))
        rna_out = rna[row, col, :].todense()

        if self.crop == 'cell':
            msk_pth = Path(self._path) / 'rna' / f'{self._img[idx]}_msk.npz'
            msk = sparse.load_npz(str(msk_pth))
            msk = msk[row, col, :].todense()
            rna_out[(msk[:, :, 0] != self._nid[idx]) &
                    (msk[:, :, 1] != self._nid[idx])] = 0
        rna_sum = rna_out.sum((0, 1))
        err = np.abs(rna_sum -
                     self.expr_cell[idx])
        if self.crop == 'tile':
            assert rna_sum.sum() == self._cnt[idx]
        return err


if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader
    parser = argparse.ArgumentParser(description='StyleGAN2 data class')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--crop', default=False,
                        help='crop method: cell-or tile-based')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    data = InfDataset(f'Data/Xenium_mouse', 379, transform=False,
                      crop=args.crop, debug=args.debug, use_labels=True)
    print(len(data))
    dload = DataLoader(data, batch_size=args.batch, shuffle=True,
                       **{'drop_last': True,
                          'num_workers': 8,
                          'pin_memory': True})
    if args.debug:
        errn, errc = 0, 0
        for i, err in enumerate(dload):
            errn += err.sum()
            errc += (err.sum((-1)) != 0).sum()
            if (i + 1) % 100 == 0:
                print(i + 1,
                      errn / (args.batch * (i + 1)),
                      errc / (args.batch * (i + 1)),)
    elif args.crop:
        for i, gene in enumerate(dload):
            if (i + 1) % 200 == 0:
                print(i + 1)
            if (i + 1) % 5000 == 0:
                break
    else:
        for i, (img, gene, lab) in enumerate(dload):
            if i % 100 == 0:
                print(i, img.shape, gene.shape, lab.shape)
