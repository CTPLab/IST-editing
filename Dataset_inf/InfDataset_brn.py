import cv2
import torch
import pyvips
import random
import sparse
import itertools
import numpy as np
import pandas as pd
import torchvision.transforms.functional as F

from pathlib import Path
from random import shuffle



def crop_ome(img, left, top, size):
    out = img.crop(left, top, size, size).numpy()
    return out[None]

def rewei_pth(pth):
    hippo_lst = ['20480_22528_38912_40960_20224_22784_38656_41216',
                 '20480_22528_40960_43008_20224_22784_40704_43264',
                 '20480_22528_65536_67584_20224_22784_65280_67840',
                 '20480_22528_67584_69632_20224_22784_67328_69888']
    out = list()
    for p in pth:
        if p.stem not in hippo_lst:
            out.append(p)
        else:
            out = out + [p, ] * 100
    return out

class InfDataset_brn(torch.utils.data.Dataset):
    def __init__(self, gdir,
                 gdim, sdim, ldim,  # 0 or 14 for label
                 stain='DAPI',
                 repeat=10**6,
                 transform=False,
                 debug=False):
        self.gdim = gdim
        self.sdim = sdim
        self.ldim = ldim
        self.stain = stain
        self.transform = transform
        self.debug = debug

        gn_pth = list(Path(gdir).glob('*.npz'))
        gn_pth = rewei_pth(gn_pth)
        gn_pth = list(itertools.repeat(gn_pth, repeat))
        self.gn_pth = list(itertools.chain(*gn_pth))
        print(len(self.gn_pth), debug)

    def _getgene(self, pth):
        gene = sparse.load_npz(pth)
        if len(gene.shape) == 3:
            gene = gene[None]
        gh, gw = gene.shape[-3:-1]
        h = (0, gh - self.sdim * 2)
        w = (0, gw - self.sdim * 2)
        # Lower the chance of sampling black gene region
        for _ in range(4):
            top = random.randint(h[0], h[1])
            left = random.randint(w[0], w[1])
            gn_crop = gene[:, top: top + self.sdim * 2,
                           left: left + self.sdim * 2, :self.gdim]
            if (len(gn_crop.data) != 0):
                break
        gn = gn_crop[:, None, :, None].reshape((gn_crop.shape[0],
                                                8, self.sdim // 4,
                                                8, self.sdim // 4, -1))
        gn = gn.sum((2, 4)).todense()
        if self.debug:
            self._gene_test(gn_crop, gn)
        gn = torch.from_numpy(gn).float()
        assert len(gn.shape) == 4
        return gn, top, left

    def _getimg(self, pth, top, left):
        # n = -1 means loading all the channels
        img = pyvips.Image.new_from_file(pth, n=-1)
        im = crop_ome(img, left + self.sdim // 2,
                      top + self.sdim // 2, self.sdim)
        im = torch.from_numpy(im).float()
        assert len(im.shape) == 3
        return im

    def __getitem__(self, idx):
        pth = str(self.gn_pth[idx])
        gn, top, left = self._getgene(pth)

        img_pth = pth.replace('gene', self.stain).replace('.npz', '.tif')
        im = self._getimg(img_pth, top, left)

        # if self.ldim != 0:
        #     msk_pth = pth.replace('gene', 'msk')
        #     msk = cv2.imread(msk_pth)
        #     mk = msk[top + self.sdim // 2: top + 3 * (self.sdim // 2),
        #              left + self.sdim // 2: left + 3 * (self.sdim // 2)]
        #     mk = np.clip(mk.round(), 0, self.ldim - 1)
        #     val, cnt = np.unique(mk[mk != 0],
        #                          return_counts=True)
        #     label = val[np.argmax(cnt)]
        # else:
        #     label = np.zeros(1)
        if self.debug:
            self._trans_test(im, gn, top, left)
        if self.transform:
            im, gn = self._trans(im, gn)

        return im / 127.5 - 1, gn.squeeze(), np.zeros(1)

    def __len__(self):
        return len(self.gn_pth)

    def _load_labels(self, label):
        onehot = np.zeros(self._ldim, dtype=np.float32)
        onehot[label] = 1
        return onehot.copy()

    def _trans(self, img, gene, p=0.5):
        # gn: [chn, h, w, gdim], im: [chn, h, w]
        n = random.randint(0, 3)
        if n > 0:
            gene = torch.rot90(gene, n, [1, 2])
            img = torch.rot90(img, n, [1, 2])
        gene = gene.permute((0, 3, 1, 2))
        if torch.rand(1) < p:
            gene = F.hflip(gene)
            img = F.hflip(img)
        return img, gene.permute((0, 2, 3, 1))

    def _trans_test(self, img, gene, top, left):
        imf0 = F.hflip(img)
        imf1 = torch.from_numpy(img.numpy()[:, :, ::-1].copy())
        assert (imf0 == imf1).all()

        gn = gene.permute((0, 3, 1, 2))
        gn = F.hflip(gn)
        gn0 = gn.permute((0, 2, 3, 1))
        gn1 = torch.from_numpy(gene.numpy()[:, :, ::-1].copy())
        assert (gn0 == gn1).all()

        # s = random.randint(0, 58 - 3)
        # n = random.randint(1, 3)
        # img = img[s:s+3]
        # im0 = torch.rot90(img, n, [1, 2])
        # # This step used as the surrogate for testing gene rot
        # # as the rotated gn image is too small
        # img_new = img[:, :, :, None].repeat((1, 1, 1, 3))
        # im1 = torch.rot90(img_new, n, [1, 2])[:, :, :, 0]
        # for ii, im in enumerate((img, im0, im1)):
        #     cv2.imwrite(f'Experiment_inf/test/{top}_{left}_{ii}.jpg',
        #                 (im.permute((1, 2, 0))).numpy().astype(np.uint8))

    def _gene_test(self, gene_raw, gene):
        # Here rna is the spatial resolved raw mrna count data
        # gene is the processed n x n x plex expression table
        for i in range(8):
            for j in range(8):
                row = slice(i * self.sdim // 4,
                            (i + 1) * self.sdim // 4)
                col = slice(j * self.sdim // 4,
                            (j + 1) * self.sdim // 4)
                gn = gene_raw[:, row, col, :].sum((1, 2)).todense()
                assert (gn == gene[:, i, j]).all()


if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader
    parser = argparse.ArgumentParser(description='StyleGAN2 data class')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--mouse', type=str,
                        default='609882',
                        choices=['609882', '609889', '638850'],
                        help='Folder to different mouses')
    args = parser.parse_args()

    data = InfDataset_brn(f'Data/MERFISH/gene_{args.mouse}',
                         500, 128, 0,
                         repeat=8, transform=False, debug=True)
    print(len(data))
    dload = DataLoader(data, batch_size=args.batch, shuffle=True,
                       **{'drop_last': True,
                          'num_workers': 8,
                          'pin_memory': True})
    for i, (img, gene, lab) in enumerate(dload):
        print(i)
        if i % 99 == 0:
            print(i, img.shape, gene.shape)