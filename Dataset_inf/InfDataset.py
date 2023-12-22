import cv2
import torch
import pyvips
import random
import sparse
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms.functional_tensor as Ft

from PIL import Image
from pathlib import Path
from random import shuffle
from Dataset_inf.Dataset import Dataset


class InfDataset(Dataset):
    def __init__(self,
                 path,  # Path to directory or zip.
                 gene,
                 resolution=128,
                 transform=False,
                 **super_kwargs):
        self._path = path
        self._gene = gene
        self._type = 'dir'
        self._zipfile = None
        self.transform = transform

        # To make sure that an input image always has the gene expr,
        # we load gene expr npz first.
        if 'CosMx' in path:
            name = 'CosMx_Liver'
            # compatabile with crop exp (e.g., 160 x 160)
            chn, self.offset, raw_sz = 2, 512, int(resolution * 1.25)
            data = 'Liver*/GeneLabels/GeneLabels*.npz'
        elif 'Xenium' in path:
            name = f'Xenium_{"Brain" if "brain" in path else "Lung"}'
            if 'Xenium_breast' in path:
                name = 'Xenium_Breast'
            chn, self.offset, raw_sz = 1, 0, resolution
            # compatabile with crop exp (e.g. 96 x 96) for Xenium (lung)
            if name == 'Xenium_Lung':
                raw_sz = int(resolution * 0.75)
            data = f'{name.split("_")[-1]}*/rna/*_rna.npz'
        self.raw_sz = raw_sz
        self._image_fnames = list(Path(path).glob(data))
        raw_shape = [len(self._image_fnames),
                     chn, resolution, resolution]
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def _load_raw_image(self, raw_idx):
        pth = str(self._image_fnames[raw_idx])
        if 'CosMx' in self._name:
            msk_pth = pth.replace('GeneLabels', 'CellLabels')
            msk = cv2.imread(msk_pth.replace('.npz', '.tif'),
                             flags=cv2.IMREAD_UNCHANGED)
            img_pth = pth.replace('GeneLabels', 'CellComposite')
            img_pth = img_pth.replace('.npz', '.jpg')
        elif 'Xenium' in self._name:
            msk_pth = pth.replace('rna.npz', 'msk.npz')
            msk = sparse.load_npz(msk_pth)
            img_pth = pth.replace('_rna.npz', '.jpg')
            if 'Breast' in self._name: 
                img_pth = img_pth.replace('rna', 'hne')
            else:
                img_pth = img_pth.replace('rna', 'dapi')
        img = pyvips.Image.new_from_file(img_pth)
        

        # Get coord for the randomly cropped tile
        for _ in range(4):
            left = random.randint(self.offset, img.width - self.raw_sz * 2)
            top = random.randint(self.offset, img.height - self.raw_sz * 2)
            # Sample util we get the non-black tile
            if msk[top: top + self.raw_sz * 2,
                   left: left + self.raw_sz * 2].sum() != 0:
                break
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
        return tiln, gene, left, top

    def _load_raw_labels(self):
        # Data/CosMx/Liver1/GeneLabels/CellLabels_F001.npz
        # For the above case, label = 1 - 1 = 0
        labels = [int(f.parent.parent.name[-1]) -
                  1 for f in self._image_fnames]
        labels = np.array(labels).astype(np.int64)
        return labels

    def __getitem__(self, idx):
        image, gene, left, top = self._load_raw_image(self._raw_idx[idx])

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
        return img / 127.5 - 1, gene, self.get_label(idx)

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


if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader
    parser = argparse.ArgumentParser(description='StyleGAN2 data class')
    parser.add_argument('--data', type=str, help='name of the dataset')
    args = parser.parse_args()

    if args.data == 'CosMx':
        gene_num = 1000
    elif args.data == 'Xenium':
        gene_num = 392
    elif args.data == 'Xenium_brain':
        gene_num = 319
    elif args.data == 'Xenium_breast':
        gene_num = 280

    data = InfDataset(f'Data/{args.data}', gene_num, transform=False, use_labels=True)
    print(len(data))
    dload = DataLoader(data, batch_size=8, shuffle=True,
                       **{'drop_last': True,
                          'num_workers': 8,
                          'pin_memory': True})
    for i, (img, gene, lab) in enumerate(dload):
        print(i, img.shape, gene.shape, lab.shape)