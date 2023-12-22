
import os
import cv2
import clip
import math
import torch
import random
import zipfile
import argparse
import torchvision

import numpy as np
import pandas as pd
import torchvision.transforms.functional_tensor as F

from tqdm import tqdm
from PIL import Image, ImageFile
from pathlib import Path
from torch.linalg import svdvals
from cleanfid.fid import build_feature_extractor
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CLIP_fx():
    def __init__(self, name="ViT-B/32", device="cuda"):
        self.model, _ = clip.load(name, device=device)
        self.model.float().eval()
        self.name = "clip_"+name.lower().replace("-","_").replace("/","_")
    
    def __call__(self, img_t):
        img_x = F.normalize(img_t, 
                            (0.48145466, 0.4578275, 0.40821073), 
                            (0.26862954, 0.26130258, 0.27577711))
        assert torch.is_tensor(img_x)
        if len(img_x.shape)==3:
            img_x = img_x.unsqueeze(0)
        B,C,H,W = img_x.shape
        with torch.no_grad():
            z = self.model.encode_image(img_x)
        return z


def post_dim(x, batch_dim=False):
    assert len(x.shape) in (3, 4)
    if len(x.shape) == 3 and batch_dim:
        x = x[None]
    if len(x.shape) == 3:
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        elif x.shape[0] == 2:
            x = torch.cat((torch.zeros_like(x[0])[None], x), 0)
    else:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 2:
            x = torch.cat((torch.zeros_like(x[:, 0])[:, None], x), 1)
    return x


def fn_resize(x, sz, mode='clean', batch_dim=False):
    assert 'float32' in str(x.dtype)
    if mode == 'clip':
        x = x.byte()
    x = F.resize(x, sz, 'bicubic', True).clamp(0, 255)
    if mode == 'clip':
        # simulate toTensor
        x = x.float().div(255)
    x = post_dim(x, batch_dim)
    return x


# def resize_vit(output_size, tsr_4=False):
#     resize = transforms.Resize(output_size,
#                                interpolation=transforms.InterpolationMode.BICUBIC,
#                                antialias=True)
#     crop = transforms.CenterCrop(output_size)

#     def func(x):
#         if not torch.is_tensor(x):
#             x = torch.from_numpy(x.transpose(2, 0, 1)).byte()
#         x = crop(resize(x)).clamp(0, 255).byte()
#         x = x.float().div(255) * 255
#         x = post_dim(x, tsr_4)
#         return x
#     return func


class ResizeDataset(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, mode, size=(299, 299), fdir=None):
        self.files = files
        self.mode = mode
        self.fdir = fdir
        self.transforms = torchvision.transforms.ToTensor()
        self.size = size
        self.fn_resize = lambda x: x
        self.custom_image_tranform = lambda x: x
        self._zipfile = None

    def _get_zipfile(self):
        assert self.fdir is not None and '.zip' in self.fdir
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self.fdir)
        return self._zipfile

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = str(self.files[i])
        if self.fdir is not None and '.zip' in self.fdir:
            with self._get_zipfile().open(path, 'r') as f:
                img_np = np.array(Image.open(f).convert('RGB'))
        elif ".npy" in path:
            img_np = np.load(path)
        else:
            img_pil = Image.open(path).convert('RGB')
            img_np = np.array(img_pil).transpose((2, 0, 1))

        img = torch.from_numpy(img_np).contiguous().float()
        if 'Xenium' in path:
            img = img[0][None]
        elif 'CosMx' in path:
            img = img[1:]
        if img.shape[1] != 128 or img.shape[2] != 128:
            img = F.resize(img, 128, 'bicubic', True).round().clamp(0, 255)
        
        if self.mode == 'clean':
            img_t = fn_resize(img, 299, self.mode)
        elif self.mode == 'clip':
            img_t = fn_resize(img, 224, self.mode)
        # apply a custom image transform before resizing the image to 299x299
        # img_np = self.custom_image_tranform(img_np)
        # fn_resize expects a np array and returns a np array
        # img_t = self.fn_resize(img_np)

        # # ToTensor() converts to [0,1] only if input in uint8
        # if img_resized.dtype == "uint8":
        #     img_t = self.transforms(np.array(img_resized))*255
        # elif img_resized.dtype == "float32":
        #     img_t = self.transforms(img_resized)
        # else:
        #     img_t = img_resized
        return img_t


class ResizeDataset_mouse(torch.utils.data.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, mode, size=(299, 299), fdir=None):
        self.mode = mode
        self.root = Path('Data/Xenium_mouse')
        self._x = files.vertex_x.values.tolist()
        self._y = files.vertex_y.values.tolist()
        self._img = files.img_id.values.tolist()
    
    def __len__(self):
        return len(self._img)
    
    def __getitem__(self, i, sz=64):
        pth = self.root / 'dapi' / f'{self._img[i]}.jpg'
        img = cv2.imread(str(pth))
        row = slice(max(self._y[i] - sz, 0),
                    min(self._y[i] + sz, img.shape[0]))
        col = slice(max(self._x[i] - sz, 0),
                    min(self._x[i] + sz, img.shape[1]))
        img_cell = (img[row, col]).transpose((2, 0, 1))
        img_cell = torch.from_numpy(img_cell).contiguous().float()

        if self.mode == 'clean':
            out = fn_resize(img_cell, 299, self.mode)
        elif self.mode == 'clip':
            out = fn_resize(img_cell, 224, self.mode)

        return out


EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
              'tif', 'tiff', 'webp', 'npy', 'JPEG', 'JPG', 'PNG'}


def get_batch_features(batch, model, device):
    with torch.inference_mode():
        feat = model(batch.to(device))
    return feat.detach()


def get_files_features(l_files, model=None, num_workers=12,
                       batch_size=128, device=torch.device("cuda"),
                       mode="clean", custom_fn_resize=None,
                       description="", fdir=None, verbose=True,
                       custom_image_tranform=None):
    # wrap the images in a dataloader for parallelizing the resize operation
    if isinstance(l_files, np.ndarray):
        dataset = ResizeDataset(l_files, fdir=fdir, mode=mode)
    else:
        dataset = ResizeDataset_mouse(l_files, fdir=fdir, mode=mode)
    if custom_image_tranform is not None:
        dataset.custom_image_tranform = custom_image_tranform
    if custom_fn_resize is not None:
        dataset.fn_resize = custom_fn_resize

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size, shuffle=False,
                                             drop_last=False, num_workers=num_workers)

    # collect all inception features
    l_feats = []
    if verbose:
        pbar = tqdm(dataloader, desc=description)
    else:
        pbar = dataloader

    for batch in pbar:
        l_feats.append(get_batch_features(batch, model, device))
    feats = torch.cat(l_feats)
    return feats


def make_custom_stats(name, path, num=None, mode="clean", model_name="inception_v3",
                      num_workers=0, batch_size=64, device=torch.device("cuda"),
                      verbose=True, shuffle=False, seed=None, custom_fn_resize=None,
                      buffer=1000000):
    stats_folder = "stats_inf" if "mouse" in name else "stats"
    os.makedirs(stats_folder, exist_ok=True)
    split, res = "custom", "na"
    if model_name == "inception_v3":
        model_modifier = ""
    else:
        model_modifier = "_"+model_name
    outf = os.path.join(
        stats_folder, f"{name}_{mode}.pt".lower())
    # if the custom stat file already exists
    if os.path.exists(outf):
        msg = f"The statistics file {name} already exists. "
        msg += "Use remove_custom_stats function to delete it first."
        raise Exception(msg)
    if model_name == "inception_v3":
        feat_model = build_feature_extractor(mode, device)
        custom_fn_resize = None
        custom_image_tranform = None
    elif model_name == "clip_vit_b_32":
        clip_fx = CLIP_fx("ViT-B/32")
        feat_model = clip_fx
        custom_fn_resize = None
        custom_image_tranform = None
    else:
        raise ValueError(
            f"The entered model name - {model_name} was not recognized.")

    # get all relevant files in the dataset
    if verbose:
        print(f"Found {len(path)} images")
    # use a subset number of files if needed
    if num is not None and "mouse" not in name:
        if shuffle:
            random.seed(seed)
            random.shuffle(path)
        path = path[:num]
    feats = get_files_features(path, feat_model, num_workers=num_workers,
                               batch_size=batch_size, device=device, mode=mode,
                               custom_fn_resize=custom_fn_resize,
                               custom_image_tranform=custom_image_tranform,
                               verbose=verbose)
    if feats.shape[0] > buffer:
        mu, scm, buf, tot = 0, 0, buffer // 5,  feats.shape[0]
        bat = math.ceil(tot / buf)
        for b in range(bat):
            f = feats[b * buf: (b + 1) * buf].double()
            mu += f.sum(0)
            scm += f.T @ f
        mu /= tot
        scm /= tot
        sigma = scm - mu[:, None] @ mu[None]
    else:
        feats = feats.double()
        mu = feats.mean(0)
        scm = (feats.T @ feats) / feats.shape[0]
        sigma = feats.T.cov(correction=0)
        _err = (scm - mu[:, None] @ mu[None]) - sigma
        print(f'{_err.abs().max()}', feats.shape, scm.shape, sigma.shape)
    torch.save((mu, sigma, svdvals(scm)), outf)
    del mu, sigma, scm


def split_df(df, name, split, subset=None):
    sdct, sub = {}, ''
    if subset is not None:
        if subset[0] != 'cellType':
            df = df[df[subset[0]] == subset[1]]
        else:
            assert name == 'CosMx'
            cnd1 = (df['slide_ID_numeric'] == 1) & (df['cellType'].str.contains('Hep.'))
            cnd2 = (df['slide_ID_numeric'] == 2) & (df['cellType'].str.contains('tumor_'))
            df = df[cnd1 | cnd2]
        sub = f'_{subset[0]}'
    if split:
        spl = np.unique(df[split])
        sdct = {
            f'{name}{sub}_{split}_{s}': df[df[split] == s].path.values for s in spl}
    else:
        sdct[f'{name}{sub}'] = df.path.values
    return sdct


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare the image stats')
    parser.add_argument('--data',
                        type=str,
                        help='Name of the ST dataset.')
    parser.add_argument('--model',
                        type=str,
                        default='inception_v3',
                        choices=['inception_v3', 'clip_vit_b_32'],
                        help='Whether to use custom resize for calc stats.')
    parser.add_argument('--subtype',
                        action='store_true',
                        help='Whether only calc stats w.r.t. cell subtypes.')
    parser.add_argument('--crop', default=False,
                        help='crop method: cell-or tile-based')
    args = parser.parse_args()

    if args.model == 'inception_v3':
        mode = 'clean'
    elif args.model == 'clip_vit_b_32':
        mode = 'clip'

    root = Path(f'Data/{args.data}/GAN/crop')
    if args.data in ('CosMx', 'Xenium'):
        df = pd.read_csv(str(root / 'metadata.csv'))
        df.path = df.path.map(lambda x: str(root / x))
        print(df.head())
        split, subset = ['slide_ID_numeric', ], None
        if args.subtype:
            if args.data == 'CosMx':
                subset = ('cellType', None)
            elif args.data == 'Xenium':
                subset = ('kmeans_2_clusters', 2)
        for spl in split:
            sdct = split_df(df, args.data, spl, subset)
            for dnm, pth in sdct.items():
                print(dnm, len(pth))
                make_custom_stats(dnm, pth,
                                  mode=mode,
                                  model_name=args.model,
                                  num_workers=8)
    elif args.data == 'Xenium_mouse':
        res = 128
        if args.crop == 'tile':
            root = root.parent / 'crop_tile'
        df = pd.read_csv(str(root / 'metadata.csv'),
                         index_col=0)
        valid_h = (df['vertex_y'] >= res) & \
            (df['vertex_y'] < df['height'] - res)
        valid_w = (df['vertex_x'] >= res) & \
            (df['vertex_x'] < df['width'] - res)
        df = df[(valid_h) & (valid_w)]
        df = df[['vertex_y', 'vertex_x', 'img_id']]
        print(len(df))
        make_custom_stats(f'{args.data}_{args.crop}', df,
                          mode=mode,
                          model_name=args.model,
                          num_workers=8)
