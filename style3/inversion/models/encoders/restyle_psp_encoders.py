import torch
import numpy as np
import spconv.pytorch as spconv

from torchvision.models.resnet import resnet34
from torch.nn import Conv2d, BatchNorm2d, PReLU, LeakyReLU, Sequential, Module, ModuleList, LayerNorm

from style3.models.stylegan2.model import EqualLinear
from SCLIP.CLIP import CLIPModel
from SCLIP.train_options import TrainOptions 


class Encoder(Module):
    def __init__(self, n_styles=12, opts=None, m_dim=512):
        super(Encoder, self).__init__()
        self.gene_type = opts.gene_type
        if self.gene_type == 'spatial':
            assert opts.dataset_type in ('CosMx', 'Xenium')
            g_dim = n_styles

        stride = 1 if opts.output_size == 128 else 2
        self.conv1 = Conv2d(
            opts.input_nc, 64, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = PReLU(64)

        resnet_basenet = resnet34(weights='ResNet34_Weights.DEFAULT')
        blocks = [
            resnet_basenet.layer1,
            resnet_basenet.layer2,
            resnet_basenet.layer3,
            resnet_basenet.layer4
        ]
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)
        self.body = Sequential(*modules)

        self.styles = ModuleList()
        self.style_count = n_styles
        for _ in range(self.style_count):
            self.styles.append(GradualStyleBlock(512, m_dim, 16))
        self.ln = LayerNorm(m_dim)

        ckpt = torch.load(f'SCLIP/{opts.dataset_type}.pt', 
                          map_location='cpu')
        cfg = TrainOptions(**ckpt['cfg'])
        cfg.dropout = 0
        cfg.trainable = False
        print('config for CLIP', cfg)
        self.clip = CLIPModel(cfg)
        self.clip.load_state_dict(ckpt['state_dict'])

        if self.gene_type == 'spatial':
            self.sp = spconv.SparseSequential(
                spconv.SparseConv2d(opts.gene_num, 64, 5, padding=2),
                spconv.ToDense(),
                BatchNorm2d(64),
                PReLU(64))

            resnet_basenet = resnet34(weights='ResNet34_Weights.DEFAULT')
            blocks = [
                resnet_basenet.layer1,
                resnet_basenet.layer2,
                resnet_basenet.layer3,
                resnet_basenet.layer4
            ]
            modules = []
            for block in blocks:
                for bottleneck in block:
                    modules.append(bottleneck)
            self.bd = Sequential(*modules)

            self.spats = ModuleList()
            self.spat_count = n_styles
            for _ in range(self.spat_count):
                self.spats.append(GradualStyleBlock(512, g_dim, 16))

            print(self.sp)
            print(self.bd)
        self.gene_num = opts.gene_num

    def forward(self, img, gene=None, lat=None):
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.body(x)

        lats = []
        for i in range(self.style_count):
            lats.append(self.styles[i](x))
        lats = [self.clip.image_projection(self.clip.image_encoder(img)).repeat(1, 2), 
                torch.stack(lats, dim=1)]

        if self.gene_type == 'spatial':
            gene = spconv.SparseConvTensor.from_dense(gene)
            gene = self.bd(self.sp(gene))
            genes = []
            for i in range(self.spat_count):
                genes.append(self.spats[i](gene))
            genes = torch.stack(genes, dim=1)
            lats = lats + genes @ lats

        return lats


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                LeakyReLU()
            ]
        self.convs = Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualTabBlock(Module):
    def __init__(self, in_c, out_c, mid_c=512, n_mlp=4):
        super(GradualTabBlock, self).__init__()
        modules = []
        for i in range(n_mlp):
            in_dim = in_c if i == 0 else mid_c
            if i != n_mlp - 1:
                out_dim, lr_mlp, act = mid_c, 0.01, True
            else:
                out_dim, lr_mlp, act = out_c, 1, None
            mlp = EqualLinear(in_dim, out_dim,
                              lr_mul=lr_mlp, activation=act)
            modules.append(mlp)
        self.mlp = Sequential(*modules)

    def forward(self, x):
        return self.mlp(x)
