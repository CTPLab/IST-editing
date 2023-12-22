import torch
from torch import nn

from style3.utils.data_utils import linspace
from style3.models.stylegan3.model import Decoder
from style3.inversion.models.encoders.restyle_psp_encoders import Encoder


class pSp(nn.Module):
    def __init__(self, opts):
        super(pSp, self).__init__()
        n_styles = 16 if opts.decoder == 'style3' else 12
        self.encoder = Encoder(n_styles, opts)

        epth = f'stats/{opts.dataset_type}_{opts.data_splt}_'
        epth = epth.lower()
        self.eigen = [torch.load(f'{epth}1_eig.pt'),
                      torch.load(f'{epth}2_eig.pt')]

        self.dnm = opts.decoder
        self.gene_num = opts.gene_num if opts.gene_use else 0
        ckpt = opts.stylegan_weights if opts.checkpoint_path is None else None
        self.decoder = Decoder(self.dnm, opts, ckpt).decoder

        if opts.checkpoint_path is not None:
            ckpt = torch.load(opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(self._get_keys(ckpt, 'encoder', []),
                                         strict=True)
            remove = ['synthesis.input.transform'] if self.dnm == 'style3' else []
            strict = False if self.dnm == 'style3' else True
            self.decoder.load_state_dict(self._get_keys(ckpt, 'decoder', remove),
                                         strict=strict)
            if self.dnm == 'style2':
                for layer_idx in range(self.decoder.num_layers):
                    res = (layer_idx + 5) // 2
                    shape = [opts.n_eval, 1, 2 ** res, 2 ** res]
                    setattr(self.decoder.noises, f'noise_{layer_idx}',
                            torch.randn(*shape))

    def edit_gene(self, x, eigx, eigr,
                  wei=None, topk=1, idx=None):
        if eigx is None:
            return x
        else:
            x = x.detach().clone()
            if len(x.shape) == 1:
                x = x[None]
            vecx, valx = eigx
            vecr, valr = eigr
            xedt = x @ vecx.float().to(x)
            if wei is None:
                wei = torch.ones_like(valx).float()
                wei[:topk] = (valr[:topk] / valx[:topk]).sqrt().float()
            xedt = wei[None].to(x) * xedt
            xedt = xedt @ vecr.T.float().to(x)
            if idx is not None:
                x[:, idx] = xedt[:, idx].clone()
                xedt = x
            xedt[xedt < 0] = 0
            return xedt

    def forward(self, x, gene,
                input_code=False, randomize_noise=True,
                return_latents=False,
                interp=False,
                shift=None):
        if len(gene.shape) == 4:
            gene = gene.sum((1, 2))

        if interp:
            gene = gene.detach().clone()
            for i in range(gene.shape[0]):
                r = torch.argmax(gene[i, self.gene_num:])
                gene[i, :self.gene_num] = self.edit_gene(gene[i, :self.gene_num],
                                                         self.eigen[r],
                                                         self.eigen[(r + 1) % 2])
        codes = x if input_code else self.encoder(x, gene)

        if self.dnm == 'style3':
            _, ds = self.decoder.mapping(torch.randn((gene.shape[0], 512)).to(gene),
                                         gene)
            images = self.decoder.synthesis(codes, ds,
                                            noise_mode='const', force_fp32=True)
        elif self.dnm == 'style2':
            images, codes = self.decoder([codes, gene],
                                         input_is_latent=True,
                                         randomize_noise=randomize_noise,
                                         return_latents=return_latents,
                                         shift=shift)

        if return_latents:
            return images, codes
        else:
            return images

    @staticmethod
    def _get_keys(d, name, remove):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items()
                  if k[:len(name)] == name and k[len(name) + 1:] not in remove}
        return d_filt
