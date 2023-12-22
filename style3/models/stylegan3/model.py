import torch
import pickle

from style3.models.stylegan2.model import Generator as G2
from style3.models.stylegan3.networks_stylegan3 import Generator as G3


class Decoder(torch.nn.Module):

    def __init__(self, model, opts, ckpt_pth=None,
                 cbase=32678, cmax=512):
        super(Decoder, self).__init__()
        if model == 'style3':
            if str(ckpt_pth).endswith("pkl"):
                with open(ckpt_pth, "rb") as f:
                    self.decoder = pickle.load(f)['G_ema']
            else:
                if opts.use_cfgr:
                    cbase, cmax = cbase * 2, cmax * 2
                self.decoder = G3(z_dim=512,
                                  g_dim=opts.gene_num,
                                  w_dim=512,
                                  img_resolution=opts.output_size,
                                  img_channels=opts.input_nc,
                                  mapping_kwargs={'conv_kernel': opts.kernel_size},
                                  conv_kernel=opts.kernel_size,
                                  channel_base=cbase,
                                  channel_max=cmax,
                                  use_radial_filters=opts.use_cfgr)
                if ckpt_pth is not None:
                    self._load_checkpoint(ckpt_pth)
        elif model == 'style2':
            self.decoder = G2(opts.output_size, opts.gene_num, 512, opts.kernel_size, 8,
                              img_chn=opts.input_nc)
            if ckpt_pth is not None:
                ckpt = torch.load(ckpt_pth, map_location='cpu')
                self.decoder.load_state_dict(ckpt['g_ema'], strict=True)
        print(self.decoder)

    def _load_checkpoint(self, ckpt):
        try:
            self.decoder.load_state_dict(torch.load(ckpt), strict=True)
        except:
            ckpt = torch.load(ckpt)
            ckpt = {k: v for k, v in ckpt.items()
                    if 'synthesis.input.transform' not in k}
            self.decoder.load_state_dict(ckpt, strict=False)