import torch


class PSNR(torch.nn.Module):
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self, mval=255.):
        super(PSNR, self).__init__()
        self.mval = mval

    def forward(self, img1, img2):
        if len(img1.shape) == 3:
            dim = [1, 2]
        elif len(img1.shape) == 4:
            dim = [1, 2, 3]
        mse = torch.mean((img1 - img2) ** 2, dim=dim)
        return 20 * torch.log10(self.mval / torch.sqrt(mse))
