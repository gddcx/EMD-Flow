import torch.nn as nn


class UpSampleMask8(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up_sample_mask = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=64 * 9, kernel_size=1, stride=1)
        )

    def forward(self, data):
        """
        :param data:  B, C, H, W
        :return:  batch, 8*8*9, H, W
        """
        mask = self.up_sample_mask(data)  # B, 64*6, H, W
        return mask


class UpSampleMask4(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up_sample_mask = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=16 * 9, kernel_size=1, stride=1)
        )

    def forward(self, data):
        """
        :param data:  B, C, H, W
        :return:  batch, 8*8*9, H, W
        """
        mask = self.up_sample_mask(data)  # B, 64*6, H, W
        return mask

