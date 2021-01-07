# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class dsconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(dsconv, self).__init__()
        self.depthconv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1,
                                   groups=in_ch)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pointconv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.depthconv(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.pointconv(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)

        return x


class double_dsconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_dsconv, self).__init__()
        self.conv = nn.Sequential(
            dsconv(in_ch, out_ch),
            dsconv(out_ch, out_ch),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv_dsc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv_dsc, self).__init__()
        self.conv = double_dsconv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down_dsc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_dsc, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_dsconv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up_dsc(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_dsc, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_dsconv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv_dsc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv_dsc, self).__init__()
        self.conv = dsconv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class shortcut_fusion(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(shortcut_fusion, self).__init__()
        self.fusion = nn.Conv2d(in_ch, out_ch, 1)
        self.relu = nn.ReLU(True)

    def forward(self, x1, x2):
        x = self.fusion(torch.cat([x1, x2], 1))
        x = self.relu(x)
        return x


class dsc_fusion(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(dsc_fusion, self).__init__()
        self.deepwise = nn.Conv2d(in_ch, in_ch, stride=1, kernel_size=3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self.relu = nn.ReLU(True)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        x = self.deepwise(x)
        x = self.pointwise(x)
        x = self.relu(x)
        return x


class SE_block(nn.Module):
    def __init__(self, n_chn, ratio):
        super(SE_block, self).__init__()
        self.GPA = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(n_chn, int(n_chn / ratio))
        self.relu = nn.ReLU(True)
        self.fc2 = nn.Linear(int(n_chn / ratio), n_chn)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_keep = x
        b, c, h, w = x_keep.shape
        x = self.GPA(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x.view(b, c, 1, 1) * x_keep

