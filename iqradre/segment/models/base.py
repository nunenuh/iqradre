import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# this code is inspired from  milesial repository
# https://github.com/milesial/Pytorch-UNet
# some of the code is taken from that repo and I make changes
# for the thing I need to change

def conv_bn_relu(in_ch, out_ch, ksize=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=ksize, padding=padding, stride=stride),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

def conv_bn_lrelu(in_ch, out_ch, ksize=3, padding=1, stride=1, neg_slope=0.2):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=ksize, padding=padding, stride=stride),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(inplace=True, negative_slope=neg_slope)
    )

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, activation='relu'):
        super(DoubleConv, self).__init__()
        if activation=='relu':
            self.conv = nn.Sequential(
                conv_bn_relu(in_ch, out_ch),
                conv_bn_relu(out_ch, out_ch)
            )
        elif activation=='lrelu':
            self.conv = nn.Sequential(
                conv_bn_lrelu(in_ch, out_ch),
                conv_bn_lrelu(out_ch, out_ch)
            )

    def forward(self, x):
        return self.conv(x)

class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        x = self.conv(x)
        return x

class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch, pool='maxpool'):
        super(DownConv, self).__init__()

        if pool=='maxpool':
            self.down = nn.MaxPool2d(2)
        elif pool=='stride':
            self.down = nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2, padding=1)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch, mode='bilinear'):
        super(UpConv, self).__init__()
        if mode=='bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        elif mode=='nearest':
            self.up = nn.Upsample(scale_factor=2, mode='nearest', align_corners=True)
        elif mode=='transpose':
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def _padfix(self, x1, x2):
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        return x1, x2

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1, x2 = self._padfix(x1, x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch, use_upsample=False, mode='bilinear'):
        super(OutConv, self).__init__()
        self.use_upsample = use_upsample
        if use_upsample:
            if mode == 'bilinear':
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            elif mode == 'nearest':
                self.up = nn.Upsample(scale_factor=2, mode='nearest', align_corners=True)
            elif mode == 'transpose':
                self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        if self.use_upsample:
            x  = self.up(x)
        x = self.conv(x)
        return x


if __name__== '__main__':
    pass