import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet
from torchvision.models.resnet import model_urls
import torch.utils.model_zoo as model_zoo
from .base import *


__all__ = ['UNet']


# this code is inspired from  milesial repository
# https://github.com/milesial/Pytorch-UNet
# some of the code is taken from that repo and I make changes
# for the thing I need to change

class UNetEncoder(nn.Module):
    def __init__(self, in_chan, start_feat=64):
        super(UNetEncoder, self).__init__()
        self.out_chan = start_feat * 8
        self.inconv = InConv(in_chan, start_feat)
        self.down1 = DownConv(start_feat, start_feat*2)
        self.down2 = DownConv(start_feat*2, start_feat*4)
        self.down3 = DownConv(start_feat*4, start_feat*8)
        self.down4 = DownConv(start_feat*8, start_feat*8)

    def forward(self, x):
        inc = self.inconv(x)
        dc1 = self.down1(inc)
        dc2 = self.down2(dc1)
        dc3 = self.down3(dc2)
        dc4 = self.down4(dc3)
        return dc4, dc3, dc2, dc1, inc


class UNetDecoder(nn.Module):
    def __init__(self, in_chan, n_classes):
        super(UNetDecoder, self).__init__()
        self.up1 = UpConv(in_chan, in_chan//4)
        self.up2 = UpConv(in_chan//2, in_chan//8)
        self.up3 = UpConv(in_chan//4, in_chan//16)
        self.up4 = UpConv(in_chan//8, in_chan//16)
        self.outconv = OutConv(in_chan//16, n_classes)

    def forward(self, dc4, dc3, dc2, dc1, inc):
        up1 = self.up1(dc4, dc3)
        up2 = self.up2(up1, dc2)
        up3 = self.up3(up2, dc1)
        up4 = self.up4(up3, inc)
        out = self.outconv(up4)
        return out


class UNet(nn.Module):
    def __init__(self, in_chan, n_classes, start_feat=64):
        super(UNet, self).__init__()
        self.encoder_in_chan = in_chan
        self.decoder_in_chan = start_feat * 16
        self.start_feat = start_feat

        self.encoder = UNetEncoder(in_chan=self.encoder_in_chan, start_feat=start_feat)
        self.decoder = UNetDecoder(in_chan=self.decoder_in_chan, n_classes=n_classes)

    def forward(self, x):
        dc4, dc3, dc2, dc1, inc = self.encoder(x)
        out = self.decoder(dc4, dc3, dc2, dc1, inc)
        return out



if __name__ == '__main__':
    pass
    # model = ResNetUNet(in_chan=3, n_classes=1, pretrained=True, version=18)
    # input = torch.rand(1, 3, 224, 224)
    # out = model(input)
    # print(out.shape)

    # expansion = 4
    # input = torch.rand(1, 3, 224, 224)
    # net = resnet.resnet50()
    # x = net.conv1(input)
    # x = net.bn1(x)
    # x = net.relu(x)
    # print('inconv',x.shape)
    # inconv = x
    #
    # x = net.maxpool(x)
    # x = net.layer1(x)
    # print('layer1',x.shape)
    # dc1 = x
    #
    # x  = net.layer2(x)
    # print('layer2',x.shape)
    # dc2 = x
    #
    # x = net.layer3(x)
    # print('layer3',x.shape)
    # dc3=x
    #
    # x = net.layer4(x)
    # print('layer4',x.shape)
    # dc4=x
    #
    # up1 = UpConv(dc4.size(1)+dc3.size(1), 512)(dc4, dc3)
    # print(up1.shape)
    #
    # up2 = UpConv(up1.size(1)+dc2.size(1), 256)(up1, dc2)
    # print(up2.shape)
    #
    # up3 = UpConv(up2.size(1)+dc1.size(1), 128)(up2, dc1)
    # print(up3.shape)
    #
    # up4 = UpConv(up3.size(1) + inconv.size(1), 64)(up3, inconv)
    # print(up4.shape)
    #
    # outconv = OutConv(64, 1, use_upsample=True)(up4)
    # print(outconv.shape)