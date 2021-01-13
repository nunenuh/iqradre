import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet
from torchvision.models.resnet import model_urls
import torch.utils.model_zoo as model_zoo
from .base import *



class ResNetUNetEncoder(resnet.ResNet):
    def __init__(self, block, layers, num_classes=1000, in_chan=3):
        super(ResNetUNetEncoder, self).__init__(block, layers, num_classes)
        self.expansion = block.expansion
        self.last_chan = block.expansion * 512
        self.conv1 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        inc = x

        x = self.maxpool(x)
        x = self.layer1(x)
        dc1 = x

        x = self.layer2(x)
        dc2 = x

        x = self.layer3(x)
        dc3 = x

        x = self.layer4(x)
        dc4 =x

        return dc4, dc3, dc2, dc1, inc


class ResNetUNetDecoder(nn.Module):
    def __init__(self, in_chan, expansion, n_classes):
        super(ResNetUNetDecoder, self).__init__()
        self.in_chan = in_chan
        self.expansion = expansion
        self.n_classes = n_classes
        self.chan = self._generate_chan()

        self.up1 = UpConv(self.chan[0]+self.chan[1], 256)
        self.up2 = UpConv(self.chan[2]+256, 128)
        self.up3 = UpConv(self.chan[3]+128, 64)
        self.up4 = UpConv(self.chan[4]+64, 64)
        self.outconv = OutConv(64, n_classes, use_upsample=True)

    def _generate_chan(self):
        chan = []
        tmp_chan = self.in_chan
        for i in range(5):
            chan.append(tmp_chan)
            if self.expansion == 4:
                if i + 1 == 4:
                    tmp_chan = tmp_chan // 4
                else:
                    tmp_chan = tmp_chan // 2
            elif self.expansion == 1:
                if i + 1 == 4:
                    tmp_chan = tmp_chan
                else:
                    tmp_chan = tmp_chan // 2
        return chan

    def forward(self, dc4, dc3, dc2, dc1, inc):
        up1 = self.up1(dc4, dc3)
        up2 = self.up2(up1, dc2)
        up3 = self.up3(up2, dc1)
        up4 = self.up4(up3, inc)
        out = self.outconv(up4)
        return out

class ResNetUNet(nn.Module):
    def __init__(self, in_chan, n_classes, pretrained=True, version=18):
        super(ResNetUNet, self).__init__()
        self.pretrained = pretrained
        self.version = version
        self.in_chan = in_chan

        if in_chan!=3 and pretrained==True:
            raise ValueError("in_chan has to be 3 when you set pretrained=True")

        self.encoder = self._build_resnet()
        self.decoder = ResNetUNetDecoder(in_chan=self.encoder.last_chan, expansion=self.encoder.expansion, n_classes=n_classes)

    def _build_resnet(self):
        block = self._get_block()
        ver = self.version
        name_ver = 'resnet'+str(ver)
        if ver>=50:
            model = ResNetUNetEncoder(resnet.Bottleneck, block[str(ver)], in_chan=self.in_chan)
        else:
            model = ResNetUNetEncoder(resnet.BasicBlock, block[str(ver)], in_chan=self.in_chan)
        if self.pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls[name_ver]))
        return model

    def _get_block(self):
        return {'18': [2, 2, 2, 2], '34': [3, 4, 6, 3],
                '50': [3, 4, 6, 3], '101': [3, 4, 23, 3],
                '152': [3, 8, 36, 3]}

    def forward(self, x):
        dc4, dc3, dc2, dc1, inc = self.encoder(x)
        out = self.decoder(dc4, dc3, dc2, dc1, inc)
        return out



if __name__ == "__main__":
    ...