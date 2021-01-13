import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet
from torchvision.models.resnet import model_urls
import torch.utils.model_zoo as model_zoo
from .base import *



class TuneableUNetEncoder(nn.Module):
    def __init__(self, in_chan, start_feat=64, deep=4):
        super(TuneableUNetEncoder, self).__init__()
        self.out_chan = start_feat * 8
        self.in_chan = in_chan
        self.start_feat = start_feat
        self.deep = deep
        self.chan = self._generate_chan()
        self.last_chan = self.chan[-1][-1]

        modules = self._make_layer(self.in_chan, self.start_feat)
        self.encoder = nn.Sequential(*modules)

    def _make_layer(self, in_chan, start_feat):
        modules = []
        modules.append(InConv(in_chan, start_feat))
        for d in range(self.deep):
           (in_chan, out_chan) = self.chan[d]
           modules.append(DownConv(in_chan, out_chan))
        return modules

    def _generate_chan(self):
        chan = []
        for d in range(self.deep):
            c1 = 2 ** d
            c2 = 2 ** (d + 1)
            if (d + 1) != self.deep:
                pair = (self.start_feat * c1, self.start_feat * c2)
            else:
                pair = (self.start_feat * c1, self.start_feat * c1)
            chan.append(pair)
        return chan

    def forward(self, x):
        output = []
        for d in range(self.deep+1):
            x = self.encoder[d](x)
            output.append(x)
        return output


class TuneableUNetDecoder(nn.Module):
    def __init__(self, in_chan, n_classes, deep=4):
        super(TuneableUNetDecoder, self).__init__()
        self.in_chan = in_chan
        self.n_classes = n_classes
        self.deep = deep
        self.chan = self._generate_chan()
        self.last_chan = self.chan[-1][-1]

        modules = self._make_layer()
        self.decoder = nn.Sequential(*modules)

    def _make_layer(self):
        modules = []
        for d in range(self.deep):
           (in_chan, out_chan) = self.chan[d]
           modules.append(UpConv(in_chan, out_chan))
        (in_chan, out_chan) = self.chan[self.deep]
        modules.append(OutConv(in_chan, self.n_classes))
        return modules

    def _generate_chan(self):
        chan = []
        self.in_chan = self.in_chan * 2
        for d in range(self.deep):
            c1 = self.in_chan // 2 ** (d)
            if (d + 1) != self.deep:
                c2 = self.in_chan // 2 ** (d + 2)
            else:
                c2 = self.in_chan // 2 ** (d + 1)
            pair = (c1, c2)
            chan.append(pair)

        output_pair = (c2, self.n_classes)

        chan.append(output_pair)
        return chan

    def forward(self, input):
        input.reverse()
        x = self.decoder[0](input[0], input[1])
        for i in range(1, self.deep+1):
            if i+1 != self.deep+1:
                x = self.decoder[i](x, input[i+1])
            else:
                x = self.decoder[self.deep](x)
        return x


class TuneableUNet(nn.Module):
    def __init__(self, in_chan, n_classes, start_feat=64, deep=4):
        super(TuneableUNet, self).__init__()
        self.in_chan = in_chan
        self.n_classes = n_classes
        self.start_feat = start_feat
        self.deep = deep
        
        self.encoder = TuneableUNetEncoder(
            in_chan=in_chan,
            start_feat=start_feat,
            deep=deep,
        )

        self.decoder = TuneableUNetDecoder(
            in_chan=self.encoder.last_chan,
            n_classes=n_classes,
            deep=deep
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    pass