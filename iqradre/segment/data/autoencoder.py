import os
import random
import pathlib
import torch.utils.data as data
import PIL
import PIL.Image
from torchvision.datasets import ImageFolder
import pandas as pd


class AutoEncoderDataset(data.Dataset):
    def __init__(self, root, feature_dirname=None, target_dirname=None,
                 feature_transform=None, pair_transform=None, target_transform=None, **kwargs):
        super(AutoEncoderDataset, self).__init__()
        self.root = pathlib.Path(root)
        self.feature_dirname = feature_dirname
        self.target_dirname = target_dirname
        self.feature_path: pathlib.Path = None
        self.target_path: pathlib.Path = None
        self.feature_files = None
        self.target_files = None
        self.feature_transform = feature_transform
        self.pair_transform = pair_transform
        self.target_transform = target_transform
        self.kwargs = kwargs
        self._build_path()
        self._build_files()
        self._build_usage()

    def _build_usage(self):
        size = self.kwargs.get("size", 1)
        total = int(self.__len__() * size)
        self.feature_files = self.feature_files[0:total]
        self.target_files = self.target_files[0:total]

    def _build_path(self):
        if self.feature_dirname is None:
            self.feature_path: pathlib.Path = self.root.joinpath('feature')
        else:
            self.feature_path: pathlib.Path = self.root.joinpath(self.feature_dirname)

        if self.target_dirname is None:
            self.target_path = self.root.joinpath('target')
        else:
            self.target_path = self.root.joinpath(self.target_dirname)

    def _build_files(self):
        self.feature_files = sorted(list(self.feature_path.glob("*")))
        self.target_files = sorted(list(self.target_path.glob("*")))
        feat_len = len(self.feature_files)
        targ_len = len(self.target_files)
        assert feat_len == targ_len, "Total files from feature dir and target " \
                                     "dir is not equal, expected equal number"

    def __len__(self):
        feat_len = len(list(self.feature_files))
        return feat_len

    def __getitem__(self, idx: int):
        feature_path = self.feature_files[idx]
        target_path = self.target_files[idx]

        feature = PIL.Image.open(feature_path)
        target = PIL.Image.open(target_path)

        if self.feature_transform:
            feature = self.feature_transform(feature)

        if self.pair_transform:
            feature, target = self.pair_transform(feature, target)

        if self.target_transform:
            target = self.target_transform(target)

        return feature, target




if __name__ == '__main__':
    # from torchwisdom.vision.transforms import transforms as ptransforms
    
    # train_tmft = ptransforms.PairCompose([
    #     ptransforms.PairResize((220)),
    #     ptransforms.PairRandomRotation(20),
    #     ptransforms.PairToTensor(),
    # ])
    # root = '/data/att_faces_new/valid'
    # sd = SiamesePairDataset(root, ext="pgm", pair_transform=train_tmft)
    # # loader = data.DataLoader(sd, batch_size=32, shuffle=True)
    # print(sd.__getitem__(0))
    ...