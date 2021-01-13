import os
import random
import pathlib
import torch.utils.data as data
import PIL
import PIL.Image
from torchvision.datasets import ImageFolder
import pandas as pd

class PairDataset(data.Dataset):
    def __init__(self, root, feature_suffix='_image.jpg', target_suffix='_mask.jpg', 
                 feature_transform=None, pair_transform=None, target_transform=None,
                 mode='train', val_split=0.2,
                 **kwargs):
        super(PairDataset, self).__init__()
        self.root = pathlib.Path(root)
        self.feature_suffix = feature_suffix
        self.target_suffix = target_suffix
        self.mode = mode
        self.val_split = val_split
        self.feature_files = None
        self.target_files = None
        self.feature_transform = feature_transform
        self.pair_transform = pair_transform
        self.target_transform = target_transform
        self.kwargs = kwargs
        self._build_files()
        self._split_validation()
        
    def _build_files(self):
        self.feature_files = sorted(list(self.root.glob(f"*{self.feature_suffix}")))
        self.target_files = sorted(list(self.root.glob(f"*{self.target_suffix}")))
        feat_len = len(self.feature_files)
        targ_len = len(self.target_files)
        assert feat_len == targ_len, "Total files from feature dir and target " \
                                     "dir is not equal, expected equal number"

    def _split_validation(self):
        total = len(self.feature_files)
        val_total = int(total * self.val_split)
        trx_total = int(total - val_total)
        index_list = [i for i in range(total)]
        random.seed(1261)
        trx_indices = random.sample(index_list, k=trx_total)
        val_indices = list(set(index_list) - set(trx_indices))
        
        if self.mode == "train":
            self.feature_files = list(map(lambda i: self.feature_files[i], trx_indices))
            self.target_files = list(map(lambda i: self.target_files[i], trx_indices))
        elif self.mode == "valid":
            self.feature_files = list(map(lambda i: self.feature_files[i], val_indices))
            self.target_files = list(map(lambda i: self.target_files[i], val_indices))
        else:
            raise Exception("supported mode only 'train' and 'valid'")
        
        
    def __len__(self):
        feat_len = len(list(self.feature_files))
        return feat_len
    
    def get_files(self, idx):
        feature_path = self.feature_files[idx]
        target_path = self.target_files[idx]
        
        fname = feature_path.name.split("_")[0]
        tname = target_path.name.split("_")[0]
        
        assert fname.split("_")[0] == tname.plit("_")[0], f"feature name {fname} is not same file with target name {tname}"

    def __getitem__(self, idx: int):
        feature_path = self.feature_files[idx]
        target_path = self.target_files[idx]
        
#         print(feature_path.name.split("_")[0], target_path.name.split("_")[0])

        feature = PIL.Image.open(feature_path)
        target = PIL.Image.open(target_path)

        if self.feature_transform:
            feature = self.feature_transform(feature)

        if self.pair_transform:
            feature, target = self.pair_transform(feature, target)

        if self.target_transform:
            target = self.target_transform(target)

        return feature, target
    