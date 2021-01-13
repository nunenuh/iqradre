from ..transforms import transforms as PT
from .pair import PairDataset
import torch.utils.data as data

train_tmft = PT.PairCompose([
    PT.PairResize((256, 256)),
    PT.PairRandomRotation(20),
    PT.PairGrayscale(),
    PT.PairToTensor(),
])

valid_tmft = PT.PairCompose([
    PT.PairResize((256, 256)),
    PT.PairGrayscale(),
    PT.PairToTensor(),
])


def get_transforms(mode='train'):
    if mode=='train':
        tmft = train_tmft
    else:
        tmft = valid_tmft
        
    return tmft


def pair_data_loader(root_path, mode='train', 
                     batch_size=24, shuffle=True, num_workers=16):

    tmft = get_transforms(mode=mode)
    dset = PairDataset(root=root_path, pair_transform=tmft, mode=mode)
    dloader = data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, 
                              num_workers=num_workers)
    
    return dloader, dset
    