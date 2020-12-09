import string
from argparse import Namespace
from typing import *

import numpy as np
import torch
import torch.nn.functional as F

from ..models.craft import CRAFT
from ..models.utils import copy_state_dict



def load_weights(weights_path: str, key=None) -> str:
    weights = torch.load(weights_path, map_location=torch.device('cpu'))
    if key is not None:
        weights = weights[key]
    weights = copy_state_dict(weights['state_dict'])
    return weights


def load_craft_network(weight_path: str, key="state_dict", device='cpu'):
    model = CRAFT(pretrained=True)
    weight = load_weights(weight_path, key=key)
    model.load_state_dict(weight)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.eval()
    return model


def load_refine_net(weight_path, device='cpu'):
    refine_net = RefineNet()
    if not device == 'cpu':
        refine_net.load_state_dict(copy_state_dict(torch.load(weight_path)))
        refine_net = refine_net.to(device)
        refine_net = torch.nn.DataParallel(refine_net)
    else:
        refine_net.load_state_dict(copy_state_dict(torch.load(weight_path, map_location='cpu')))

    refine_net.eval()
    return refine_net