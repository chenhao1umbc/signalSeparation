#@title Adding all the packages
import os
import h5py 
import numpy as np
import scipy.io as sio
from scipy.signal import stft 
import itertools

import torch
from torch import nn
import torch.nn.functional as Func
import torch.utils.data as Data
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100

from unet.unet_model import UNet

"make the result reproducible"
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Opt():
    def __init__(self, ifgendata=False) -> None:
        self.ifgendata = ifgendata
        self.n_epochs = 101
        self.lr = 1e-3
        self.train_model = False


def label_gen(n):
    """This function generates labels in the mixture

    Parameters
    ----------
    n : [int]
        how many components in the mixture
    """
    lb_idx = np.array(list(itertools.combinations([0,1,2,3,4,5], 2)))
    label_n = np.zeros( (lb_idx.shape[0], 6) )
    for i in range(lb_idx.shape[0]):
        label_n[i, lb_idx[i]] = 1
    return label_n