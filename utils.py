#@title Adding all the packages
import os
import scipy.io as sio
from scipy.signal import stft 
# import h5py 
import numpy as np


import torch
from torch import nn
import torch.nn.functional as Func
import torch.utils.data as Data
import matplotlib.pyplot as plt

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
