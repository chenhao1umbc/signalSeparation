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
    lb_idx = np.array(list(itertools.combinations([0,1,2,3,4,5], n)))
    label_n = np.zeros( (lb_idx.shape[0], 6) )
    for i in range(lb_idx.shape[0]):
        label_n[i, lb_idx[i]] = 1
    return label_n


def mix_data_torch(x, labels):
    """This functin will mix the data according the to labels

    Parameters
    ----------
    x : [tensor of complex]
        [data with shape of [n_classes, n_samples, time_len]]
    labels : [matrix of int]
        [maxtrix of [n_samples, n_classes]]

    Returns
    -------
    [complex pytorch]
        [mixture data with shape of [n_samples, time_len] ]
    """
    n = labels.shape[0]
    output = np.zeros( (n, x.shape[1], x.shape[2]) ).astype('complex64')
    for i1 in range(n):
        s = 0
        for i2 in range(6):
            if labels[i1, i2] == 1:
                s = s + x[i2]
            else:
                pass
        output[i1] = s
    return torch.tensor(output), torch.tensor(labels).to(torch.float)


def save_mix(x, lb1, lb2, lb3, lb4, lb5, lb6, pre='_'):
    mix_1, label1 = mix_data_torch(x, lb1)  # output is in pytorch tensor
    mix_2, label2 = mix_data_torch(x, lb2)
    mix_3, label3 = mix_data_torch(x, lb3)
    mix_4, label4 = mix_data_torch(x, lb4)
    mix_5, label5 = mix_data_torch(x, lb5)
    mix_6, label6 = mix_data_torch(x, lb6)

    torch.save({'data':mix_1, 'label':label1}, pre+'dict_mix_1.pt')
    torch.save({'data':mix_2, 'label':label2}, pre+'dict_mix_2.pt')
    torch.save({'data':mix_3, 'label':label3}, pre+'dict_mix_3.pt')
    torch.save({'data':mix_4, 'label':label4}, pre+'dict_mix_4.pt')
    torch.save({'data':mix_5, 'label':label5}, pre+'dict_mix_5.pt')
    torch.save({'data':mix_6, 'label':label6}, pre+'dict_mix_6.pt')


def get_label(lb, shape):
    """repeat the labels for the shape of mixture data

    Parameters
    ----------
    lb : [torch.float matrix]
        [matrix of labels]
    shape : [tuple int]
        [data shape]]

    Returns
    -------
    [labels]
        [large matrix]
    """
    n_comb, n_sample = shape
    label = np.repeat(lb, n_sample, axis=0).reshape(n_comb, n_sample, 6 )
    return label


def get_mixdata_label(mix=1):
    """loading mixture data and prepare labels

    Parameters
    ----------
    mix : int, optional
        [how many components in the mixture], by default 1

    Returns
    -------
    [data, label]
    """
    dict = torch.load('../data_ss/train_dict_mix_'+str(mix)+'.pt')
    label = get_label(dict['label'], dict['data'].shape[:2])
    return dict['data'], label
