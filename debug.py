#%%
import torch
import pickle
file = open('/home/chenhao1/zeyu/datasets/dataset_0426_14000_128x20/train_set.pickle','rb')
d = pickle.load(file)
d0 = d.dataset.dataset.mixture_list[0]['mixture']
l0 = d.dataset.dataset.mixture_list[0]['label']

# %%
import matplotlib.pyplot as plt


# %%
