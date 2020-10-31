#%%
from utils import *

#%% load data
"""sources shape of [n_comb, n_sample, time_len]
    labels shape of [n_comb, n_class=6]"""
n_sources = 6
sources, l_s = get_mixdata_label(mix=1, pre='train_200_')
mix, l_mix = get_mixdata_label(mix=n_sources, pre='train_200_')

"Mixture for the EM"
n_comb, n = 0, 0  # which example to test
_, _, zm = stft(mix[n_comb,n], fs=4e7, nperseg=200, boundary=None)
stft_mixture = np.roll(zm, 100, axis=0).reshape(200, 200, 1).astype(np.complex)
plt.figure()
plt.imshow(np.log(abs(np.roll(zm, 100, axis=0))+1e-20), \
    vmax=-3, vmin=-11, aspect='auto', interpolation='None')
plt.title('One example of 6-component mixture')
plt.colorbar()
   

# %%
