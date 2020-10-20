#%%
from utils import *

#%% load data
"""sources shape of [n_comb, n_sample, time_len]
    labels shape of [n_comb, n_class=6]"""
sources, l_s = get_mixdata_label(mix=1, pre='train_200_')
mix, l_mix = get_mixdata_label(mix=6, pre='train_200_')

f, t, Z = stft(mix[0,0], fs=4e7, nperseg=200, boundary=None)
plt.figure()
plt.imshow(abs(np.roll(Z, 100, axis=0)), aspect='auto', interpolation='None')
plt.title('One example of 6-component mixture')

#%% EM with Wiener filtering
n = 0  # which example to test
s = awgn(sources[:, n], snr=-10)  # shape of [n_sources, Time_len]
_, _, zs = stft(s, fs=4e7, nperseg=200, boundary=None)
temp = np.roll(zs, 100, axis=1)
n_sources = s.shape[0]
stft_sources = np.zeros( (200,200, 1, 6)).astype(np.complex)
for i in range(n_sources):
    stft_sources[...,0, i] = temp[i]

_, _, zm = stft(mix[:,n], fs=4e7, nperseg=200, boundary=None)
stft_mixture = np.roll(zm, 100, axis=1).reshape(200, 200, 1).astype(np.complex)

stft_sources, flatten_sources, cov_matrix = norbert.expectation_maximization(stft_sources, stft_mixture)
    
for i in range(6):
    plt.figure()
    plt.imshow(abs(stft_sources[..., i]))

#%% EM algorithm
# "data is x in [Channels, t], cj in [Channels, f, n]"
# II = np.eye(6)

# # conpute Wiener filter
# Wj = vj * Rj @ (vjp*Rjp).sum(0)
# # Estimate spacial source
# cj = Wj @ x
# # compute the posterior second-order raw moment
# Rhcj = chj @ chn.conj().T + vj*(II - Wj) @ Rj
# # update spacial covariance
# Rj = 1/N * ( (1/vj) * Rhcj ).sum(-1)  # sum over N
# # compute unsonstrained source spectrogram
# zj = 1/I * np.trace(np.inv(Rj) @ Rhcj)
# # update source spectrogram
# for i in range(n_class):
#     v[j] = nets[i](z[j])

