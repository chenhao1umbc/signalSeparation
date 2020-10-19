#%%
from utils import *


#%% EM algorithm
"data is x in [Channels, t], cj in [Channels, f, n]"
II = np.eye(6)

# conpute Wiener filter
Wj = vj * Rj @ (vjp*Rjp).sum(0)
# Estimate spacial source
cj = Wj @ x
# compute the posterior second-order raw moment
Rhcj = chj @ chn.conj().T + vj*(II - Wj) @ Rj
# update spacial covariance
Rj = 1/N * ( (1/vj) * Rhcj ).sum(-1)  # sum over N
# compute unsonstrained source spectrogram
zj = 1/I * np.trace(np.inv(Rj) @ Rhcj)
# update source spectrogram
for i in range(n_class):
    v[j] = nets[i](z[j])

#%% Wiener filtering
stft_sources, flatten_sources, cov_matrix = norbert.expectation_maximization(stft_sources, stft_mixture)