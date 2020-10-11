#%%
from utils import *

# stft will be done on the last dimension
#%% data processing
"raw data processing"
var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
data = {}
for i in range(6):
    a = sio.loadmat( '/home/chenhao1/Matlab/LMdata/'+var_name[i]+'_2000.mat')
    dd = (np.sum((abs(a['x'])**2), 1)**0.5).reshape(2000, 1)
    data[i] = a['x'] / dd  # normalized very data to 1
    # d = torch.tensor(a['x']).to(torch.cfloat)  # torch complex64

"shuffle and split data to train_val and test"
np.random.seed(0)
for i in range(6): # data[i].shape is (2000, 32896)
    np.random.shuffle(data[i])

train_val = np.zeros((6, 1600, data[0].shape[1]))
test = np.zeros((6, 400, data[0].shape[1]))
for i in range(6): # split 1600 for tain_val, 400 for test
    train_val[i] = data[i][:1600]
    test = data[i][-400:]
"make train_val datasets with labels"

np.abs()
"make test dataset with labels"




#%% generate train, val, test data
# import itertools
# all_combos = list(itertools.combinations([0,1,2,3,4,5], 2))




# #%% algorithm
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