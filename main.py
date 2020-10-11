#%%
from utils import *

# stft will be done on the last dimension
#%% data processing
"raw data processing"
var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
data = {}
for i in range(6):
    temp = sio.loadmat( '/home/chenhao1/Matlab/LMdata/'+var_name[i]+'_256_2k.mat')
    dd = (np.sum((abs(temp['x'])**2), 1)**0.5).reshape(2000, 1)
    data[i] = temp['x'] / dd  # normalized very data to 1
    # d = torch.tensor(a['x']).to(torch.cfloat)  # torch complex64

"shuffle and split data to train_val and test"
np.random.seed(0)
for i in range(6): # data[i].shape is (2000, 32896)
    np.random.shuffle(data[i])

train_val = np.zeros((6, 1600, data[0].shape[1])).astype('complex64')
test = np.zeros((6, 400, data[0].shape[1])).astype('complex64')
for i in range(6): # split 1600 for tain_val, 400 for test
    train_val[i] = data[i][:1600]
    test[i] = data[i][-400:]

#%% generate labels
idx = np.arange(6)
label1 = np.zeros((idx.size, idx.max()+1))
label1[np.arange(idx.size),idx] = 1  # one hot encoding

label2, label3 = label_gen(2), label_gen(3)
label4, label5 = label_gen(4), label_gen(5)
label6 = np.ones((1,6))
labels = np.concatenate( (label1, label2, label3, label4, label5, label6), axis=0)

#%% save mixture data
save_mix(train_val, label1, label2, label3, label4, label5, label6, pre='train_')
save_mix(test, label1, label2, label3, label4, label5, label6, pre='test_')

print('done')




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
# %%
