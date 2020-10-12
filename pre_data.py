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

#%% save mixture data
save_mix(train_val, label1, label2, label3, label4, label5, label6, pre='train_')
save_mix(test, label1, label2, label3, label4, label5, label6, pre='test_')

print('done')




#%% ___________________assuming mixture data is done___________________

dict = torch.load('../data/train_dict_mix_6.pt')
f, t, Z = stft(dict['data'][0,0], fs=4e7, nperseg=256, boundary=None)
plt.figure()
plt.imshow(abs(np.roll(Z, 128, axis=0)), aspect='auto', interpolation='None')

#%% 
"""dict have keys ['data'] shape of [n_comb, n_sample, time_len]
    ['label'] shape of [n_comb, n_class=6]
"""
d, l = get_mixdata_label(mix=1)
d1 = d.clone()
for i in range(2,7):
    dt, lt = get_mixdata_label(mix=i)  # temp
    d, l = torch.cat( (d, dt)), torch.cat( (l , lt))
xtr, ltr, ytr = d[:, :1200], l[:, :1200], d1[:, :1200]
xva, lva, yva = d[:, 1200:], l[:, 1200:], d1[:, 1200:]

"train data for ble" # "training data is the log(abs(stft(x)))"
n_sample, t_len = xtr.shape[1:]
xtr = xtr.reshape(-1, t_len)
ltr = ltr.reshape(-1, 6)

ind = ltr[:, 0]==1.0  # find the index, which belonged to this class
ble_ltr = ltr[ind]  # find labels

"get the stft with low freq. in the center"
f, t, Z = stft(xtr[ind], fs=4e7, nperseg=256, boundary=None)
ble_xtr = torch.tensor(np.log(abs(np.roll(Z, 128, axis=1))))

"get the cleaned source as the ground-truth"
f, t, Z = stft(ytr[0], fs=4e7, nperseg=256, boundary=None)
temp = torch.tensor(np.log(abs(np.roll(Z, 128, axis=1))))
n_tile = int(ble_xtr.shape[0]/n_sample)
ble_ytr = torch.tensor(np.tile(temp, (n_tile, 1,1)))

tr = Data.TensorDataset(ble_xtr, ble_ytr, ble_ltr)
tr = Data.DataLoader(tr, batch_size=30, shuffle=True)

#%% algorithm
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