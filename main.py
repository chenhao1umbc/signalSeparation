#%%
from utils import *

#%% load data
"""sources shape of [n_comb, n_sample, time_len]
    labels shape of [n_comb, n_class=6]"""
sources, l_s = get_mixdata_label(mix=1, pre='train_200_')
mix, l_mix = get_mixdata_label(mix=6, pre='train_200_')

f, t, Z = stft(mix[0,0], fs=4e7, nperseg=200, boundary=None)
plt.figure()
plt.imshow(np.log(abs(np.roll(Z, 100, axis=0))+1e-20), \
    vmax=-3, vmin=-11, aspect='auto', interpolation='None')
plt.title('One example of 6-component mixture')
plt.colorbar()

#%% EM with Wiener filtering

n = 0  # which example to test
s = awgn(sources[:, n], snr=0)  # shape of [n_sources, Time_len]
_, _, zs = stft(s, fs=4e7, nperseg=200, boundary=None)
temp = np.roll(zs, 100, axis=1)
n_sources = s.shape[0]
stft_sources = np.zeros( (200,200, 1, 6)).astype(np.complex)
for i in range(n_sources):
    stft_sources[...,0, i] = temp[i]

_, _, zm = stft(mix[:,n], fs=4e7, nperseg=200, boundary=None)
stft_mixture = np.roll(zm, 100, axis=1).reshape(200, 200, 1).astype(np.complex)

for i in range(n_sources):
    model = UNet(n_channels=1, n_classes=1).cuda()
    model.load_state_dict(torch.load('./models/'+i+'_unet4.pt'))  # l1+l2
    model.eval()

    va = torch.load('../data/data_ss/fhss1_va_200.pt')
    a = next(iter(va))
    with torch.no_grad():
        xte, yte, l = a
        te_cuda = xte.unsqueeze(1).cuda()
        te_yh = model(te_cuda).cpu().squeeze()
        torch.cuda.empty_cache()








stft_sources, flatten_sources, cov_matrix = \
    norbert.expectation_maximization(stft_sources, stft_mixture, iterations=10)

var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
for i in range(6):
    plt.figure()
    plt.imshow(np.log(abs(stft_sources[..., i])+1e-20), vmax=-3, vmin=-11)
    plt.colorbar()
    plt.title(var_name[i])



