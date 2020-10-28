#%%
from utils import *

#%% load data
"""sources shape of [n_comb, n_sample, time_len]
    labels shape of [n_comb, n_class=6]"""
n_sources = 6
sources, l_s = get_mixdata_label(mix=1, pre='train_200_')
mix, l_mix = get_mixdata_label(mix=n_sources, pre='train_200_')

f, t, Z = stft(mix[0,0], fs=4e7, nperseg=200, boundary=None)
plt.figure()
plt.imshow(np.log(abs(np.roll(Z, 100, axis=0))+1e-20), \
    vmax=-3, vmin=-11, aspect='auto', interpolation='None')
plt.title('One example of 6-component mixture')
plt.colorbar()

#%% EM with Wiener filtering
"Mixture for the EM"
n_comb, n = 0, 0  # which example to test
_, _, zm = stft(mix[n_comb,n], fs=4e7, nperseg=200, boundary=None)
stft_mixture = np.roll(zm, 100, axis=0).reshape(200, 200, 1).astype(np.complex)

#%%
"Initial estimate"
# s = awgn(sources[:, n], snr=0)  # shape of [n_sources, Time_len]
# _, _, zs = stft(s, fs=4e7, nperseg=200, boundary=None)
# s_stft = np.roll(zs, 100, axis=1)

s_stft = torch.rand(6, 200, 200)
fname = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
xte = torch.tensor(stft_mixture).reshape(1,1,200,200).abs().log().float()
te_cuda = xte.cuda()
for i in range(n_sources):
    model = UNet(n_channels=1, n_classes=1).cuda()
    model.load_state_dict(torch.load('../data/data_ss/'+fname[i]+'_unet20.pt'))
    model.eval()

    with torch.no_grad():
        s_stft[i] = model(te_cuda).cpu().squeeze()
        torch.cuda.empty_cache()

#%%
"EM to get each sources"
stft_sources = np.zeros( (200,200, 1, 6)).astype(np.complex)
for i in range(n_sources):  # change index order to fit EM
    stft_sources[...,0, i] = s_stft[i]

stft_sources, flatten_sources, cov_matrix = \
    norbert.expectation_maximization(stft_sources, stft_mixture, iterations=10)

"Visualize the output"
var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
for i in range(6):
    plt.figure()
    plt.imshow(np.log(abs(stft_sources[..., i])+1e-20), vmax=-3, vmin=-11)
    plt.colorbar()
    plt.title(var_name[i])



