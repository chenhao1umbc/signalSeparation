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

#%%
"Initial estimate"
# s = awgn(sources[:, n], snr=0)  # shape of [n_sources, Time_len]
# _, _, zs = stft(s, fs=4e7, nperseg=200, boundary=None)
# s_stft = np.roll(zs, 100, axis=1)

s_stft = torch.zeros(6, 200, 200)
fname = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
xte = torch.tensor(stft_mixture).reshape(1,1,200,200).abs().log().float()
te_cuda = xte.cuda()
for i in range(6):
    model = UNet(n_channels=1, n_classes=1).cuda()
    model.load_state_dict(torch.load('../data/data_ss/'+fname[i]+'_unet20.pt'))
    model.eval()

    with torch.no_grad():
        s_stft[i] = model(te_cuda).cpu().squeeze()
        torch.cuda.empty_cache()

#%%
# EM from Norbert for only 1 Channel, 1 sample
n_iter = 10 # how many iterations for EM
n_s, n_f, n_t = s_stft.shape # number of sources, freq. bins, time bins
n_c = 1 # number of channels
cjh = s_stft.clone().to(torch.complex64).exp()
x = torch.tensor(stft_mixture).squeeze()
eps = 1e-20
# Rj = torch.ones(n_s, n_f).to(torch.complex64)

mse = []
for i in range(n_iter):
    vj = cjh.abs()**2  #shape of [n_s, n_f, n_t], mean of all channels
    Rcj = cjh * cjh.conj()  # shape of [n_s, n_f, n_t]
    Rj =  (Rcj/(vj+eps)).sum(2)/n_t  # shape of [n_s, n_f]
    
    "Compute mixture covariance"
    Rx = (vj * Rj[..., None]).sum(0)  #shape of [n_f, n_t]
    "Calc. Wiener Filter"
    Wj = vj*Rj[..., None] / (Rx +eps) # shape of [n_s, n_f, n_t]
    "get STFT estimation"
    cjh = Wj * x

    s = 0
    for ii in range(6):
        s = s + ((cjh[ii] - gt_stft[ii]).abs()**2).sum()**0.5
    mse.append(s.item())

plt.plot(mse)

# %%
"Visualize the output"
var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
for i in range(6):
    plt.figure()
    plt.imshow(np.log(abs(s_hat[..., i])+1e-20), vmax=-3, vmin=-11)
    plt.colorbar()

    plt.figure()
    plt.imshow(np.log(abs(cjh[i].numpy())+1e-20), vmax=-3, vmin=-11)
    plt.colorbar()
   





# %%
