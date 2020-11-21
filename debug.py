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

# %% Test data with power difference without EM
"""sources shape of [n_comb, n_sample, time_len]
    labels shape of [n_comb, n_class=6]"""
n_sources = 6
sources, l_s = get_mixdata_label(mix=1, pre='train_200_')
mix, l_mix = get_mixdata_label(mix=n_sources, pre='train_200_')

"Mixture for the EM"
n_comb, n = 0, 0  # which example to test
# x = mix[n_comb,n]
db = 10  # db in [0, 20]
power_ratio = 10**(-1*db/20)
x = sources[1, n]*power_ratio + sources[2, n] + sources[3, n]

_, _, zm = stft(x, fs=4e7, nperseg=200, boundary=None)
stft_mixture = np.roll(zm, 100, axis=0).reshape(200, 200, 1).astype(np.complex)
plt.figure()
plt.imshow(np.log(abs(np.roll(zm, 100, axis=0))+1e-20), \
    vmax=-3, vmin=-11, aspect='auto', interpolation='None')
plt.title('Input mixture')
plt.colorbar()

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

for i in range(6):
    plt.figure()
    plt.imshow(s_stft[i], vmax=-3, vmin=-11, aspect='auto', interpolation='None')
    plt.title(fname[i])
    plt.colorbar()


#%%

# "EM to get each sources"
# gt_stft = torch.rand(6, 200, 200, dtype=torch.complex64)
# for i in range(6):
#     _, _, zm = stft(sources[i, n], fs=4e7, nperseg=200, boundary=None)
#     gt_stft[i] = torch.tensor(np.roll(zm, 100, axis=0))

# n_iter = 30
# mse = []
# init = s_stft #  gt_stft.abs().log()
# for i in range(1, n_iter):
#     cjh = em_simple(init_stft=init, stft_mix=stft_mixture, n_iter=i)  # instead of import Norbert
#     mse.append((((cjh - gt_stft).abs()**2).sum()**0.5).item())

# plt.plot(mse, '-x')


"EM to get each sources"
gt_stft = torch.rand(6, 200, 200, dtype=torch.complex64)
for i in range(6):
    _, _, zm = stft(sources[i, n], fs=4e7, nperseg=200, boundary=None)
    gt_stft[i] = torch.tensor(np.roll(zm, 100, axis=0))

n_iter = 30
mse = []
init = s_stft #  gt_stft.abs().log()
for i in range(1, n_iter):
    cjh = em_10paper(init_stft=init, stft_mix=stft_mixture, n_iter=i)  # instead of import Norbert
    mse.append((((cjh - gt_stft).abs()**2).sum()**0.5).item())
plt.figure()
plt.plot(mse, '-x')


# %%
"Visualize the output"
var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
for i in range(6):
    plt.figure()
    plt.imshow(np.log(abs(cjh[i])+1e-20), vmax=-3, vmin=-11)
    plt.colorbar()
    plt.title(var_name[i])


# %%
gt = sources[:, n]  # ground truth
_, ss = istft(np.roll(cjh[0], 100, axis=0), fs=4e7, nperseg=200, boundary=None)
_, sss = istft(np.roll(gt_stft[0], 100, axis=0), fs=4e7, nperseg=200, boundary=None)

# %%
a = np.random.rand(20100) +1j*np.random.rand(20100)
_, _, A = stft(a, fs=4e7, nperseg=200, boundary=None)
_, aa = istft(A, fs=4e7, nperseg=200, input_onesided=False, boundary=None )
plt.plot(aa.imag[:500]+1)
plt.plot(a.imag[:500])