#%%
from utils import *

#%% load data
"""sources shape of [n_comb, n_sample, time_len]
    labels shape of [n_comb, n_class=6]"""
n_sources = 6
sources, l_s = get_mixdata_label(mix=1, pre='train_200_')
mix, l_mix = get_mixdata_label(mix=n_sources, pre='train_200_')

"Mixture for the EM"
n_comb, n = 0, 0  # which example to test, first sample of 6 components
plot_x(st_ft(mix[n_comb,n]), title='One example of 6-component mixture')  # plot input

# %% Test data with power difference without EM
"""sources shape of [n_comb, n_sample, time_len]
    labels shape of [n_comb, n_class=6]"""
n_sources = 6
sources, l_s = get_mixdata_label(mix=1, pre='train_200_')
mix, l_mix = get_mixdata_label(mix=n_sources, pre='train_200_')

"Mixture for the EM"
n_comb, n = 0, 0  # which example to test
# x = mix[n_comb,n]
db = 20  # db in [0, 20]
power_ratio = 10**(-1*db/20)
x = sources[1, n]*power_ratio + sources[2, n] + sources[3, n]
plot_x(x, title='Input mixture')  # plot input

s_stft = torch.zeros(6, 200, 200)
fname = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
xte = torch.tensor(st_ft(x)).reshape(1,1,200,200).abs().log().float()
te_cuda = xte.cuda()
for i in range(6):
    model = UNet(n_channels=1, n_classes=1).cuda()
    model.load_state_dict(torch.load('../data/data_ss/'+fname[i]+'_unet20.pt'))
    model.eval()

    with torch.no_grad():
        s_stft[i] = model(te_cuda).cpu().squeeze()
        torch.cuda.empty_cache()

"plot results"
for i in range(6):
    plot_log_stft(s_stft[i], title=fname[i])

#%%
"EM to get each sources"
x = sources[1, n] + sources[2, n]
gt_stft = torch.rand(2, 200, 200, dtype=torch.complex64)
for i in [1, 2]:
    gt_stft[i-1] = torch.tensor(st_ft(sources[i, n]))

n_iter = 30
mse = []
init = s_stft[1:3] #  gt_stft.abs().log()
for i in range(1, n_iter):
    cjh = em_simple(init_stft=init, stft_mix=st_ft(x), n_iter=i)  # instead of import Norbert
    mse.append((((cjh - gt_stft).abs()**2).sum()**0.5).item())
plt.figure()
plt.plot(mse, '-x')


# %%
"Visualize the output"
var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
for i in [0, 1]:
    plot_x(cjh[i], title=var_name[i+1])

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