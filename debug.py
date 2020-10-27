# %%  visualize raw data
from utils import *

#%% EM + NeuralNet
"""sources shape of [n_comb, n_sample, time_len]
    labels shape of [n_comb, n_class=6]"""
sources, l_s = get_mixdata_label(mix=1, pre='train_200_')
mix, l_mix = get_mixdata_label(mix=6, pre='train_200_')

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

    with torch.no_grad():
        te_cuda = xte.unsqueeze(1).cuda()
        te_yh = model(te_cuda).cpu().squeeze()
        torch.cuda.empty_cache()
