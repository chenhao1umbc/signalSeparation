#%%
from utils import *
import pickle
file = open('/home/chenhao1/zeyu/datasets/dataset_0426_14000_128x20//component/Blt.mat.pickle','rb')
d = pickle.load(file)

plt.imshow(d[0].T, interpolation='None', aspect='auto')
plt.colorbar()


# %%
from scipy.signal import stft
from scipy.signal import istft
fs = 10e3
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
mod = 500*np.cos(2*np.pi*0.25*time)
carrier = amp * np.sin(2*np.pi*3e3*time + mod)
noise = np.random.normal(scale=np.sqrt(noise_power),
                         size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise

f, t, Zxx = stft(x, fs, nperseg=500)
xh = istft(Zxx, nperseg=500)
print((xh[1] - x).sum())
print('done')


# %%
