#%%
from utils import *

#%% the body of the EM structure

def neural_EM(init_stft, stft_mix, n_iter):
    """This function is implemented using 2010's paper, for multiple channels with pytorch

        Parameters
        ----------
        init_stft : [real tensor]
            [shape of [n_source, f, t]]

        stft_mix : [complex tensor]
            [shape of [f, t, n_channel]]
            
        n_iter : [int]
            [how many iterations for EM]

        Returns
        -------
        [complex tensor]
            [shape of [n_s, n_f, n_t, n_c]]
    """
    n_s = init_stft.shape[0]
    n_f, n_t, n_c =  stft_mix.shape 
    I =  torch.ones(n_s, n_f, n_t, n_c).diag_embed().to(torch.complex64)
    eps = 1e-20  # no smaller than 1e-22
    x = torch.tensor(stft_mix).unsqueeze(-1)  #shape of [n_s, n_f, n_t, n_c, 1]
    "Initialize spatial covariance matrix"
    Rj =  torch.ones(n_s, n_f, 1, n_c).diag_embed().to(torch.complex64) 
    vj = init_stft.clone().to(torch.complex64).exp()
    cjh = vj.clone().unsqueeze(-1)  # for n_ter == 0
    cjh_list = []
    for i in range(n_c-1):
        cjh = torch.cat((cjh, vj.unsqueeze(-1)), dim=-1)
    cjh_list.append(cjh.squeeze())
    likelihood = torch.zeros(n_iter).to(torch.complex64)

    #define Unet
    opt = {}
    opt['n_epochs'] = 25
    opt['lr'] = 0.001
    model = UNet(n_channels=1, n_classes=1).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.RAdam(
                    model.parameters(),
                    lr= opt['lr'],
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0)
                
    for i in range(n_iter):
        # the E-step
        Rcj = (vj * Rj.permute(3,4,0,1,2)).permute(2,3,4,0,1) # shape as Rcjh
        "Compute mixture covariance"
        Rx = Rcj.sum(0)  #shape of [n_f, n_t, n_c, n_c]
        "Calc. Wiener Filter"
        Wj = Rcj @ torch.tensor(np.linalg.inv(Rx)) # shape of [n_s, n_f, n_t, n_c, n_c]
        "get STFT estimation, the conditional mean"
        cjh = Wj @ x  # shape of [n_s, n_f, n_t, n_c, 1]
        "get covariance"
        Rcjh = cjh@cjh.permute(0,1,2,4,3).conj() + (I -  Wj) @ Rcj

        #record the result 
        cjh_list.append(cjh.squeeze())
        likelihood[i] = calc_likelihood(torch.tensor(stft_mix), Rx)

        # the M-step
        "Get spectrogram- power spectram"  #shape of [n_s, n_f, n_t]
        gammaj = (torch.tensor(np.linalg.inv(Rj))\
             @ Rcjh).diagonal(dim1=-2, dim2=-1).sum(-1)/n_c
        "cal spatial covariance matrix"
        Rj = ((Rcjh/(vj+eps)[...,None, None]).sum(2)/n_t).unsqueeze(2)

        # the neural network step
        for epoch in range(opt['n_epochs']):    
            model.train()
            for i, (x, y, l) in enumerate(tr): 
                out = model(x.unsqueeze(1).cuda())
                optimizer.zero_grad()  

                loss = criterion(out.squeeze(), y.cuda())              
                loss.backward()
                optimizer.step()
                loss_train.append(loss.data.item())
                torch.cuda.empty_cache()
                if i%50 == 0: print(f'Current iter is {i} in epoch {epoch}')
        
            model.eval()
            with torch.no_grad():
                cv_loss = 0
                for xval, yval, lval in va: 
                    cv_cuda = xval.unsqueeze(1).cuda()
                    cv_yh = model(cv_cuda).cpu().squeeze()
                    cv_loss = cv_loss + Func.mse_loss(cv_yh, yval)
                    torch.cuda.empty_cache()
                loss_cv.append(cv_loss/106)  # averaged over all the iterations
            
            if epoch%1 ==0:
                plt.figure()
                plt.plot(loss_train[-1400::50], '-x')
                plt.title('train loss per 50 iter in last 1400 iterations')

                plt.figure()
                plt.plot(loss_cv, '--xr')
                plt.title('val loss per epoch')
                plt.show()
            
            torch.save(model.state_dict(), './f1_unet'+str(epoch)+'.pt')
            print('current epoch is ', epoch)


    return cjh_list, likelihood
# %% Test data with power difference without EM
"""sources shape of [n_comb, n_sample, time_len]
    labels shape of [n_comb, n_class=6]"""
n_sources = 6
n_comb, n = 0, 0  # which example to test
sources, l_s = get_mixdata_label(mix=1, pre='train_200_')
mix, l_mix = get_mixdata_label(mix=n_sources, pre='train_200_')

"Mixture for the EM"
db = 10  # db in [0, 20]
power_ratio = 10**(-1*db/20)
# x = sources[1, n]*power_ratio + sources[2, n] + sources[3, n]
x = mix[n_comb,n]  # mixture of 6 components without power diff.
plot_x(x, title='Input mixture')  # plot input

s_stft = torch.zeros(6, 200, 200)
fname = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
xte = st_ft(x).reshape(1,1,200,200).abs().log().float()
te_cuda = xte.cuda()
for i in range(6):
    model = UNet(n_channels=1, n_classes=1).cuda()
    model.load_state_dict(torch.load('../data/data_ss/'+fname[i]+'_unet20.pt'))
    model.eval()

    with torch.no_grad():
        s_stft[i] = model(te_cuda).cpu().squeeze()
        torch.cuda.empty_cache()
 
#%% Multi-Channel
"EM to get each sources"
n_iter = 100
n_c = 2  # 2 channels
mse = []
var_name = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']

which_source = torch.tensor([3,4])
x = sources[which_source, n].sum(0)
gt_stft = torch.rand(which_source.shape[0],200, 200, n_c, dtype=torch.complex64)
for i in range(which_source.shape[0]):
    s = sources[which_source[i], n]
    gt_stft[i, ... , 0] = st_ft(s)
    gt_stft[i, ... , 1] = st_ft(s*e**(1j*pi/12*(i+1))) # awgn(st_ft(s), snr=20)

init = awgn(s_stft[which_source], snr=10) #  gt_stft.abs().log()
# init = torch.rand(2, 200, 200) -9.8
cjh_list, likelihood = em10(init_stft=init, stft_mix=gt_stft.sum(0), n_iter=n_iter) 

for i in [0,1,30,50,90]:
    for ii in range(which_source.shape[0]):
        plot_x(cjh_list[i][ii,...,0], title=f'{var_name[which_source[ii]]} iter {i}')

for i in range(n_iter+1):
    mse.append((((cjh_list[i] - gt_stft).abs()**2).sum()).item())   
plt.figure()
plt.plot(mse, '-x')
plt.figure()
plt.plot(likelihood, '-x')

# # %% Norbert Multi-Channel
    # import norbert
    # mse = []
    # x = gt_stft.sum(0).permute(1,0,2).numpy()
    # y = torch.stack((init.permute(2,1,0), init.permute(2,1,0)), -1 ).numpy()
    # init = awgn(s_stft[which_source], snr=20)
    # for ii in range(20):
    #     yh, vh, rh = norbert.expectation_maximization(
    #         y=y, x=x, iterations=ii) 
    #     mse.append((((torch.tensor(yh) - gt_stft.permute(2,1, 3, 0)).abs()**2).sum()).item())
    # plt.figure()
    # plt.plot(mse, '-x')

# %%  how to do inverse STFT
a = np.random.rand(20100) +1j*np.random.rand(20100)
_, _, A = stft(a, fs=4e7, nperseg=200, boundary=None)
_, aa = istft(A, fs=4e7, nperseg=200, input_onesided=False, boundary=None )
plt.plot(aa.imag[:500]+1)
plt.plot(a.imag[:500])
# %%
