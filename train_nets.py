
from utils import *

opt = {}
opt['n_epochs'] = 101
opt['lr'] = 0.001

model = UNet(n_ch_in=1, n_ch_out=1).cuda()
# optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'], weight_decay=1e-5)
criterion = nn.MSELoss()
tr = torch.load('fhss1_tr.pt')  # x, l, y
va = torch.load('fhss1_va.pt')

loss_train = []
loss_cv = []

for epoch in range(opt['n_epochs']):
    
    model.train()
    iter = 0
    for (x, l, y) in tr: 
        out = model(x.cuda())
        optimizer.zero_grad()  

        loss = criterion(out, y.cuda())              
        loss.backward()
        optimizer.step()
        loss_train.append(loss.data.item())
        torch.cuda.empty_cache()
 
    model.eval()
    with torch.no_grad():
        cv_loss = 0
        for (xval, yval, mval) in va: 
            cv_cuda = xval.cuda()
            cv_yh = model(cv_cuda)[0].cpu()
            cv_loss = cv_loss + Func.mse_loss(cv_yh, yval)
            torch.cuda.empty_cache()
        loss_cv.append(cv_loss/5)
    
    if epoch%10 ==0:
        plt.figure()
        plt.plot(loss_train[-249:], '-x')
        plt.title('train loss per 200 iter')

        plt.figure()
        plt.plot(loss_cv, '--xr')
        plt.title('train loss per 200 iter')
        plt.show()

    torch.save(model.state_dict(), './eig_unet'+str(epoch)+'.pt')
    print('current epoch is ', epoch)

