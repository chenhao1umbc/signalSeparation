import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, loader, device):
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    print(f'Validation sample num:{n_val}')
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs = batch['mixture'].unsqueeze(1)
            true_masks = batch['source_labels'][:, 2, :, :].unsqueeze(1)
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            tot += F.mse_loss(mask_pred, true_masks).item()
            pbar.update()
    print(f'Validation loss:{tot/n_val}')
    return tot / n_val
