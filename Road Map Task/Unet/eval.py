import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from dice_loss import dice_coeff
from roadmap import images_prep, masks_prep, BEV


def eval_net(net, loader, device, bev):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for sample, _, road_image in loader:
            # if bev:
            #     imgs = BEV(sample)
            # else:
            #     imgs = images_prep(sample)
            samples = torch.stack(sample)
            imgs = F.interpolate(samples, size=(3,800,800))
            true_masks = masks_prep(road_image)

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
                # print(mask_pred.shape)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    return tot / n_val
