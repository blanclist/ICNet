from .dataset import get_loader
from .smeasure import calc_smeasure
from .fmeasure import calc_p_r_fmeasure
from .mae import calc_mae
import numpy as np
import torch
from decimal import Decimal

def tf(data):
    return float(data)

def tn(data):
    return np.array(data.cpu())

def td(data):
    return Decimal(data).quantize(Decimal('0.000'))

def get_n(gt, pred, n_mask):
    H, W = gt.shape
    HW = H * W
    n_gt = gt.view(1, HW).repeat(255, 1)  # [255, HW]
    n_pred = pred.view(1, HW).repeat(255, 1)  # [255, HW]
    n_pred = torch.where(n_pred <= n_mask, torch.zeros_like(n_pred), torch.ones_like(n_pred))
    return n_gt, n_pred

def evaluate_dataset(roots, dataset, batch_size, num_thread, demical, suffixes, pin):
    with torch.no_grad():
        dataloader = get_loader(roots, suffixes, batch_size, num_thread, pin=pin)
        p = np.zeros(255)
        r = np.zeros(255)
        s = 0.0
        f = np.zeros(255)
        mae = 0.0
        n_mask = torch.FloatTensor(np.array(range(255)) / 255.0).view(255, 1).repeat(1, 224 * 224).cuda()  # [255, HW]
        for batch in dataloader:
            gt, pred = batch['gt'].cuda().view(224, 224), batch['pred'].cuda().view(224, 224)

            _s = calc_smeasure(gt, pred)
            _mae = calc_mae(gt, pred)
            n_gt, n_pred = get_n(gt, pred, n_mask)
            _p, _r, _f = calc_p_r_fmeasure(n_gt, n_pred, n_mask)
            _mean_f = torch.mean(_f)
            _max_f = torch.max(_f)

            _s = tf(_s)
            _p = tn(_p)
            _r = tn(_r)
            _f = tn(_f)
            _mae = tf(_mae)
            _mean_f = tf(_mean_f)
            _max_f = tf(_max_f)

            p += _p
            r += _r
            s += _s
            f += _f
            mae += _mae
        num = len(dataloader)
        p /= num
        r /= num
        f /= num
        s, mae, mean_f, max_f = s / num, mae / num, np.mean(f), np.max(f)
        if demical == True:
            s, mae, mean_f, max_f = td(s), td(mae), td(mean_f), td(max_f)
        
        results = {'s': s, 'p': p, 'r': r, 'f': f, 
                   'mae': mae, 
                   'mean_f': mean_f, 'max_f': max_f}
    return results