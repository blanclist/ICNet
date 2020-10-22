import torch

def calc_p_r_fmeasure(n_gt, n_pred, n_mask):
    tp = torch.sum(n_pred * n_gt, dim=1)  # [255]
    tp_plus_fp = torch.sum(n_pred, dim=1)  # [255]
    temp = torch.ones_like(tp_plus_fp)
    tp_plus_fp = torch.where(tp_plus_fp == 0.0, temp, tp_plus_fp)
    tp_plus_fn = torch.sum(n_gt, dim=1)  # [255]
    tp_plus_fn = torch.where(tp_plus_fn == 0.0, temp, tp_plus_fn)
    precision = tp / tp_plus_fp
    recall = tp / tp_plus_fn
    a = 1.3 * precision * recall
    b = 0.3 * precision + recall
    temp = torch.ones_like(b) * 1e31
    b = torch.where(b == 0.0, temp, b)
    fBetaScore = a / b
    return precision, recall, fBetaScore