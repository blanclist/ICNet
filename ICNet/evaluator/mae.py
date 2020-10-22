import torch

def calc_mae(gt, pred):
    return torch.mean(torch.abs(gt - pred))