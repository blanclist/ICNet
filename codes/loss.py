import torch

"""
IoU_loss:
    Compute IoU between predictions and ground-truths as training loss [Equation 3].
"""
def IoU_loss(preds_list, gt):
    preds = torch.cat(preds_list, dim=1)
    N, C, H, W = preds.shape
    min_tensor = torch.where(preds < gt, preds, gt) # [N, C, H, W]
    max_tensor = torch.where(preds > gt, preds, gt) # [N, C, H, W]
    min_sum = min_tensor.view(N, C, H * W).sum(dim=2)  # [N, C]
    max_sum = max_tensor.view(N, C, H * W).sum(dim=2)  # [N, C]
    loss = 1 - (min_sum / max_sum).mean()
    return loss