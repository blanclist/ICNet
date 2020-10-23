import torch

"""
IoU_loss:
    Compute IoU loss between predictions and ground-truths for training [Equation 3].
"""
def IoU_loss(preds_list, gt):
    preds = torch.cat(preds_list, dim=1)
    N, C, H, W = preds.shape
    min_tensor = torch.where(preds < gt, preds, gt)    # shape=[N, C, H, W]
    max_tensor = torch.where(preds > gt, preds, gt)    # shape=[N, C, H, W]
    min_sum = min_tensor.view(N, C, H * W).sum(dim=2)  # shape=[N, C]
    max_sum = max_tensor.view(N, C, H * W).sum(dim=2)  # shape=[N, C]
    loss = 1 - (min_sum / max_sum).mean()
    return loss