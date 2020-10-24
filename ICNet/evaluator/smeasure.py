import torch
import numpy as np

def calc_smeasure(gt, pred):
    def mean(x):
        return torch.mean(x)

    def cov(x, y):
        mean_x = mean(x)
        mean_y = mean(y)
        return torch.mean((x - mean_x) * (y - mean_y))

    def ssim(x, y):
        mean_x = mean(x)
        mean_y = mean(y)
        cov_x_x = cov(x, x)
        cov_y_y = cov(y, y)
        cov_x_y = cov(x, y)
        a = 4.0 * mean_x * mean_y * cov_x_y
        b = (mean_x ** 2 + mean_y ** 2) * (cov_x_x + cov_y_y)
        return a / (b + 1e-12)

    def O(x, mask):
        mean = torch.sum(x * mask) / (1e-12 + torch.sum(mask))
        var = torch.sqrt(torch.sum(((x - mean) ** 2) * mask) / (1e-12 + torch.sum(mask)))
        return mean * 2.0 / (1.0 + mean ** 2 + var)        

    def centroid(y):
        h, w = y.shape
        total = 1e-12 + torch.sum(y)
        hw = int(torch.round(torch.sum(torch.sum(y, axis=0) * torch.from_numpy(np.array(range(1, 1 + w))).cuda()) / total))
        hh = int(torch.round(torch.sum(torch.sum(y, axis=1) * torch.from_numpy(np.array(range(1, 1 + h))).cuda()) / total))

        area = h * w
        w1 = hh * hw / area
        w2 = hh * (w - hw) / area
        w3 = (h - hh) * hw / area
        w4 = 1.0 - w1 - w2 - w3
        return hh, hw, h, w, w1, w2, w3, w4

    def seg(x, hh, hw, h, w):
        x1 = x[0:hh, 0:hw]
        x2 = x[0:hh, hw:w]
        x3 = x[hh:h, 0:hw]
        x4 = x[hh:h, hw:w]
        return x1, x2, x3, x4

    def Sr(x, y):
        hh, hw, h, w, w1, w2, w3, w4 = centroid(y)
        x1, x2, x3, x4 = seg(x, hh, hw, h, w)
        y1, y2, y3, y4 = seg(y, hh, hw, h, w)
        return ssim(x1, y1) * w1 + ssim(x2, y2) * w2 + ssim(x3, y3) * w3 + ssim(x4, y4) * w4

    def So(x, y):
        mu = mean(y)
        return O(x, y) * mu + O(1.0 - x, 1.0 - y) * (1.0 - mu)

    def Sm(x, y):
        return Sr(x, y) * 0.5 + So(x, y) * 0.5

    return Sm(pred, gt)