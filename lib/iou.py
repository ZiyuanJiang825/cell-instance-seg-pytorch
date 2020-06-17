import numpy as np
import torch


def iou(x, y):
    x = torch.squeeze(x).detach().numpy()
    x = x.astype(np.float32)
    y = torch.squeeze(y).detach().numpy()
    y = y.astype(np.float32)
    x[x < 0.5] = 0
    inters = np.logical_and(x, y)
    unit = np.logical_or(x, y)
    result = np.sum(inters > 0) / np.sum(unit > 0)
    return result
