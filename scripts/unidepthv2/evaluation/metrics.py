# evaluation/metrics.py
import torch

def absrel(pred, gt):
    return torch.mean(torch.abs(pred - gt) / gt)

def rmse(pred, gt):
    return torch.sqrt(torch.mean((pred - gt)**2))

def delta(pred, gt, t):
    r = torch.max(pred/gt, gt/pred)
    return (r < t).float().mean()
