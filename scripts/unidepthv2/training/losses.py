# training/losses.py
import torch
import kornia

def metric_depth_loss(pred, gt, gtrs):
    absrel = torch.abs(pred - gt) / (gt + 1e-6)
    return (gtrs * absrel.mean(dim=[1,2])).mean()

def pose_consistency_loss(D1, D2, T, K):
    # Warp D1 → view2
    D1_warp = warp_depth(D1, T, K)
    return torch.median(torch.abs(D1_warp - D2)) / torch.median(D2)


def structural_edge_loss(pred, img):
    edges_d = kornia.filters.canny(pred.unsqueeze(1))[0]
    edges_i = kornia.filters.canny(img)[0]
    return 1 - kornia.metrics.fscore(edges_d, edges_i)

def total_loss(batch, model):
    pred = model(batch["image"])

    L_depth = metric_depth_loss(pred, batch["depth"], batch["gtrs"])

    L_struct = structural_edge_loss(pred, batch["image"])

    if batch["pose"] is not None:
        L_pose = pose_consistency_loss(...)
    else:
        L_pose = 0.0
    
    # REMOVE pose loss completely
    return L_depth + 0.3 * L_struct

    #return L_depth + 0.3*L_struct + 0.2*L_pose
