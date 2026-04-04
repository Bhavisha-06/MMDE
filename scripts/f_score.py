import numpy as np
import cv2

def compute_edges(depth, thresh=0.02):
    """
    Extract binary edge map from a depth map using Sobel gradients.

    Args:
        depth (np.ndarray): Depth map (H, W), any dtype.
        thresh (float): Normalized gradient threshold (0-1).

    Returns:
        np.ndarray: Binary edge mask (H, W), dtype=bool
    """
    depth = depth.astype(np.float32)
    # Normalize to [0,1] for stable thresholding
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    # Sobel gradients
    gx = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Binary edge mask
    edges = grad_mag > thresh
    return edges


def fscore_2d_edges(gt_depth, pred_depth, thresh=0.02):
    """
    Compute 2D F-score based on edge maps of GT and prediction.

    Args:
        gt_depth (np.ndarray): Ground truth depth (H, W).
        pred_depth (np.ndarray): Predicted depth (H, W).
        thresh (float): Gradient threshold for edge extraction.

    Returns:
        precision, recall, fscore
    """
    gt_edges = compute_edges(gt_depth, thresh)
    pred_edges = compute_edges(pred_depth, thresh)

    tp = np.logical_and(pred_edges, gt_edges).sum()   # True positives
    fp = np.logical_and(pred_edges, np.logical_not(gt_edges)).sum()  # False positives
    fn = np.logical_and(np.logical_not(pred_edges), gt_edges).sum()  # False negatives

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    fscore = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, fscore


# --- Example usage ---
if __name__ == "__main__":
    # Load GT depth (8-bit PNG)
    gt_path = "/Users/bhavishachaudhari/Desktop/1_gt.png"
    gt_depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

    # Load predicted depth (.npy)
    pred_path = "/Users/bhavishachaudhari/Desktop/iitm/depth Estimation/testimages/1_distances.npy"
    pred_depth = np.load(pred_path)

    # Ensure same resolution (resize prediction if needed)
    if pred_depth.shape != gt_depth.shape:
        pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)

    prec, rec, f = fscore_2d_edges(gt_depth, pred_depth, thresh=0.02)
    print(f"Precision={prec:.3f}, Recall={rec:.3f}, F-score={f:.3f}")
