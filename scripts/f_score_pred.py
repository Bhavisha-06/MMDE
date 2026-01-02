import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_canny_edges(img, low=100, high=200):
    """Extract edges from RGB image using Canny."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    edges = cv2.Canny(gray, low, high)
    return edges.astype(bool)

def compute_sobel_edges(depth, thresh=0.02):
    """Extract edges from depth map using Sobel gradients."""
    depth = depth.astype(np.float32)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    gx = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    edges = grad_mag > thresh
    return edges

def fscore_edges(gt_edges, pred_edges, visualize=True):
    """Compute precision, recall, and F-score between two edge maps."""
    tp = np.logical_and(pred_edges, gt_edges).sum()
    fp = np.logical_and(pred_edges, np.logical_not(gt_edges)).sum()
    fn = np.logical_and(np.logical_not(pred_edges), gt_edges).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    fscore = 2 * precision * recall / (precision + recall + 1e-8)

    if visualize:
        overlap = np.zeros((*gt_edges.shape, 3), dtype=np.uint8)
        overlap[gt_edges] = [0, 255, 0]      # Green = GT edge (RGB)
        overlap[pred_edges] = [255, 0, 0]    # Red = Pred depth edge
        overlap[np.logical_and(gt_edges, pred_edges)] = [255, 255, 0]  # Yellow = match

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(gt_edges, cmap="gray")
        axs[0].set_title("GT Edges (RGB Canny)")
        axs[1].imshow(pred_edges, cmap="gray")
        axs[1].set_title("Pred Depth Edges (Sobel)")
        axs[2].imshow(overlap)
        axs[2].set_title("Overlap (Green=GT, Red=Pred, Yellow=Match)")
        for ax in axs:
            ax.axis("off")
        plt.show()

    return precision, recall, fscore


# --- Example usage ---
if __name__ == "__main__":
    rgb_path = "/Users/bhavishachaudhari/Desktop/iitm/MMDE/F-scores/1.png"     # RGB input image
    gt_depth_path = "/Users/bhavishachaudhari/Desktop/iitm/MMDE/F-scores/1_gt.png"   # GT depth (optional, just for comparison)
    pred_depth_path = "/Users/bhavishachaudhari/Desktop/iitm/MMDE/F-scores/unidepth_pred.npy"

    # Load inputs
    rgb = cv2.imread(rgb_path)
    pred_depth = np.load(pred_depth_path)

    # Resize depth to match image if needed
    if pred_depth.shape != rgb.shape[:2]:
        pred_depth = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Extract edges
    gt_edges = compute_canny_edges(rgb)
    pred_edges = compute_sobel_edges(pred_depth, thresh=0.02)

    # Compute metrics
    prec, rec, f = fscore_edges(gt_edges, pred_edges, visualize=True)
    print(f"Precision={prec:.3f}, Recall={rec:.3f}, F-score={f:.3f}")
