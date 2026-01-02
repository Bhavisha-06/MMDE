import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_edges_canny(img, low=50, high=150):
    """Compute binary edges using Canny."""
    if img.ndim == 3:  # RGB image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.astype(np.uint8)
    edges = cv2.Canny(gray, low, high)
    return edges.astype(bool)

def compute_edges_depth(depth, low=50, high=150):
    """Compute edges from depth map using Canny after normalization."""
    depth = depth.astype(np.float32)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    edges = cv2.Canny(depth_uint8, low, high)
    return edges.astype(bool)

def fscore_edges(gt_edges, pred_edges, visualize=True):
    """Compute F-score between two edge maps."""
    tp = np.logical_and(pred_edges, gt_edges).sum()
    fp = np.logical_and(pred_edges, np.logical_not(gt_edges)).sum()
    fn = np.logical_and(np.logical_not(pred_edges), gt_edges).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    fscore = 2 * precision * recall / (precision + recall + 1e-8)

    if visualize:
        overlap = np.zeros((*gt_edges.shape, 3), dtype=np.uint8)
        overlap[gt_edges] = [0, 255, 0]      # Green = GT edges
        overlap[pred_edges] = [255, 0, 0]    # Red = Pred edges
        overlap[np.logical_and(gt_edges, pred_edges)] = [255, 255, 0]  # Yellow = match

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(gt_edges, cmap="gray"); axs[0].set_title("GT Edges (RGB)")
        axs[1].imshow(pred_edges, cmap="gray"); axs[1].set_title("Pred Edges (Depth)")
        axs[2].imshow(overlap); axs[2].set_title("Overlap (Yellow=Match)")
        for ax in axs: ax.axis("off")
        plt.show()

    return precision, recall, fscore


# --- Example usage ---
if __name__ == "__main__":
    rgb_path = "/Users/bhavishachaudhari/Desktop/iitm/MMDE/F-scores/1.png"       # RGB image
    gt_depth_path = "/Users/bhavishachaudhari/Desktop/iitm/MMDE/F-scores/1_gt.png"   # GT depth (uint8, optional if you want depth edges too)
    pred_depth_path = "/Users/bhavishachaudhari/Desktop/iitm/MMDE/F-scores/depthpro_pred.npy"

    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    pred_depth = np.load(pred_depth_path)

    # Resize pred depth if needed
    pred_depth = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Edges
    gt_edges = compute_edges_canny(rgb)          # from RGB
    pred_edges = compute_edges_depth(pred_depth) # from prediction

    prec, rec, f = fscore_edges(gt_edges, pred_edges, visualize=True)
    print(f"Precision={prec:.3f}, Recall={rec:.3f}, F-score={f:.3f}")
