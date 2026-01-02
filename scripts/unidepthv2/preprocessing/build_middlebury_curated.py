import os
import json
import cv2
import numpy as np
import torch
import re
from tqdm import tqdm

# ============================================================
# Paths (EDIT ONLY ROOT)
# ============================================================
RAW_ROOT = "data/raw/middlebury"
OUT_ROOT = "data/curated"

IMG_OUT = os.path.join(OUT_ROOT, "images")
DEPTH_OUT = os.path.join(OUT_ROOT, "depth_gt")

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(DEPTH_OUT, exist_ok=True)

# ============================================================
# Utilities
# ============================================================

def load_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().decode("ascii").rstrip()
        color = header == "PF"

        dims = f.readline().decode("ascii").strip()
        w, h = map(int, dims.split())

        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"

        data = np.fromfile(f, endian + "f")
        data = data.reshape((h, w, 3) if color else (h, w))
        data = np.flipud(data)

    # ---- CRITICAL SANITIZATION ----
    data = data.astype(np.float32)

    # Replace inf / nan
    data[~np.isfinite(data)] = 0.0

    # Clamp absurd values (Middlebury convention)
    data[data > 1e4] = 0.0

    return data



def load_png(path):
    """
    Loads PNG/JPG image and returns float32 image
    in the same convention as load_pfm():
    - RGB
    - float32
    - range [0, 1]
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Failed to load image: {path}")

    # If grayscale, convert to 3-channel
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)

    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to float32 [0,1]
    img = img.astype(np.float32) / 255.0

    return img


def parse_calib(path):
    fx = None
    fy = None
    baseline = None

    with open(path, "r") as file:
        for line in file:
            line = line.strip()

            # ---- cam0 intrinsic matrix ----
            if line.startswith("cam0="):
                # Extract only what's inside [...]
                inside = line.split("[", 1)[1].split("]", 1)[0]

                # Now extract numbers safely
                nums = list(map(float,
                                re.findall(r"[-+]?\d*\.\d+|\d+", inside)))
                # nums = [fx, 0, cx, 0, fy, cy, 0, 0, 1]

                fx = nums[0]
                fy = nums[4]

            # ---- baseline ----
            elif line.startswith("baseline="):
                baseline = float(line.split("=")[1])

    if fx is None or fy is None:
        raise ValueError(f"Failed to parse cam0 intrinsics from {path}")
    if baseline is None:
        raise ValueError(f"Failed to parse baseline from {path}")

    f = (fx + fy) / 2.0
    baseline = baseline / 1000.0  # mm → meters

    return {
        "f": f,
        "baseline": baseline
    }


def disparity_to_depth(disp, f, B):
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = disp > 0
    depth[valid] = (f * B) / disp[valid]
    return depth


# ============================================================
# GTRS COMPONENTS (MONOCULAR)
# ============================================================

def structural_score(depth, image):
    """
    Structure alignment via gradient correlation
    (robust for metric depth).
    """
    d = depth.copy()
    d[d <= 0] = np.nan

    logd = np.log(d)
    logd = np.nan_to_num(logd)

    dzx, dzy = np.gradient(logd)

    img_gray = cv2.cvtColor((image * 255).astype(np.uint8),
                            cv2.COLOR_RGB2GRAY).astype(np.float32)
    ix, iy = np.gradient(img_gray)

    # Normalize gradients
    dz_mag = np.sqrt(dzx**2 + dzy**2)
    i_mag = np.sqrt(ix**2 + iy**2)

    mask = (dz_mag > np.percentile(dz_mag, 90))  # top 10% depth changes

    if mask.sum() < 100:
        return 0.0

    corr = np.corrcoef(dz_mag[mask].flatten(),
                       i_mag[mask].flatten())[0, 1]

    return float(np.clip((corr + 1) / 2, 0, 1))



def smoothness_score(depth):
    d = depth.copy()
    d[d <= 0] = np.nan

    logd = np.log(d)
    logd = np.nan_to_num(logd)

    gx, gy = np.gradient(logd)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Look at smoothest regions only
    smooth_region = grad_mag < np.percentile(grad_mag, 30)

    if smooth_region.sum() < 100:
        return 0.0

    sigma = np.std(grad_mag[smooth_region])
    return float(1 / (1 + sigma))




def gradient_alignment(depth, image):
    dz = np.gradient(depth)
    img_gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    di = np.gradient(img_gray.astype(np.float32))

    g = np.mean(np.abs(dz[0] - di[0])) + np.mean(np.abs(dz[1] - di[1]))
    return float(1 / (1 + g))


# ============================================================
# MAIN LOOP
# ============================================================

gtrs_scores = {}

scenes = sorted(os.listdir(RAW_ROOT))

for scene in tqdm(scenes, desc="Processing Middlebury scenes"):
    scene_dir = os.path.join(RAW_ROOT, scene)
    print(scene_dir)
    if not os.path.isdir(scene_dir):
        print("no path")
        continue

    img_path = os.path.join(scene_dir, "im0.png")
    disp_path = os.path.join(scene_dir, "disp0.pfm")
    calib_path = os.path.join(scene_dir, "calib.txt")

    if not all(os.path.exists(p) for p in [img_path, disp_path, calib_path]):
        continue

    # --- Load data ---
    img = load_png(img_path)
    disp = load_pfm(disp_path)

    # ---- sanitize disparity ----
    disp = disp.astype(np.float32)
    disp[~np.isfinite(disp)] = 0.0
    disp[disp <= 0] = 0.0

    calib = parse_calib(calib_path)
    print("CALIB:", calib)

    depth = disparity_to_depth(disp, calib["f"], calib["baseline"])

    print(scene, img_path, disp_path, calib_path)


    # --- Normalize image for saving ---
    #img_8u = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img_norm = img - img.min()
    img_norm = img_norm / (img_norm.max() + 1e-6)
    img_8u = (img_norm * 255).astype(np.uint8)

    img_name = f"{scene}.png"
    depth_name = f"{scene}.npy"

    cv2.imwrite(os.path.join(IMG_OUT, img_name),
                cv2.cvtColor(img_8u, cv2.COLOR_RGB2BGR))
    np.save(os.path.join(DEPTH_OUT, depth_name), depth)

    # --- GTRS computation ---
    E_struct = structural_score(depth, img)
    E_smooth = smoothness_score(depth)
    E_grad = gradient_alignment(depth, img)

    GTRS = (
        0.45 * E_struct +
        0.25 * E_smooth +
        0.30 * E_grad
    )


    gtrs_scores[scene] = {
        "image": f"{IMG_OUT}/{img_name}",
        "depth": f"{DEPTH_OUT}/{depth_name}",
        "E_struct": E_struct,
        "E_smooth": E_smooth,
        "E_grad": E_grad,
        "score": GTRS
    }

# ============================================================
# SAVE JSON FILES
# ============================================================

with open(os.path.join(OUT_ROOT, "gtrs_scores.json"), "w") as f:
    json.dump(gtrs_scores, f, indent=2)

# Filtered
gtrs_filtered = {
    k: v for k, v in gtrs_scores.items() if v["score"] > 0.55 #keep between 0.53 to 0.6
}

with open(os.path.join(OUT_ROOT, "gtrs_filtered.json"), "w") as f:
    json.dump(gtrs_filtered, f, indent=2)


print(f"\nTotal scenes: {len(gtrs_scores)}")
print(f"Kept after GTRS filtering: {len(gtrs_filtered)}")
