import os
import numpy as np
import nibabel as nib
import cv2


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
    if (p99 - p1) < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - p1) / (p99 - p1)
    return np.clip(y, 0, 1).astype(np.float32)


def dice_iou(pred: np.ndarray, gt: np.ndarray):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    inter = int((pred & gt).sum())
    p = int(pred.sum())
    g = int(gt.sum())

    dice = (2.0 * inter) / (p + g + 1e-8)

    union = int((pred | gt).sum())
    iou = inter / (union + 1e-8)

    return float(dice), float(iou)


def save_overlay(vol: np.ndarray, mask: np.ndarray, out_png: str):
    # show mid slice overlay
    H, W, Z = vol.shape
    mid = Z // 2

    img = normalize01(vol[:, :, mid])
    base = (img * 255).astype(np.uint8)
    base_rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    m = mask[:, :, mid].astype(np.uint8)
    overlay = base_rgb.copy()

    # highlight mask in yellow
    overlay[m > 0] = (0.55 * overlay[m > 0] + 0.45 * np.array([0, 255, 255])).astype(np.uint8)

    cv2.imwrite(out_png, overlay)


def run_pipeline(ct_path: str, gt_path: str | None, run_dir: str):
    nii = nib.load(ct_path)
    vol = nii.get_fdata().astype(np.float32)   # (H,W,Z)

    if vol.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI. Got shape {vol.shape}")

    H, W, Z = vol.shape
    artery = np.zeros((H, W, Z), dtype=np.uint8)

    # FAST baseline vessel-ish mask (per-slice top intensity threshold)
    # This WILL vary because each patient's intensity distribution differs.
    for i in range(Z):
        s = normalize01(vol[:, :, i])
        thr = np.percentile(s, 90.0)   # you can adjust 90..95
        artery[:, :, i] = (s > thr).astype(np.uint8)

    artery_coverage = 100.0 * float(artery.sum()) / float(artery.size)


    artery_vals = vol[artery > 0].astype(np.float32)

# ---------------------------
# PLAQUE (HU THRESHOLDS) - STABLE + VARIES PER PATIENT
# ---------------------------

    artery_mask = (artery > 0)  # artery is your predicted/GT artery segmentation (0/1)
    artery_voxels = int(artery_mask.sum())

# defaults (always defined)
    plaque_percent = 0.0
    severe_plaque_percent = 0.0
    t1 = 250.0
    t2 = 500.0
    method = "HU thresholds (130 / 350)"

# Convert to HU-like values (many NIfTI are stored 0..4095; shift to approx HU)
    vol_hu = vol.astype(np.float32)
    if vol_hu.max() > 2000:   # typical sign it's not already HU
        vol_hu = vol_hu - 1024.0

    MIN_ARTERY_VOXELS = 2000  # important: prevent fake zeros caused by empty artery

    if artery_voxels < MIN_ARTERY_VOXELS:   
        method = f"Artery mask too small ({artery_voxels} voxels)"
        plaque_percent = 0.0
        severe_plaque_percent = 0.0
    else:
        plaque_mask = (vol_hu >= t1) & artery_mask
        severe_mask = (vol_hu >= t2) & artery_mask

        plaque_percent = 100.0 * float(plaque_mask.sum()) / float(artery_voxels)
        severe_plaque_percent = 100.0 * float(severe_mask.sum()) / float(artery_voxels)
    # Optional GT Dice/IoU
    dice = None
    iou = None
    if gt_path:
        gt = nib.load(gt_path).get_fdata()
        # make sure shapes match
        if gt.shape == artery.shape:
            dice, iou = dice_iou(artery, gt)

    # Save overlay
    overlay_path = os.path.join(run_dir, "overlay.png")
    save_overlay(vol, artery, overlay_path)

    return {
        "ct_shape": [int(H), int(W), int(Z)],
        "artery_coverage_percent": float(artery_coverage),
        "plaque_method": method,
        "plaque_percent": float(plaque_percent),
        "severe_plaque_percent": float(severe_plaque_percent),
        "t1": t1,
        "t2": t2,
        "dice": dice,
        "iou": iou
    }