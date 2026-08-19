"""
Converts the Kaggle surface-crack-detection CLASSIFICATION dataset
into a SEGMENTATION dataset ready for training.

Dataset: https://www.kaggle.com/datasets/arunrk7/surface-crack-detection
  Positive/  ← 20,000 crack images  (227×227 RGB)
  Negative/  ← 20,000 no-crack images  (not used)

What this script does
─────────────────────
1. Reads every image from Positive/
2. Generates a binary crack mask using the Otsu method  (same as the paper)
3. Scores each mask for quality (rejects nearly-empty or too-noisy masks)
4. Keeps the N best images + their masks
5. Saves them into:
     data/images/   ← crack photos
     data/masks/    ← binary PNG masks  (white=crack, black=background)

Usage
─────
# Download the dataset from Kaggle first, then:
python prepare_kaggle_dataset.py --positive_dir path/to/Positive \
                                 --out_dir      data \
                                 --n_images     600

How to download from Kaggle
────────────────────────────
Option A (browser):
  1. Go to https://www.kaggle.com/datasets/arunrk7/surface-crack-detection
  2. Click Download → unzip → you get  Positive/  and  Negative/  folders

Option B (Kaggle API):
  pip install kaggle
  kaggle datasets download -d arunrk7/surface-crack-detection
  unzip surface-crack-detection.zip
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────
# Otsu-based mask generator  (paper §2.2)
# ─────────────────────────────────────────────

def generate_otsu_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Converts an RGB crack image to a binary mask using the Otsu method.
    Returns a mask where 255 = crack, 0 = background.

    Steps from the paper:
      1. Convert to grayscale
      2. Compute Otsu threshold
      3. Classify pixels  (dark pixels = crack for concrete images)
      4. Apply median filter to reduce noise
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Otsu threshold
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Median filter (3×3) to remove noise  (paper step 4)
    mask = cv2.medianBlur(binary, 3)

    return mask


# ─────────────────────────────────────────────
# Mask quality scorer
# ─────────────────────────────────────────────

def quality_score(mask: np.ndarray) -> float:
    """
    Returns a quality score between 0 and 1.
    A good crack mask should:
      - Have some crack pixels (not nearly empty)
      - Not be mostly white (not over-segmented / noisy)
      - Have connected crack regions (not scattered dots)

    Returns 0.0 for bad masks, higher = better.
    """
    h, w     = mask.shape
    total_px = h * w
    crack_px = int((mask > 0).sum())
    ratio    = crack_px / total_px

    # Reject masks where crack covers < 1% or > 60% of the image
    if ratio < 0.01 or ratio > 0.60:
        return 0.0

    # Reward masks with connected structure (fewer, larger blobs = better)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels < 2:          # only background
        return 0.0

    # Largest crack component relative to total crack pixels
    component_sizes = stats[1:, cv2.CC_STAT_AREA]  # exclude background
    largest = component_sizes.max()
    connectivity_score = largest / (crack_px + 1e-8)

    # Final score: prefer 5–25% crack coverage, well-connected
    coverage_score = 1.0 - abs(ratio - 0.12) / 0.12   # peak at 12%
    coverage_score = max(0.0, coverage_score)

    return float(connectivity_score * 0.6 + coverage_score * 0.4)


# ─────────────────────────────────────────────
# Optional: morphological cleanup
# ─────────────────────────────────────────────

def refine_mask(mask: np.ndarray) -> np.ndarray:
    """
    Remove tiny isolated noise blobs (< 50 pixels).
    Keeps only reasonably-sized crack components.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    refined = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 50:
            refined[labels == i] = 255
    return refined


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def prepare(positive_dir: str, out_dir: str,
            n_images: int, img_size: int,
            score_threshold: float):

    positive_dir = Path(positive_dir)
    out_images   = Path(out_dir) / "images"
    out_masks    = Path(out_dir) / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    # Find all images
    all_imgs = sorted([
        p for p in positive_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ])
    print(f"\nFound {len(all_imgs):,} images in {positive_dir}")

    if len(all_imgs) == 0:
        print("ERROR: No images found. Check --positive_dir path.")
        return

    # Score all images
    print(f"\nScoring masks (Otsu method)...")
    scored = []
    for img_path in tqdm(all_imgs, desc="  Scoring"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        mask  = generate_otsu_mask(img)
        score = quality_score(mask)
        if score >= score_threshold:
            scored.append((score, img_path, img, mask))

    print(f"  {len(scored):,} images passed quality threshold ({score_threshold})")

    if len(scored) == 0:
        print("\nERROR: No images passed the quality filter.")
        print("Try lowering --score_threshold (current: {score_threshold})")
        return

    # Sort by score, keep top N
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = scored[:n_images]
    print(f"  Keeping top {len(selected)} images by quality score")

    # Save
    print(f"\nSaving to {out_dir}/images/ and {out_dir}/masks/ ...")
    saved = 0
    for rank, (score, img_path, img, mask) in enumerate(
            tqdm(selected, desc="  Saving")):

        # Resize to target size
        img_out  = cv2.resize(img,  (img_size, img_size))
        mask_out = cv2.resize(mask, (img_size, img_size),
                              interpolation=cv2.INTER_NEAREST)
        mask_out = refine_mask(mask_out)

        # Name: rank_originalname
        stem     = f"{rank:04d}_{img_path.stem}"
        cv2.imwrite(str(out_images / f"{stem}.jpg"),  img_out)
        cv2.imwrite(str(out_masks  / f"{stem}.png"),  mask_out)
        saved += 1

    print(f"\n✓ Saved {saved} image+mask pairs")
    print(f"  Images → {out_images}")
    print(f"  Masks  → {out_masks}")

    # ── Show quality stats ────────────────────────────────────────
    scores = [s for s, *_ in selected]
    print(f"\n  Quality score stats:")
    print(f"    Min  : {min(scores):.3f}")
    print(f"    Mean : {float(np.mean(scores)):.3f}")
    print(f"    Max  : {max(scores):.3f}")

    # ── Save a visual sample grid ─────────────────────────────────
    _save_sample_grid(selected[:12], out_dir)
    print(f"\n  Preview grid → {out_dir}/sample_preview.jpg")
    print(f"\nNext step: python STEP2_train.py "
          f"--img_dir {out_dir}/images --mask_dir {out_dir}/masks")


def _save_sample_grid(selected, out_dir, cols=4):
    """Save a grid of 12 sample (image, mask) pairs for visual inspection."""
    rows   = (len(selected) + cols - 1) // cols
    cell_h, cell_w = 227, 227 * 2 + 4   # image | mask side by side
    grid   = np.zeros((rows * (cell_h + 4), cols * (cell_w + 4), 3),
                      dtype=np.uint8)

    for i, (score, img_path, img, mask) in enumerate(selected):
        row, col = divmod(i, cols)
        y = row * (cell_h + 4)
        x = col * (cell_w + 4)

        img_r  = cv2.resize(img, (227, 227))
        mask_r = cv2.resize(mask, (227, 227),
                            interpolation=cv2.INTER_NEAREST)
        mask_3 = cv2.cvtColor(mask_r, cv2.COLOR_GRAY2BGR)

        combined = np.hstack([img_r, np.full((227, 4, 3), 60, np.uint8), mask_3])
        grid[y:y+cell_h, x:x+cell_w] = combined

        # Score label
        cv2.putText(grid, f"{score:.2f}", (x+4, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(str(Path(out_dir) / "sample_preview.jpg"), grid)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Kaggle crack dataset → segmentation masks")

    parser.add_argument(
        "--positive_dir", required=True,
        help="Path to Kaggle 'Positive' folder (images with cracks)")
    parser.add_argument(
        "--out_dir", default="data",
        help="Output folder (default: data/)")
    parser.add_argument(
        "--n_images", type=int, default=600,
        help="How many images to keep (default: 600). "
             "Paper used 510. More = better but slower to train.")
    parser.add_argument(
        "--img_size", type=int, default=256,
        help="Resize images to this size in pixels (default: 256)")
    parser.add_argument(
        "--score_threshold", type=float, default=0.15,
        help="Minimum quality score to keep an image (default: 0.15). "
             "Lower = keep more images, higher = stricter filter.")

    args = parser.parse_args()
    prepare(args.positive_dir, args.out_dir,
            args.n_images, args.img_size, args.score_threshold)
