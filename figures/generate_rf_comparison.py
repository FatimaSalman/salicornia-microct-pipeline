#!/usr/bin/env python3
"""
generate_rf_comparison.py
================================

Creates a visual RF-versus-reference comparison for:

  S5 / Ciechocinek 0 mM:
      moderate RF agreement and directional under-segmentation
      Dice = 0.742; Recall = 0.704

  S8 / Ciechocinek 1000 mM:
      near-complete RF-reference agreement
      Dice = 0.975; Recall = 0.966

Output:
  Figure6_RF_reference_prediction_comparison.png

Rows:
  S5 (C-0), S8 (C-1000)

Columns:
  1. Reference CC mask
  2. RF-predicted CC mask
  3. Voxel-wise error map:
       green = true positive
       blue  = false negative
       red   = false positive
"""

import os

# Root of the data tree. Override with:  export SALICORNIA_DATA=/path
DATA = os.environ.get("SALICORNIA_DATA", ".")


from pathlib import Path

import numpy as np
import nrrd
import tifffile
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from skimage.transform import resize


# ==========================================================
# Configuration
# ==========================================================

BASE = Path(DATA)

CC_CLASS = -1

SAMPLES = [
    {
        "id": "S5",
        "title": "Ciechocinek, 0 mM",
        "prediction_path": (
            BASE / "Salicornia_5" / "image_folder" /
            "Salicornia5_" / "MLresults" /
            "Salicornia5_fullstack_prediction.tif"
        ),
        # IMPORTANT:
        # This is the actual final reference used by evaluation_slices.py.
        "reference_path": (
            BASE / "Salicornia_5" /
            "C-0_cc_reference_mask.nrrd"
        ),
        "z": 500,
        "dice": 0.742,
        "precision": 0.785,
        "recall": 0.704,
        "volume_difference": "RF volume −10.4%",
    },
    {
        "id": "S8",
        "title": "Ciechocinek, 1000 mM",
        "prediction_path": (
            BASE / "Salicornia_8" / "image_folder" /
            "Salicornia8_" / "MLresults" /
            "Salicornia8_fullstack_prediction.tif"
        ),
        # IMPORTANT:
        # This is the actual final reference used by evaluation_slices.py.
        "reference_path": (
            BASE / "Salicornia_8" /
            "cc_labelmap.nrrd"
        ),
        "z": 492,
        "dice": 0.975,
        "precision": 0.983,
        "recall": 0.966,
        "volume_difference": "RF volume −1.8%",
    },
]

OUTPUT_PATH = BASE / "Figure6_RF_reference_prediction_comparison.png"


# ==========================================================
# Helper functions
# ==========================================================

def resize_binary_mask(mask_yx, output_shape):
    """Nearest-neighbour resize of a binary reference mask."""
    resized = resize(
        mask_yx.astype(np.uint8),
        output_shape=output_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return resized >= 0.5


def load_reference_as_zyx(path):
    """
    Reference NRRD is stored as X,Y,Z.
    Convert it to the TIFF/RF convention Z,Y,X.
    """
    reference_xyz, _ = nrrd.read(str(path))

    if reference_xyz.ndim != 3:
        raise ValueError(
            f"Expected a 3D labelmap NRRD, got shape "
            f"{reference_xyz.shape} from {path}"
        )

    return np.transpose(reference_xyz, (2, 1, 0)).astype(bool)


def crop_around_union(reference_mask, predicted_mask, padding=25):
    """
    Crop both masks around their union to make the CC comparison readable.
    """
    union = reference_mask | predicted_mask

    ys, xs = np.where(union)

    if len(ys) == 0:
        raise ValueError("No foreground voxels found in selected slice.")

    y0 = max(0, ys.min() - padding)
    y1 = min(union.shape[0], ys.max() + padding + 1)
    x0 = max(0, xs.min() - padding)
    x1 = min(union.shape[1], xs.max() + padding + 1)

    return (
        reference_mask[y0:y1, x0:x1],
        predicted_mask[y0:y1, x0:x1],
    )


def build_error_rgb(reference_mask, predicted_mask):
    """
    RGB error map:
      Green = true positive
      Blue  = false negative
      Red   = false positive
      Black = true negative/background
    """
    tp = reference_mask & predicted_mask
    fn = reference_mask & ~predicted_mask
    fp = ~reference_mask & predicted_mask

    rgb = np.zeros((*reference_mask.shape, 3), dtype=np.float32)

    rgb[tp] = (0.10, 0.70, 0.20)  # green
    rgb[fn] = (0.15, 0.35, 0.95)  # blue
    rgb[fp] = (0.90, 0.15, 0.10)  # red

    return rgb


# ==========================================================
# Load selected slices
# ==========================================================

loaded = []

for sample in SAMPLES:
    print(f"\nLoading {sample['id']} ...")

    prediction = np.squeeze(
        tifffile.imread(sample["prediction_path"])
    )

    reference_zyx = load_reference_as_zyx(
        sample["reference_path"]
    )

    # S8 reference has 985 slices while prediction has 984.
    # This exactly matches the documented final-slice trimming.
    if reference_zyx.shape[0] > prediction.shape[0]:
        reference_zyx = reference_zyx[:prediction.shape[0]]

    z = sample["z"]

    if z >= prediction.shape[0]:
        raise ValueError(
            f"{sample['id']}: selected z={z} is outside "
            f"prediction stack with {prediction.shape[0]} slices."
        )

    ref_mask = resize_binary_mask(
        reference_zyx[z],
        prediction[z].shape,
    )

    pred_mask = prediction[z] == CC_CLASS

    ref_crop, pred_crop = crop_around_union(
        ref_mask,
        pred_mask,
        padding=30,
    )

    error_rgb = build_error_rgb(
        ref_crop,
        pred_crop,
    )

    loaded.append({
        **sample,
        "reference": ref_crop,
        "prediction": pred_crop,
        "error_rgb": error_rgb,
    })

    print(
        f"  z={z}; reference pixels={ref_crop.sum():,}; "
        f"RF pixels={pred_crop.sum():,}"
    )


# ==========================================================
# Plot
# ==========================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(10.5, 7.0),
    dpi=300,
)

column_titles = [
    "Reference CC mask",
    "RF-predicted CC mask",
    "Voxel-wise comparison",
]

for col, title in enumerate(column_titles):
    axes[0, col].set_title(
        title,
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

for row, item in enumerate(loaded):
    reference = item["reference"]
    prediction = item["prediction"]
    error_rgb = item["error_rgb"]

    # Column 1: reference mask
    axes[row, 0].imshow(
        reference,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # Column 2: RF-predicted mask
    axes[row, 1].imshow(
        prediction,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # Column 3: TP / FN / FP error representation
    axes[row, 2].imshow(
        error_rgb,
        interpolation="nearest",
    )

    axes[row, 0].set_ylabel(
        f"{item['title']}\n"
        f"Dice = {item['dice']:.3f}\n"
        f"{item['volume_difference']}",
        fontsize=10,
        fontweight="bold",
    )

    for col in range(3):
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        axes[row, col].set_aspect("equal")

# Error-map legend
legend_handles = [
    Patch(facecolor=(0.10, 0.70, 0.20), label="True positive"),
    Patch(facecolor=(0.15, 0.35, 0.95), label="False negative (missed CC)"),
    Patch(facecolor=(0.90, 0.15, 0.10), label="False positive"),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=9,
    bbox_to_anchor=(0.5, 0.01),
)

fig.text(
    0.5,
    0.965,
    "Random Forest prediction compared with the three-dimensional reference mask",
    ha="center",
    va="top",
    fontsize=13,
    fontweight="bold",
)

plt.tight_layout(
    rect=(0.02, 0.06, 0.98, 0.93)
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(f"\nSaved: {OUTPUT_PATH}")