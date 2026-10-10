#!/usr/bin/env python3

import os

# Root of the data tree. Override with:  export SALICORNIA_DATA=/path
DATA = os.environ.get("SALICORNIA_DATA", ".")


from pathlib import Path
import pickle

import numpy as np
import nrrd
import tifffile
from skimage.transform import resize


# ==========================================================
# Paths and acquisition parameters
# ==========================================================

BASE = Path(
    DATA
)

SAMPLE_DIR = BASE / "Salicornia 1, Inowroclaw 0 mM"

PREDICTION_PATH = (
    SAMPLE_DIR
    / "image_folder"
    / "Salicornia1_"
    / "MLresults"
    / "Salicornia1_fullstack_prediction.tif"
)

REFERENCE_MASK_PATH = (
    SAMPLE_DIR
    / "S1_cc_labelmap_proper.nrrd"
)

RADII_RESULTS_PATH = (
    SAMPLE_DIR
    / "s1_radii_results.pkl"
)

VOXEL_UM = 2.50002
RESCALE_FACTOR = 2

# Final reference result from the radial half-maximum method
REFERENCE_VOLUME_MM3 = 0.07353272957348922


# ==========================================================
# Helper functions
# ==========================================================

def resize_binary_mask(mask_yx, output_shape):
    """
    Resize a binary reference mask using nearest-neighbour
    interpolation to preserve discrete labels.
    """
    result = resize(
        mask_yx.astype(np.uint8),
        output_shape=output_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )

    return result >= 0.5


def safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan

    return numerator / denominator


# ==========================================================
# Load RF prediction
# ==========================================================

print("\nLoading RF prediction...")

prediction = tifffile.imread(PREDICTION_PATH)
prediction = np.squeeze(prediction)

print("Prediction shape:", prediction.shape)
print("Prediction dtype:", prediction.dtype)

class_values, class_counts = np.unique(
    prediction,
    return_counts=True
)

print("\nPrediction classes and full-stack counts:")

for value, count in zip(class_values, class_counts):
    print(f"  class {value}: {count:,} voxels")


if prediction.ndim != 3:
    raise ValueError(
        f"Prediction should be 3D (Z,Y,X), but shape is "
        f"{prediction.shape}"
    )


# ==========================================================
# Load final cc reference mask
# ==========================================================

print("\nLoading final S1 cc reference mask...")

reference_xyz, reference_header = nrrd.read(
    str(REFERENCE_MASK_PATH)
)

print("Original NRRD shape (X,Y,Z):", reference_xyz.shape)
print("Reference unique values:", np.unique(reference_xyz))

# Convert NRRD (X,Y,Z) to TIFF/RF convention (Z,Y,X)
reference_zyx = np.transpose(
    reference_xyz,
    (2, 1, 0)
).astype(bool)

print("Converted reference shape (Z,Y,X):", reference_zyx.shape)


if prediction.shape[0] != reference_zyx.shape[0]:
    raise ValueError(
        "Prediction and reference have different Z dimensions: "
        f"{prediction.shape[0]} vs {reference_zyx.shape[0]}"
    )


# ==========================================================
# Confirm valid axial range from the radial-method output
# ==========================================================

with open(RADII_RESULTS_PATH, "rb") as handle:
    radial_results = pickle.load(handle)

valid_z_from_results = sorted(
    int(item[0]) for item in radial_results["all_radii"]
)

print("\nRadial result information:")
print("  measured slices:", len(valid_z_from_results))
print("  first measured z:", valid_z_from_results[0])
print("  last measured z:", valid_z_from_results[-1])
print("  failed slices:", len(radial_results["failed_slices"]))
print("  stored reference volume:",
      radial_results["volume_mm3"])


# Use exactly the slices quantified by the final radial method.
# This avoids comparing RF predictions in the unmeasured edge slices.
evaluation_slices = [
    z for z in valid_z_from_results
    if 0 <= z < prediction.shape[0]
]

if len(evaluation_slices) != len(valid_z_from_results):
    raise ValueError(
        "Some radial-result slice indices fall outside "
        "the prediction stack."
    )


# ==========================================================
# Compare each RF output class with the cc reference
# ==========================================================

candidate_classes = class_values.tolist()

predicted_counts = {
    value: 0 for value in candidate_classes
}

intersection_counts = {
    value: 0 for value in candidate_classes
}

reference_count = 0

print("\nCalculating overlap with the final reference mask...")

for number, z in enumerate(evaluation_slices, start=1):

    reference_slice_ds = resize_binary_mask(
        reference_zyx[z],
        prediction[z].shape
    )

    reference_count += int(
        np.count_nonzero(reference_slice_ds)
    )

    predicted_slice = prediction[z]

    for value in candidate_classes:

        candidate_mask = predicted_slice == value

        predicted_counts[value] += int(
            np.count_nonzero(candidate_mask)
        )

        intersection_counts[value] += int(
            np.count_nonzero(
                candidate_mask & reference_slice_ds
            )
        )

    if number % 100 == 0 or number == len(evaluation_slices):
        print(
            f"  processed {number}/"
            f"{len(evaluation_slices)} slices"
        )


# ==========================================================
# Calculate overlap metrics
# ==========================================================

results = []

print("\nOverlap with final cc reference:")
print(
    "class | Dice | IoU | precision | recall | "
    "predicted voxels"
)

for value in candidate_classes:

    pred_count = predicted_counts[value]
    intersection = intersection_counts[value]

    union = pred_count + reference_count - intersection

    dice = safe_divide(
        2 * intersection,
        pred_count + reference_count
    )

    iou = safe_divide(
        intersection,
        union
    )

    precision = safe_divide(
        intersection,
        pred_count
    )

    recall = safe_divide(
        intersection,
        reference_count
    )

    results.append({
        "class": value,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "predicted_count": pred_count,
        "intersection": intersection,
    })

    print(
        f"{str(value):>5} | "
        f"{dice:0.4f} | "
        f"{iou:0.4f} | "
        f"{precision:0.4f} | "
        f"{recall:0.4f} | "
        f"{pred_count:,}"
    )


# Select the output class with the highest Dice overlap.
best_result = max(
    results,
    key=lambda item: (
        -np.inf
        if np.isnan(item["dice"])
        else item["dice"]
    )
)

cc_class = best_result["class"]

print("\nSelected cc class:", cc_class)
print("Best Dice:", best_result["dice"])
print("Best IoU:", best_result["iou"])
print("Precision:", best_result["precision"])
print("Recall:", best_result["recall"])


# ==========================================================
# Compute RF cc volume
# ==========================================================

# The RF stack was downsampled only in X and Y.
# Z spacing remains equal to the original slice spacing.
effective_voxel_volume_um3 = (
    VOXEL_UM
    * (VOXEL_UM * RESCALE_FACTOR)
    * (VOXEL_UM * RESCALE_FACTOR)
)

rf_cc_voxels = best_result["predicted_count"]

rf_volume_mm3 = (
    rf_cc_voxels
    * effective_voxel_volume_um3
    / 1e9
)

relative_difference_percent = (
    abs(rf_volume_mm3 - REFERENCE_VOLUME_MM3)
    / REFERENCE_VOLUME_MM3
    * 100
)


# Equivalent radius over the same measured axial length
evaluated_length_um = (
    len(evaluation_slices) * VOXEL_UM
)

rf_equivalent_radius_um = np.sqrt(
    (rf_volume_mm3 * 1e9)
    / (np.pi * evaluated_length_um)
)

rf_equivalent_diameter_um = (
    2 * rf_equivalent_radius_um
)


print("\n========================================")
print("S1 RF QUANTIFICATION")
print("========================================")
print("Evaluated slices:", len(evaluation_slices))
print(f"Evaluated length: {evaluated_length_um:.2f} µm")
print("Reference downsampled cc voxels:",
      f"{reference_count:,}")
print("Selected RF cc voxels:",
      f"{rf_cc_voxels:,}")
print(f"RF cc volume: {rf_volume_mm3:.6f} mm³")
print(
    f"Reference 3D volume: "
    f"{REFERENCE_VOLUME_MM3:.6f} mm³"
)
print(
    f"Relative volume difference: "
    f"{relative_difference_percent:.2f}%"
)
print(
    f"RF equivalent radius: "
    f"{rf_equivalent_radius_um:.2f} µm"
)
print(
    f"RF equivalent diameter: "
    f"{rf_equivalent_diameter_um:.2f} µm"
)
print("========================================")