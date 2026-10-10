#!/usr/bin/env python3
"""
Generic RF vs 3D-reference-mask evaluation script.
Works for S1, S4, S5, S8 with per-sample configuration.
"""

import os

# Root of the data tree. Override with:  export SALICORNIA_DATA=/path
DATA = os.environ.get("SALICORNIA_DATA", ".")


from pathlib import Path
import numpy as np
import nrrd
import tifffile
from skimage.transform import resize


def resize_binary_mask(mask_yx, output_shape):
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


def evaluate_sample(sample_name, prediction_path, reference_mask_path,
                     voxel_um, rescale_factor, reference_volume_mm3,
                     z_start=None, z_end=None):
    """
    z_start, z_end: inclusive nonzero-coverage range of the reference mask
                    (None -> auto-detect from mask itself)
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING: {sample_name}")
    print(f"{'='*60}")

    # --- Load prediction ---
    prediction = np.squeeze(tifffile.imread(prediction_path))
    print("Prediction shape:", prediction.shape)

    class_values, class_counts = np.unique(prediction, return_counts=True)
    print("\nFull-stack class counts:")
    for v, c in zip(class_values, class_counts):
        print(f"  class {v}: {c:,} voxels")

    # --- Load reference mask, convert to (Z,Y,X) ---
    reference_xyz, _ = nrrd.read(str(reference_mask_path))
    reference_zyx = np.transpose(reference_xyz, (2, 1, 0)).astype(bool)
    auto_ref_vol = reference_zyx.sum() * voxel_um**3 / 1e9
    print(f"Reference volume from mask: {auto_ref_vol:.6f} mm3")
    if abs(auto_ref_vol - reference_volume_mm3) / reference_volume_mm3 > 0.01:
        print(f"  WARNING: hard-coded value is {reference_volume_mm3:.6f} - MISMATCH")  
    print("Reference shape (Z,Y,X):", reference_zyx.shape)

    # Allow known, documented Z-trimming mismatches (e.g. S8 pipeline
    # trims the LAST slice when stack Z-dimension isn't divisible by rescale_factor) 
    z_diff= reference_zyx.shape[0] - prediction.shape[0]
    if z_diff < 0:
        raise ValueError(
            f"Reference has FEWER Z slices than prediction "
            f"({reference_zyx.shape[0]} vs {prediction.shape[0]}) - unexpected."
        )
    if z_diff > 0:
        print(f"Note: reference mask has {z_diff} more slice(s) than prediction "
              f"(RF pipeline trims from the END of the stack). "
              f"Reference will be cropped to prediction.shape[0]={prediction.shape[0]}.")
        reference_zyx = reference_zyx[:prediction.shape[0]]

    if prediction.shape[0] != reference_zyx.shape[0]:
        raise ValueError(
            f"Z mismatch: prediction={prediction.shape[0]}, "
            f"reference={reference_zyx.shape[0]}"
        )

    # --- Determine evaluation z range ---
    if z_start is None or z_end is None:
        nonzero_per_z = reference_zyx.reshape(reference_zyx.shape[0], -1).sum(axis=1)
        nonzero_idx = np.where(nonzero_per_z > 0)[0]
        z_start, z_end = nonzero_idx[0], nonzero_idx[-1]

    evaluation_slices = list(range(z_start, z_end + 1))
    print(f"Evaluation z-range: {z_start}-{z_end} ({len(evaluation_slices)} slices)")

    # --- Compute overlap per class ---
    candidate_classes = class_values.tolist()
    predicted_counts = {v: 0 for v in candidate_classes}
    intersection_counts = {v: 0 for v in candidate_classes}
    reference_count = 0

    for i, z in enumerate(evaluation_slices, 1):
        ref_slice_ds = resize_binary_mask(reference_zyx[z], prediction[z].shape)
        reference_count += int(np.count_nonzero(ref_slice_ds))

        pred_slice = prediction[z]
        for v in candidate_classes:
            cmask = pred_slice == v
            predicted_counts[v] += int(np.count_nonzero(cmask))
            intersection_counts[v] += int(np.count_nonzero(cmask & ref_slice_ds))

        if i % 100 == 0 or i == len(evaluation_slices):
            print(f"  processed {i}/{len(evaluation_slices)}")

    # --- Metrics per class ---
    results = []
    print("\nclass | Dice | IoU | precision | recall | predicted voxels")
    for v in candidate_classes:
        pred_c = predicted_counts[v]
        inter = intersection_counts[v]
        union = pred_c + reference_count - inter

        dice = safe_divide(2 * inter, pred_c + reference_count)
        iou = safe_divide(inter, union)
        precision = safe_divide(inter, pred_c)
        recall = safe_divide(inter, reference_count)

        results.append(dict(class_value=v, dice=dice, iou=iou,
                             precision=precision, recall=recall,
                             predicted_count=pred_c, intersection=inter))
        print(f"{str(v):>5} | {dice:.4f} | {iou:.4f} | "
              f"{precision:.4f} | {recall:.4f} | {pred_c:,}")

    best = max(results, key=lambda r: -np.inf if np.isnan(r["dice"]) else r["dice"])
    cc_class = best["class_value"]
    print(f"\nSelected cc class: {cc_class}  (Dice={best['dice']:.4f})")

    # --- Volume computation ---
    effective_voxel_volume_um3 = voxel_um * (voxel_um * rescale_factor) ** 2
    rf_cc_voxels = best["predicted_count"]
    rf_volume_mm3 = rf_cc_voxels * effective_voxel_volume_um3 / 1e9

    rel_diff_pct = abs(rf_volume_mm3 - reference_volume_mm3) / reference_volume_mm3 * 100

    evaluated_length_um = len(evaluation_slices) * voxel_um
    rf_equiv_radius_um = np.sqrt((rf_volume_mm3 * 1e9) / (np.pi * evaluated_length_um))

    print(f"\n--- {sample_name} SUMMARY ---")
    print(f"RF cc volume: {rf_volume_mm3:.6f} mm³")
    print(f"Reference volume: {reference_volume_mm3:.6f} mm³")
    print(f"Relative difference: {rel_diff_pct:.2f}%")
    print(f"RF equivalent radius: {rf_equiv_radius_um:.2f} µm")
    print(f"Dice: {best['dice']:.4f} | IoU: {best['iou']:.4f} | "
          f"Precision: {best['precision']:.4f} | Recall: {best['recall']:.4f}")

    return dict(
        sample=sample_name, cc_class=cc_class,
        rf_volume_mm3=rf_volume_mm3,
        reference_volume_mm3=reference_volume_mm3,
        relative_difference_pct=rel_diff_pct,
        rf_equiv_radius_um=rf_equiv_radius_um,
        dice=best["dice"], iou=best["iou"],
        precision=best["precision"], recall=best["recall"],
        n_evaluated_slices=len(evaluation_slices),
    )


if __name__ == "__main__":
    BASE = Path(DATA)

    all_results = []

    # S1
    all_results.append(evaluate_sample(
        sample_name="S1",
        prediction_path=BASE / "Salicornia 1, Inowroclaw 0 mM" / "image_folder" /
                        "Salicornia1_" / "MLresults" / "Salicornia1_fullstack_prediction.tif",
        reference_mask_path=BASE / "Salicornia 1, Inowroclaw 0 mM" / "S1_cc_labelmap_proper.nrrd",
        voxel_um=2.50002,
        rescale_factor=2,
        reference_volume_mm3=0.07353272957348922,
        z_start=15,
        z_end=984,
    ))

    # --- S4 (run this next) ---
    all_results.append(evaluate_sample(
        sample_name="S4",
        prediction_path=BASE / "Salicornia 4, Inowroclaw 1000 mM" / "image_folder" /
                        "Salicornia4_" / "MLresults" / "Salicornia4_fullstack_prediction.tif",
        reference_mask_path=BASE / "Salicornia 4, Inowroclaw 1000 mM" / "cc_lm_v5.nrrd",
        voxel_um=2.50001997,
        rescale_factor=4,
        reference_volume_mm3=0.7180,  # from your 3D texture+threshold method
    ))

      # --- S5 ---
    all_results.append(evaluate_sample(
        sample_name="S5",
        prediction_path=BASE / "Salicornia_5" / "image_folder" /
                        "Salicornia5_" / "MLresults" / "Salicornia5_fullstack_prediction.tif",
        reference_mask_path=BASE / "Salicornia_5" / "S5_cc_labelmap_bufferfix.nrrd",
        voxel_um=2.5000006400000005,
        rescale_factor=2,
        reference_volume_mm3=0.278187,  # radia max-gradient method, after buffer fix
        z_start=16,
        z_end=986,
    ))

    # S8
    all_results.append(evaluate_sample(
        sample_name="S8",
        prediction_path=BASE / "Salicornia_8" / "image_folder" /
                        "Salicornia8_" / "MLresults" /
                        "Salicornia8_fullstack_prediction.tif",
        reference_mask_path=BASE / "Salicornia_8" / "cc_labelmap.nrrd",
        voxel_um=2.50002788,
        rescale_factor=2,
        reference_volume_mm3=1.5131,
        z_start=0,
        z_end=983,
    ))

    # Print final summary table
    print("\n\n" + "="*70)
    print("FINAL SUMMARY TABLE")
    print("="*70)
    for r in all_results:
        print(r)