#!/bin/bash
# run_commands_reference.sh
# -----------------------------------------------------------------------------
# The order in which the scripts in this repository were run to produce the
# results reported in the manuscript.
#
# This file is documentation. It is meant to be read, and run one block at a
# time against your own data. It is not a turnkey pipeline: several steps were
# run once per sample with sample-specific settings, and the intermediate
# volumes are large. The settings that matter are stated in the comments and
# are also hard-coded at the top of each script.
#
# Before anything:
#
#   export SALICORNIA_DATA=/path/to/the/data
#
# Every script resolves its inputs relative to that variable. The imaging data
# are deposited in the BioImage Archive; see the data availability statement in
# the manuscript.
#
# Sample identifiers follow the manuscript:
#
#   ID       Population     NaCl     Scan        Slices   Criterion
#   -------------------------------------------------------------------------
#   I-0      Inowroclaw        0 mM  Normalscan    1000   half maximum
#   I-1000   Inowroclaw     1000 mM  Normalscan     800   texture threshold 3500
#   C-0      Ciechocinek       0 mM  Fastscan      1002   maximum gradient
#   C-1000   Ciechocinek    1000 mM  Fastscan       985   texture Otsu (2841.73)
#
# Internal sample folders use the original numbering: S1 = I-0, S4 = I-1000,
# S5 = C-0, S8 = C-1000. Voxel size is 2.5 um throughout.
# -----------------------------------------------------------------------------

set -eu
: "${SALICORNIA_DATA:?set SALICORNIA_DATA to the directory holding the volumes}"
cd "$SALICORNIA_DATA"


# =============================================================================
# 1. Format conversion and quality control
# =============================================================================
# The reconstructions leave VGSTUDIO MAX as .vol / .vgl with a detached header.

python3 preprocessing/vgl_to_nrrd.py
python3 preprocessing/vol_to_tiff.py
python3 preprocessing/npy_to_nrrd.py          # for volumes handled as .npy

# Confirm that the voxel spacing survived every conversion. Spacing errors are
# silent and scale the reported volumes by a constant factor.
python3 validation/check_spacing.py
python3 validation/volume_quality_check.py


# =============================================================================
# 2. Denoising (I-0 only)
# =============================================================================
# Only the Normalscan 0 mM volume was denoised. The remaining volumes were
# processed from the reconstructions as they came out of VGSTUDIO MAX.

python3 preprocessing/nlm_denoise_reestimate.py   # non-local means, sigma re-estimated
python3 preprocessing/crop_denoise.py             # crop to the denoised extent
python3 preprocessing/median_filter.py            # used in the comparison in Fig. 2

# Single slices used for the figures and for the radial profiles.
python3 preprocessing/extract_slice.py
python3 preprocessing/extract_slice_I1000.py


# =============================================================================
# 3. Stem centre detection
# =============================================================================
# The radial profile is only meaningful about a correctly placed centre.

python3 centre_detection/stem_center_detection.py
python3 centre_detection/ring_center_fit.py        # vascular-ring fit, used for C-0
python3 centre_detection/z_projection_center_C0.py # centre from the z projection
python3 centre_detection/center_drift_check.py     # centre drift along z


# =============================================================================
# 4. Radial profiling
# =============================================================================
python3 radial_profiling/directional_radial_profile.py   # 360 rays
python3 radial_profiling/radial_boundary_estimation.py
python3 radial_profiling/radial_texture_profile.py
python3 radial_profiling/raw_vs_denoised_profile.py      # effect of denoising

# Per-sample boundary estimation with the sample-specific settings.
python3 radial_profiling/per_sample/radial_boundaries_I0.py
python3 radial_profiling/per_sample/radial_boundaries_I1000.py
python3 radial_profiling/per_sample/radial_boundaries_C0.py
python3 radial_profiling/per_sample/radial_boundaries_C1000.py


# =============================================================================
# 5. Texture volumes
# =============================================================================
# Local standard deviation in a sliding window. Streaming versions process the
# stack slice by slice and are the ones used for the full volumes; the full
# version keeps the stack in memory and was used for checking.
# Window: 5 x 5 for I-0, 9 x 5 for C-0.

python3 texture/compute_texture_volume_streaming_I0.py
python3 texture/compute_texture_volume_streaming_I1000.py
python3 texture/compute_texture_volume_streaming_C0.py
python3 texture/compute_texture_volume_streaming_C1000.py
python3 texture/compute_texture_volume_full_C1000.py      # verification

# Single-slice tests used to choose the window size.
python3 texture/slice_tests/compute_texture_slice_test_I0.py
python3 texture/slice_tests/compute_texture_slice_test_I1000.py
python3 texture/slice_tests/compute_texture_slice_test_C0.py
python3 texture/slice_tests/compute_texture_slice_test_C1000.py


# =============================================================================
# 6. Volumetric quantification: one criterion per sample
# =============================================================================
# This is the sample-adaptive step. The criterion differs between samples
# because the boundary contrast does; see Materials and methods.

# --- I-0: half maximum of the radial intensity profile
python3 volumetric/I0_half_maximum.py

# --- C-0: maximum gradient of the radial profile
python3 volumetric/C0_maximum_gradient.py

# --- I-1000 and C-1000: threshold on the texture volume.
# Both were originally thresholded interactively in the 3D Slicer Segment
# Editor. texture_threshold_volume.py reproduces that step non-interactively
# so the published volumes can be regenerated from the archived texture
# volumes. Per slice: threshold -> binary closing (radius 2) -> largest
# connected component -> fill holes.

python3 volumetric/texture_threshold_volume.py \
    --texture-nhdr Salicornia_4_texture.nhdr \
    --threshold 3500 \
    --label-out S4_cc_labelmap.npy

python3 volumetric/texture_threshold_volume.py \
    --texture-nhdr Salicornia_8_texture.nhdr \
    --threshold otsu            # Otsu returns 2841.73 on this volume

# Voxel counting from a finished label map.
python3 volumetric/calculate_size.py


# =============================================================================
# 7. Random forest segmentation
# =============================================================================
# Training and prediction are run with the third-party leaf-traits-microct
# pipeline, which is not redistributed here. It does not run unmodified on a
# current Python stack: see compatibility_patches.md for the required edits.
# The scripts below prepare its inputs and interpret its outputs.

# Grid and phase stacks in the layout the pipeline expects.
python3 preprocessing/prepare_grid_phase.py
python3 random_forest/prepare_I0.py
python3 random_forest/prepare_I1000.py
python3 random_forest/prepare_C0.py
python3 random_forest/prepare_C1000.py

# Training label stacks, derived from the final reference masks.
python3 random_forest/build_label_stack.py
python3 random_forest/build_label_stack_12.py          # 12-slice variant, I-0
python3 random_forest/build_rf_labels_from_final_masks.py
python3 random_forest/check_labels.py                  # label values and counts

# --- run the third-party pipeline here ---
#     python3 Leaf_Segmentation_py3.py <argfile>
#     Delete *_2x-smaller.tif, *_4x-smaller.tif, *_RF_model.joblib,
#     *GridPhase_invert_ds.tif and *local_thick.tif first: the pipeline reuses
#     them silently and will otherwise reproduce the previous run.

# The pipeline does not guarantee which output class is the central cylinder.
python3 random_forest/identify_prediction_classes.py
python3 random_forest/identify_classes_fixed.py
python3 random_forest/analyze_rf_prediction_I0.py


# =============================================================================
# 8. Validation
# =============================================================================
# Dice, IoU, precision and recall on the evaluated slices (Table 3).
python3 validation/evaluation_slices.py

python3 validation/verify_mask.py
python3 validation/boundary_check.py
python3 validation/check_circularity.py       # solidity and circularity per slice
python3 validation/check_alignment.py

# Volumes reported in Table 2, recomputed from the deposited masks.
python3 validation/check_mask_volume.py --expect 0.0735 I-0_cc_reference_mask.nrrd
python3 validation/check_mask_volume.py --expect 0.7180 I-1000_cc_reference_mask.nrrd
python3 validation/check_mask_volume.py --expect 0.2782 C-0_cc_reference_mask.nrrd
python3 validation/check_mask_volume.py --expect 1.5131 C-1000_cc_reference_mask.nrrd


# =============================================================================
# 9. Derived measurements
# =============================================================================
# Cortical parenchyma and peripheral tissue areas and ratios (Table 5), derived
# from the manual diameter measurements and the segmented volumes. Prints the
# table and writes pc_pt_derived_table.csv.
python3 measurements/calculate_pc_pt_areas.py


# =============================================================================
# 10. Figures
# =============================================================================
python3 figures/generate_figure2_workflow.py     # Fig. 2, workflow panels
python3 figures/generate_panel_d.py              # Fig. 2D, boundary criteria
python3 figures/generate_panel_e.py              # Fig. 2E
python3 figures/generate_figure3.py              # Fig. 3
python3 figures/generate_figure4.py              # Fig. 4
python3 figures/generate_figure5.py              # Fig. 5
python3 figures/generate_rf_comparison.py        # Fig. 6, RF vs reference
python3 figures/generate_figureS1.py             # Fig. S1

# Fig. 5c, the 3D rendering, is produced inside 3D Slicer:
#     exec(open('figures/slicer_render_C0.py').read())


# =============================================================================
# 11. Preparation of the archive deposition
# =============================================================================
python3 deposition/normalize_mask.py       # uint8, single label, correct spacing
python3 deposition/prepare_bia_upload.py   # directory tree and file lists


# =============================================================================
# Not part of the reported analysis
# =============================================================================
# exploratory/ holds the diagnostic scripts written while the boundary criteria
# were being chosen: threshold sweeps, Otsu diagnostics, vessel-ring detection,
# denoising-strength tests. They are kept because they document how the
# criteria in section 6 were arrived at, and because several of them record
# approaches that did not work. They are not needed to reproduce the results.
