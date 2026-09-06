#!/usr/bin/env bash
# run_commands_reference.sh
# ==========================
# Reference log of the exact parameters used for each of the four Salicornia
# europaea stem volumes analysed in the associated manuscript:
#
#   Salman F., Kováč J., Ďurkovič J. — "MicroCT-Based Structural Phenotyping
#   and Random Forest Segmentation of the Central Cylinder in Salicornia
#   europaea Stems: A Methodological Pilot Study" (The Plant Journal, submitted)
#
# The scripts in this repository take these values either as command-line
# flags (radial_boundary_estimation.py, nlm_denoise_crop.py) or as constants
# edited at the top of the file (build_label_stack.py, class_identification.py).
#
# This file is NOT meant to be executed as-is; it documents the exact values
# used, for reproducibility. The original per-sample scripts, run as-is to
# produce the manuscript results, are archived unmodified in the Zenodo data
# record (per_sample_scripts/).

# ------------------------------------------------------------------
# Software and hardware environment used for all analyses
# ------------------------------------------------------------------
# Hardware:     MacBook Air (13-inch, 2017), 1.8 GHz dual-core Intel Core i5,
#               8 GB 1600 MHz DDR3 RAM, Intel HD Graphics 6000 (no GPU acceleration)
# OS:           macOS Monterey 12.7.6 (21H1320)
# Python        3.14.4  (/Library/Frameworks/Python.framework/Versions/3.14)
# numpy         2.5.0
# scipy         1.18.0
# scikit-image  0.26.0
# scikit-learn  1.9.0
# tifffile      2026.6.1
# PyWavelets    1.8.0   (present -> nlm_denoise_crop.py used skimage's
#                        wavelet-based estimate_sigma, not the MAD fallback)
# leaf-traits-microct: master branch snapshot downloaded as ZIP from GitHub;
#                      file timestamps correspond to the last upstream commit of
#                      6 Nov 2019 (commit 82c09df971621a6bf6e5d51060f1dc31cfae4700),
#                      https://github.com/plant-microct-tools/leaf-traits-microct,
#                      run with the modifications listed in compatibility_patches.md
# Fiji:         Fiji app 2.0.2 (ImageJ 1.54p, Java 17.0.20 64-bit, installed 8 April 2025)
#               used for 16-bit -> 8-bit conversion and pixel measurement (Figure 2b)
# 3D Slicer:    5.6.2     (visual inspection and manual anatomical annotation)
#
# Exact Python package versions are pinned in requirements.txt.

# ------------------------------------------------------------------
# Sample overview: acquisition, centers, voxel sizes, boundary radii (um)
# ------------------------------------------------------------------
# Sample | Site        | NaCl    | Scan       | Stack (Z,Y,X)      | Depth (mm) | Center (Y,X) | Voxel (um)  | cc/pc | pc/pt | pt/out
# S1     | Inowroclaw  | 0 mM    | Normalscan | 1000 x  900 x  900 | 2.50       | 520, 591     | 2.5         | 90*   | 190   | 250
# S4     | Inowroclaw  | 1000 mM | Normalscan |  800 x 1590 x 1590 | 2.00       | 795, 1051    | 2.50001997  | 279   | 470   | 1050
# S5     | Ciechocinek | 0 mM    | Fastscan   | 1002 x  803 x  660 | 2.51       | 325, 320     | 2.50000064  | 90    | 200   | 300
# S8     | Ciechocinek | 1000 mM | Fastscan   |  985 x 1079 x  884 | 2.46       | 539, 441     | 2.50002788  | 215   | 470   | 590
#
# * S1 cc/pc radius: 87.5 um from the original (un-denoised) radial profile,
#   revised to 90 um after NLM denoising (see section 2). Table 2 of the
#   manuscript (geometric volume) reports 87.5 um; training labels (section 3)
#   used 90 um.
# Voxel sizes are as stored in the .vgl headers; the per-sample scripts used
# 2.5 um for S1 and S5 (difference < 1e-5 um, negligible).
# Centers were fixed per sample in full-frame voxel coordinates and used for
# all subsequent radial analyses.

# ------------------------------------------------------------------
# 0. Format conversion, image-quality check, 8-bit conversion
# ------------------------------------------------------------------
# 0a. .vgl -> detached NRRD header (no data duplication); verified in 3D Slicer 5.6.2
python3 vgl_to_nrrd.py "Salicornia 1 scan 2.vgl"      # -> "Salicornia 1 scan 2.nhdr"
python3 vgl_to_nrrd.py "Salicornia 4 normalscan.vgl"      # -> "Salicornia 4 normalscan.nhdr"
python3 vgl_to_nrrd.py "Salicornia_5.vgl"      # -> "Salicornia_5.nhdr"
python3 vgl_to_nrrd.py "Salicornia_8.vgl"      # -> "Salicornia_8.nhdr"

# 0b. .vol + .nhdr -> multi-page 16-bit TIFF stack
python3 vol_to_tiff.py "Salicornia 1 scan 2.nhdr" Salicornia_1_stack.tif
python3 vol_to_tiff.py "Salicornia 4 normalscan.nhdr" Salicornia_4_stack.tif
python3 vol_to_tiff.py "Salicornia_5.nhdr" Salicornia_5_stack.tif
python3 vol_to_tiff.py "Salicornia_8.nhdr" Salicornia_8_stack.tif

# 0c. Contrast-to-noise ratio on 5 equally spaced mid-volume slices (20th-80th
#     percentile of stack depth), Otsu split, CNR = |mu_high - mu_low| / min(sigma)
python3 volume_quality_check.py "Salicornia 1 scan 2.nhdr"    # CNR = 3.52
# S4: CNR = 3.43 | S5: CNR = 3.78 | S8: CNR = 3.19  (manuscript Table 1)

# 0d. 16-bit -> 8-bit conversion (prepare_grid_phase.py): linear contrast
# stretch, clipping to [12000, 50000] then rescaling to [0 , 255],
# processed slice-by-slice (memory-safe for large stacks).
python3 prepare_grid_phase.py --input Salicornia_1_stack.tif \
    --sample-name Salicornia1_ --low 12000 --high 50000

python3 prepare_grid_phase.py --input Salicornia_4_stack.tif \
    --sample-name Salicornia4_ --low 12000 --high 50000

python3 prepare_grid_phase.py --input Salicornia_5_stack.tif \
    --sample-name Salicornia5_ --low 12000 --high 50000

python3 prepare_grid_phase.py --input Salicornia_8_stack.tif \
    --sample-name Salicornia8_ --low 12000 --high 50000
    
#     Output: image_folder/Salicornia1_/Salicornia1_GRID-8bit.tif
#     PHASE-8bit.tif is symlink of GRID-8bit.tif: the single-channel
#     data were supplied to both inputs of the dual-channel
#     leaf-traits-microct pipeline (see manuscript Methods).

# ------------------------------------------------------------------
# 1. Radial boundary estimation (radial_boundary_estimation.py)
# ------------------------------------------------------------------
# Local intensity standard deviation in a 9 x 9-pixel sliding window, averaged
# over concentric annuli (4-voxel steps) centered on the stem center.
# Candidate radii were read from local extrema of the profile and confirmed
# visually on the boundary-check image. For S1 and S5 (smallest central
# cylinders) visual confirmation was weaker and the estimate relies primarily
# on the quantitative profile.

python3 radial_boundary_estimation.py --stack Salicornia_1_stack.tif --slice 500 \
    --voxel-um 2.5 --center-y 520 --center-x 591 --candidate-radii 90 190 250

python3 radial_boundary_estimation.py --stack Salicornia_4_stack.tif --slice 400 \
    --voxel-um 2.50001997 --center-y 795 --center-x 1051 --candidate-radii 279 470 1050
    # The default profile range (--max-radius-um 700) does not reach the
    # pt/background boundary (1050 um); that boundary was confirmed on the
    # boundary-check image overlay rather than on the profile plot.
    # The 279 um cc/pc radius was additionally confirmed by direct pixel
    # measurement in Fiji on the manually annotated cross-section (Figure 2b).

python3 radial_boundary_estimation.py --stack Salicornia_5_stack.tif --slice 501 \
    --voxel-um 2.50000064 --center-y 325 --center-x 320 --max-radius-um 500 \
    --candidate-radii 90 200 300

python3 radial_boundary_estimation.py --stack Salicornia_8_stack.tif --slice 150 \
    --voxel-um 2.50002788 --candidate-radii 215 470 590
    # S8 center was auto-estimated by the script's solid-shape method
    # (threshold on local-texture map at its 90th percentile, dilate, fill
    # holes, erode, centroid of largest component): Y=539, X=441.
    # This value was then used as the fixed center in all subsequent steps.

# ------------------------------------------------------------------
# 2. Non-local-means denoising (nlm_denoise_crop.py) - Salicornia 1 only
# ------------------------------------------------------------------
# Applied to S1 (lowest-contrast volume), whose radial profile was confounded
# by radially increasing noise amplitude and the star-shaped vascular geometry.
# Crop: 800 x 800 pixels centered on the stem (half-width 400).
# NLM: patch_size=7, patch_distance=9, h = 1.6 * sigma, fast_mode=False,
# sigma from skimage.restoration.estimate_sigma (wavelet-based; PyWavelets 1.8.0).
python3 nlm_denoise_crop.py --stack Salicornia_1_stack.tif --slice 500 \
    --center-y 520 --center-x 591 --half-width 400 --voxel-um 2.5 \
    --known-radius-um 87.5 --patch-size 7 --patch-distance 9 --h-factor 1.6
# Re-estimated boundaries on the denoised slice: cc/pc 87.5 -> 90 um,
# pc/pt 190 um, pt/background 250 um.
#
# Consistency check: the same crop + denoise + profile procedure applied to a
# representative slice of S4 (slice 400) and S5 (slice 501) reproduced the
# original radii (279 and 90 um) without change (manuscript Figure S1).
# S4 and S5 labels were therefore built from the raw geometric radii above.

# ------------------------------------------------------------------
# 3. Training label generation (build_label_stack.py)
# ------------------------------------------------------------------
# Ternary label stacks generated programmatically (not by manual tracing):
# each pixel assigned by Euclidean distance from the fixed stem center.
# Label values: 153 = cc + pi, 102 = pc, 51 = pt, 0 = background
# (leaf-traits-microct remaps these internally).
# Slice indices below are 0-based (tifffile page index).
#
# S1 (values currently set in build_label_stack.py):
#   center=(520,591), radii=(90,190,250), voxel=2.5
#   reference_slices = [50, 130, 220, 310, 400, 500, 600, 700, 780, 850, 920, 970]  # 12 slices
#   NOTE: a first S1 attempt used 6 slices with the original (un-denoised)
#   radii (87.5,190,250) and yielded a cc volume more than twice the geometric
#   estimate; it was superseded by the 12-slice, denoised-radii run above.
#
# S4, S5, S8 (6 reference slices each, approximately evenly spaced; edit the
# constants at the top of build_label_stack.py accordingly):
#   S4: center=(795,1051), radii=(279,470,1050), voxel=2.50001997
#       reference_slices = [80, 200, 320, 480, 600, 720]
#   S5: center=(325,320),  radii=(90,200,300),   voxel=2.50000064
#       reference_slices = [80, 200, 400, 600, 800, 950]
#   S8: center=(539,441),  radii=(215,470,590),  voxel=2.50002788
#       reference_slices = [150, 300, 450, 550, 700, 850]
#
# Output per sample: image_folder/Salicornia<N>_/labelled-stack.tif
python3 build_label_stack.py

# ------------------------------------------------------------------
# 4. Random Forest training / prediction (leaf-traits-microct, third-party)
# ------------------------------------------------------------------
# Leaf_Segmentation_py3.py is part of the leaf-traits-microct repository,
# run with the compatibility fixes listed in compatibility_patches.md.
# Feature layers: local intensity, Gaussian, Hessian, local thickness
# (37 layers, as in Théroux-Rancourt et al. 2020).
# NOTE: leaf-traits-microct uses 1-based (ImageJ-style) slice numbering;
# slice_numbers_training_slices below = reference_slices (section 3) + 1.
# Volumes were spatially downsampled (rescale_factor) to keep computation
# tractable on the 8 GB-RAM laptop described above.

# S1 - 12 training slices, rescale 2  (OOB accuracy 99.93 %)
python3 Leaf_Segmentation_py3.py sample_name=Salicornia1_ \
    phase_filename=PHASE-8bit.tif threshold_phase=35 \
    grid_filename=GRID-8bit.tif threshold_grid=35 \
    slice_numbers_training_slices=51,131,221,311,401,501,601,701,781,851,921,971 \
    nb_training_slices=12 \
    "path_to_image_folder=/path/to/Salicornia_1/image_folder/" \
    rescale_factor=2

# S4 - 6 training slices, rescale 4 (larger raw volume dimensions)  (OOB 99.85 %)
python3 Leaf_Segmentation_py3.py sample_name=Salicornia4_ \
    phase_filename=PHASE-8bit.tif threshold_phase=35 \
    grid_filename=GRID-8bit.tif threshold_grid=35 \
    slice_numbers_training_slices=81,201,321,481,601,721 \
    nb_training_slices=6 \
    "path_to_image_folder=/path/to/Salicornia_4/image_folder/" \
    rescale_factor=4

# S5 - 6 training slices, rescale 2  (OOB 99.88 %)
python3 Leaf_Segmentation_py3.py sample_name=Salicornia5_ \
    phase_filename=PHASE-8bit.tif threshold_phase=35 \
    grid_filename=GRID-8bit.tif threshold_grid=35 \
    slice_numbers_training_slices=81,201,401,601,801,951 \
    nb_training_slices=6 \
    "path_to_image_folder=/path/to/Salicornia_5/image_folder/" \
    rescale_factor=2

# S8 - 6 training slices, rescale 2  (OOB 99.89 %)
python3 Leaf_Segmentation_py3.py sample_name=Salicornia8_ \
    phase_filename=PHASE-8bit.tif threshold_phase=35 \
    grid_filename=GRID-8bit.tif threshold_grid=35 \
    slice_numbers_training_slices=151,301,451,551,701,851 \
    nb_training_slices=6 \
    "path_to_image_folder=/path/to/Salicornia_8/image_folder/" \
    rescale_factor=2

# Outputs per sample (image_folder/Salicornia<N>_/MLresults/):
#   Salicornia<N>_RF_model.joblib, Salicornia<N>_fullstack_prediction.tif
# Prediction stacks are signed 8-bit: label values above 127 wrap to negative
# values; output value -1 corresponds to cc + pi in all four samples,
# value 127 marks unresolved boundary pixels (0.09-0.5 % of voxels).

# ------------------------------------------------------------------
# 5. Class identification (class_identification.py)
# ------------------------------------------------------------------
# Overlays the known cc radius (converted to the downsampled voxel scale) on a
# representative predicted slice to confirm which output value falls inside it.
# Edit the `samples` dict at the top of the script with your prediction paths;
# center / radius / rescale_factor / voxel values are already filled in:
#   S1: z=500, rescale 2 | S4: z=400, rescale 4 | S5: z=501, rescale 2 | S8: z=492, rescale 2
python3 class_identification.py
# -> <sample>_prediction_class_identification.png

# ------------------------------------------------------------------
# 6. Volume computation (manuscript Tables 2 and 3)
# ------------------------------------------------------------------
# Geometric volume (Table 2): cylindrical mask of the estimated cc radius,
# centered on the stem center, spanning the full stack depth:
#   V = pi * r_cc^2 * (n_slices * voxel)   [voxel count x voxel volume]
#   S1: r=87.5 um -> 0.0601 mm3 | S4: r=279 um -> 0.4891 mm3
#   S5: r=90 um   -> 0.0637 mm3 | S8: r=215 um -> 0.3577 mm3
#   (logged in cc_volumetric_log_S<N>.txt, archived on Zenodo)
#
# Random Forest volume (Table 3): count of voxels with output value -1 in the
# full-stack prediction x rescale-adjusted voxel volume (voxel * rescale)^3:
#   S1: 0.0482 mm3 | S4: 0.4157 mm3 | S5: 0.0512 mm3 | S8: 0.3595 mm3
#
# Both computations were performed with the per-sample scripts
# (calculate_S<N>.py, size.py) archived unmodified in the Zenodo data record
# (per_sample_scripts/).
