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
                        file timestamps correspond to the last upstream commit of 6 Nov 2019 (commit <HASH>),  
                        https://github.com/plant-microct-tools/leaf-traits-microct,
#                      run with the modifications listed in compatibility_patches.md
# Fiji:  Fiji app 2.0.2  (ImageJ 1.54p, Java 17.0.20 64-bit, installed 8 April 2025) used for 16-bit -> 8-bit conversion; pixel measurement (Figure 2b)
# 3D Slicer:    5.6.2     (visual inspection and manual annotation)

#!/usr/bin/env bash
# run_commands_reference.sh
# ==========================
# Reference log of the exact parameters used for each of the four Salicornia
# samples in the associated manuscript. The scripts in this repository take
# these values either as command-line flags (radial_boundary_estimation.py,
# nlm_denoise_crop.py) or as constants edited at the top of the file
# (build_label_stack_12.py, class_identification.py) — see comments below.
#
# This file is NOT meant to be executed as-is; it documents the exact
# values used, for reproducibility.

# ------------------------------------------------------------------
# Sample centers, voxel sizes, and known boundary radii (all in um)
# ------------------------------------------------------------------
# S1: center Y=520, X=591 | voxel=2.5           | cc=90 (revised from 87.5), pc/pt=190, pt/out=250
# S4: center Y=795, X=1051| voxel=2.50001997    | cc=279, pc/pt=470, pt/out=1050
# S5: center Y=325, X=320 | voxel=2.50000064    | cc=90 , pc/pt=200, pt/out=300
# S8: center Y=539, X=441 | voxel=2.50002788    | cc=215, pc/pt=470, pt/out=590

# ------------------------------------------------------------------
# 1. Radial boundary estimation (radial_boundary_estimation.py)
# ------------------------------------------------------------------
python3 radial_boundary_estimation.py --stack Salicornia_1_stack.tif --slice 500 \
    --voxel-um 2.5 --center-y 520 --center-x 591 --candidate-radii 90 190 250

python3 radial_boundary_estimation.py --stack Salicornia_4_stack.tif --slice 400 \
    --voxel-um 2.50001997 --center-y 795 --center-x 1051 --candidate-radii 279 470 1050

python3 radial_boundary_estimation.py --stack Salicornia_5_stack.tif --slice 501 \
    --voxel-um 2.50000064 --center-y 325 --center-x 320 --max-radius-um 500 \ --candidate-radii 90 200 300

python3 radial_boundary_estimation.py --stack Salicornia_8_stack.tif --slice 150 \
    --voxel-um 2.50002788 --candidate-radii 215 470 590
    # (S8 center was auto-estimated via the solid-shape method; omit --center-y/-x
    #  to reproduce, or pass the value printed by the script if you need it fixed.)

# ------------------------------------------------------------------
# 2. Non-local-means denoising (nlm_denoise_crop.py) — Salicornia 1 only
# ------------------------------------------------------------------
# Applied only to S1 (lowest-contrast volume). S4 and S5 were also tested
# with the same procedure as a control (see manuscript Methods); in both
# cases the recomputed profile confirmed the original boundaries without
# requiring revision, so their labels were built directly from the raw
# geometric radii above, without denoising.
python3 nlm_denoise_crop.py --stack Salicornia_1_stack.tif --slice 500 \
    --center-y 520 --center-x 591 --half-width 400 --voxel-um 2.5 \
    --known-radius-um 87.5 --patch-size 7 --patch-distance 9 --h-factor 1.6

# ------------------------------------------------------------------
# 3. Training label generation (build_label_stack_12.py / analogous per-sample scripts)
# ------------------------------------------------------------------
# S1 (this repository's build_label_stack_12.py, values already set):
#   center=(520,591), radii=(90,190,250), voxel=2.5
#   12 reference slices: 50,130,220,310,400,500,600,700,780,850,920,970
#
# S4, S5, S8 (6 reference slices each — edit the equivalent constants
# at the top of your copy of the script before running):
#   S4: center=(795,1051), radii=(279,470,1050), voxel=2.50001997
#       reference_slices = [80, 200, 320, 480, 600, 720]
#   S5: center=(325,320),  radii=(90, 200,300)      , voxel=2.5000006400000005
#       reference_slices = [80, 200, 400, 600, 800, 950]
#   S8: center=(539,441),  radii=(215,470,590) , voxel=2.50002788
#       reference_slices = [150, 300, 450, 550, 700, 850]

# ------------------------------------------------------------------
# 4. Random Forest training/prediction (leaf-traits-microct, third-party tool)
# ------------------------------------------------------------------
python3 Leaf_Segmentation_py3.py sample_name=Salicornia1_ \
    phase_filename=PHASE-8bit.tif threshold_phase=35 \
    grid_filename=GRID-8bit.tif threshold_grid=35 \
    slice_numbers_training_slices=51,131,221,311,401,501,601,701,781,851,921,971 \
    nb_training_slices=12 \
    "path_to_image_folder=/path/to/Salicornia_1/image_folder/" \
    rescale_factor=2

# S4, S5, S8 used 6 training slices each and rescale_factor as noted:
#   S4: rescale_factor=4 (larger raw volume dimensions)
#   S5: slice_numbers_training_slices=81,201,401,601,801,951 nb_trainig_slices=6 rescale_factor=2
#   S8: rescale_factor=2

# ------------------------------------------------------------------
# 5. Class identification and volume computation (class_identification.py)
# ------------------------------------------------------------------
# Edit the `samples` dict at the top of class_identification.py with your
# prediction file paths; center/radius/rescale_factor/voxel values are
# already filled in for all four samples as listed above, then run:
python3 class_identification.py
