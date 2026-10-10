#!/bin/bash
# run_commands_reference.sh
# The exact command-line invocations used to produce the reported volumes.
# This file is documentation: it is meant to be read, and run one block at a
# time against your own data, not executed end to end.
#
#   export SALICORNIA_DATA=/path/to/data
#
# Sample identifiers follow the manuscript:
#   I-0     Inowroclaw, 0 mM        1000 slices   half maximum
#   I-1000  Inowroclaw, 1000 mM      800 slices   texture threshold 3500
#   C-0     Ciechocinek, 0 mM       1002 slices   maximum gradient
#   C-1000  Ciechocinek, 1000 mM     985 slices   texture Otsu 2841.73

set -eu
: "${SALICORNIA_DATA:?set SALICORNIA_DATA to the directory holding the volumes}"
cd "$SALICORNIA_DATA"

# ----------------------------------------------------------------------
# 1. Format conversion and quality control
# ----------------------------------------------------------------------
python3 preprocessing/vgl_to_nrrd.py        # <ARGS — FILL IN>
python3 preprocessing/vol_to_tiff.py        # <ARGS>
python3 validation/volume_quality_check.py  # <ARGS>

# ----------------------------------------------------------------------
# 2. Denoising (I-0 only; the Fastscan samples were not denoised)
# ----------------------------------------------------------------------
python3 preprocessing/nlm_denoise_reestimate.py   # <ARGS>
python3 preprocessing/crop_denoise.py             # <ARGS>

# ----------------------------------------------------------------------
# 3. Stem centre detection
# ----------------------------------------------------------------------
python3 centre_detection/stem_center_detection.py   # <ARGS, per sample>

# ----------------------------------------------------------------------
# 4. Volumetric quantification, one criterion per sample
# ----------------------------------------------------------------------
# I-0: half maximum of the radial profile
python3 volumetric/I0_half_maximum.py       # <ARGS>

# I-1000: fixed texture threshold 3500, followed by closing with radius 2
python3 texture/compute_texture_volume_streaming_I1000.py   # <ARGS>
python3 volumetric/I1000_texture_threshold.py               # <ARGS>

# C-0: maximum gradient
python3 volumetric/C0_maximum_gradient.py   # <ARGS>

# C-1000: Otsu on the texture volume, threshold 2841.73
python3 texture/compute_texture_volume_streaming_C1000.py   # <ARGS>
# <SCRIPT THAT APPLIED THE OTSU THRESHOLD — FILL IN>

# ----------------------------------------------------------------------
# 5. Random forest: inputs, training, evaluation
# ----------------------------------------------------------------------
python3 preprocessing/prepare_grid_phase.py              # <ARGS>
python3 random_forest/build_rf_labels_from_final_masks.py # <ARGS>
# training is run from the third-party pipeline; see compatibility_patches.md
python3 random_forest/identify_prediction_classes.py      # <ARGS>
python3 validation/evaluation_slices.py                   # <ARGS>

# ----------------------------------------------------------------------
# 6. Verification against the reported numbers
# ----------------------------------------------------------------------
python3 validation/check_mask_volume.py --expect 0.0735 I-0_cc_reference_mask.nrrd
python3 validation/check_mask_volume.py --expect 0.7180 I-1000_cc_reference_mask.nrrd
python3 validation/check_mask_volume.py --expect 0.2782 C-0_cc_reference_mask.nrrd
python3 validation/check_mask_volume.py --expect 1.5131 C-1000_cc_reference_mask.nrrd

# ----------------------------------------------------------------------
# 7. Figures
# ----------------------------------------------------------------------
python3 figures/generate_panel_d.py          # <ARGS>
python3 figures/generate_rf_comparison.py    # <ARGS>
