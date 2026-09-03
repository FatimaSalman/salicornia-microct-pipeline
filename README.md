# salicornia-microct-pipeline

Python scripts for converting VGStudio microCT files to an open, vendor-independent format (NRRD), assessing reconstructed-volume image quality (contrast-to-noise ratio, CNR), estimating anatomical tissue boundaries from radial texture profiles, denoising low-contrast volumes, and preparing training labels for Random Forest–based automated segmentation of *Salicornia europaea* stem tissue. Developed as part of a pilot study on microCT-based structural phenotyping of *Salicornia europaea* stems.

**Associated manuscript:** *A Pipeline for microCT-Based Structural Phenotyping and Preliminary Deep Learning Assessment in Salicornia: A Methodological Pilot Study* (Salman, Kováč, Ďurkovič — submitted to *The Plant Journal*).

## Contents

| File | Purpose |
|---|---|
| `vgl_to_nrrd.py` | Converts a VGStudio project file (`.vgl`) into a detached NRRD header (`.nhdr`) that points to the original `.vol` data, so the volume can be opened directly in 3D Slicer (or any NRRD-aware tool) without needing VGStudio or a Windows VM. |
| `volume_quality_check.py` | Computes objective image-quality metrics (Otsu-based contrast-to-noise ratio, intensity statistics, edge sharpness) from a raw volume. |
| `radial_boundary_estimation.py` | Computes a windowed local-intensity-texture radial profile centered on the stem, to identify candidate tissue-boundary radii (e.g., central cylinder/parenchyma cortex transition). |
| `nlm_denoise_crop.py` | Applies non-local-means denoising to a cropped region centered on the stem for low-contrast volumes where the raw radial-texture profile is confounded by heteroscedastic noise. |
| `build_label_stack.py` | Generates a multi-slice label stack (central cylinder+pith, parenchyma cortex, palisade tissue) from a fixed stem center and boundary radii, for use as Random Forest training input. |
| `class_identification.py` | Confirms which predicted class value in a Random Forest output stack corresponds to which tissue class, and computes tissue volumes from voxel counts. |
| `run_commands_reference.sh` | Reference log of the exact command-line invocations (training slice indices, rescale factors) used for each of the four samples. |

## Third-party dependency: leaf-traits-microct

We performed Random Forest training and full-stack prediction using the **leaf-traits-microct** pipeline (Théroux-Rancourt et al. 2020; https://github.com/gtrancourt/leaf-traits-microct), without modifying its core algorithm. Running it under a modern Python environment (3.14) required several compatibility fixes, documented in [`compatibility_patches.md`](compatibility_patches.md), including updated imports (`sklearn.externals` → `joblib`), NumPy/SciPy API renames, and an `img_as_ubyte` safe-casting workaround. These patches are provided as a diff against the original tool for transparency and reproducibility; please cite the original tool's authors when using it.

## Requirements

- Python 3.8+ (compatibility notes above apply to 3.14 specifically)
- `numpy`, `scipy`, `scikit-image`, `tifffile`
- `pip install numpy scipy scikit-image tifffile --break-system-packages` (or use a virtual environment)

## Usage

### 1. Convert a `.vgl` project to an open format
```bash
python3 vgl_to_nrrd.py "/path/to/Salicornia_5/Salicornia_5.vgl"
```

### 2. Check reconstruction quality
```bash
python3 volume_quality_check.py "/path/to/Salicornia_5.nhdr"
```

### 3. Estimate anatomical boundaries
python3 radial_boundary_estimation.py --stack Salicornia_1_stack.tif --slice 500 \
    --voxel-um 2.5 --center-y 520 --center-x 591 --candidate-radii 90 190 250

### 4. (If needed) Denoise a low-contrast volume
python3 nlm_denoise_crop.py --stack Salicornia_1_stack.tif --slice 500 \
    --center-y 520 --center-x 591 --half-width 400 --voxel-um 2.5 --known-radius-um 87.5

### 5. Build training labels
Edit the center/radii/slice-list constants at the top of `build_label_stack.py`
for the sample you're processing (see run_commands_reference.sh for the values
used for each of the four samples), then run:
    python3 build_label_stack.py

### 6. Train and predict (leaf-traits-microct, see reference commands)
See `run_commands_reference.sh` for the exact per-sample invocations used in the manuscript.

### 7. Identify predicted classes and compute volumes
Edit the `samples` dictionary at the top of `class_identification.py` with your
prediction file paths, then run:
    python3 class_identification.py

## Data

Raw microCT volumes, trained Random Forest models, manually annotated training label stacks, or full-stack prediction volumes are not included in this repository due to file size (>100 MB per file). The corresponding author will provide them upon reasonable request.

## License

Released under the [MIT License](LICENSE). Note: the third-party `leaf-traits-microct` tool referenced above has its own license; consult the original repository before redistributing.

## Citation

If you use this pipeline, please cite the associated manuscript (details above; full citation to be updated upon publication).
