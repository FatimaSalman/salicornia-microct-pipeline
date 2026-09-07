# Salicornia_Project — microCT structural phenotyping pipeline for *Salicornia europaea* stems

Python scripts for converting VGSTUDIO microCT project files to an open,
vendor-independent format (NRRD), assessing reconstructed-volume image quality
(contrast-to-noise ratio, CNR), estimating anatomical tissue boundaries from
radial texture profiles, denoising low-contrast volumes, generating training
labels, and identifying predicted classes for Random Forest–based segmentation
of the central cylinder in *Salicornia europaea* stems.

**Associated manuscript:** Salman F., Kováč J., Ďurkovič J. — *MicroCT-Based
Structural Phenotyping and Random Forest Segmentation of the Central Cylinder in
Salicornia europaea Stems: A Methodological Pilot Study* (submitted to
*The Plant Journal*).

**Archived release:** https://doi.org/10.5281/zenodo.22641406 (code)
**Data record:** https://doi.org/10.5281/zenodo.22641406 (raw volumes, models, predictions)

## Contents

| File | Purpose |
|---|---|
| `vgl_to_nrrd.py` | Parses a VGSTUDIO project file (`.vgl`) and writes a detached NRRD header (`.nhdr`) pointing at the original `.vol` data, so the volume opens directly in 3D Slicer (or any NRRD-aware tool) without VGSTUDIO. No data are duplicated. |
| `vol_to_tiff.py` | Reads a `.vol`/`.nhdr` pair and writes a multi-page 16-bit TIFF stack for downstream processing. |
| `volume_quality_check.py` | Computes an Otsu-based contrast-to-noise ratio (CNR) and intensity statistics on five mid-volume slices, to flag volumes that need denoising before segmentation. |
| `prepare_grid_phase.py` | Converts a 16-bit TIFF stack (from `vol_to_tiff.py`) into an 8-bit `GRID-8bit.tif` by linear contrast-stretching a fixed intensity window (default 12000–50000 → 0–255) to 255, processed slice-by-slice; also creates a `PHASE-8bit.tif` symlink for the dual-channel leaf-traits-microct input. |
| `radial_boundary_estimation.py` | Computes a local-texture (windowed standard deviation) vs. radius profile centred on the stem and overlays candidate boundary circles, to estimate tissue-boundary radii (central cylinder / parenchyma cortex / palisade tissue). Includes a robust "solid-shape" stem-centre estimator. |
| `nlm_denoise_crop.py` | Applies non-local-means denoising to a cropped region around the stem for low-contrast volumes whose radial profile is confounded by heteroscedastic noise, and compares raw vs. denoised profiles. |
| `build_label_stack.py` | Generates ternary training-label stacks (central cylinder + pith, parenchyma cortex, palisade tissue) geometrically from a fixed stem centre and boundary radii. Labels are **not** manually traced. |
| `class_identification.py` | Overlays the known central-cylinder radius on a predicted slice to identify which output value of the Random Forest prediction corresponds to central cylinder + pith. |
| `run_commands_reference.sh` | Documentation file (not executable as-is) listing the exact parameters, software/hardware environment, and commands used for each of the four samples. |
| `compatibility_patches.md` | Table of minimal changes required to run the third-party leaf-traits-microct pipeline under current Python/NumPy/SciPy/scikit-image versions. |
| `requirements.txt` | Python package versions used for the manuscript. |

## Relationship to the original analysis scripts

The scripts in this repository are consolidated, parameterised versions of the
per-sample scripts used during the analysis. The original per-sample scripts,
run as-is to produce the manuscript results (including the volume computations
reported in Tables 2 and 3), are archived unmodified in the Zenodo data record
(`per_sample_scripts/`). The parameter values used for each sample are listed
in `run_commands_reference.sh`.

## Third-party dependency: leaf-traits-microct

Random Forest training and full-stack prediction were performed with the
**leaf-traits-microct** pipeline (Théroux-Rancourt et al., 2020;
https://github.com/plant-microct-tools/leaf-traits-microct), without modifying
its classifier logic, feature layers, or training methodology. Running it under
Python 3.14 required minimal compatibility fixes (e.g. `sklearn.externals` →
`joblib`, removed NumPy/SciPy aliases, `img_as_ubyte` safe casting), documented
as a table of changes in [`compatibility_patches.md`](compatibility_patches.md).
Because the pipeline expects dual-channel (phase + absorption) input and the
present data are single-channel, the same 8-bit stack was supplied to both
inputs. Please cite the original tool when using it.

## Requirements

- Python 3.8+ (the manuscript analyses used Python 3.14.4 on macOS 12.7.6)
- `numpy`, `scipy`, `scikit-image`, `scikit-learn`, `tifffile`, `matplotlib`,
  `joblib`, `PyWavelets`
- Exact versions used for the manuscript are pinned in `requirements.txt`:
  `pip install -r requirements.txt`
- 3D Slicer 5.6.2 was used for visual inspection and manual annotation; Fiji (ImageJ 1.54p) was used for pixel-distance measurement to confirm the Salicornia 4 central-cylinder radius on the annotated cross-section (Figure 2b).

All analyses, including Random Forest training and prediction, were run on a
2017 MacBook Air (dual-core Intel Core i5, 8 GB RAM, no GPU); volumes were
downsampled (rescale factor 2 or 4) before training for this reason.

## Usage

### 1. Convert a `.vgl` project to an open format
```bash
python3 vgl_to_nrrd.py "/path/to/Salicornia_5/Salicornia_5.vgl"
```

### 2. Check reconstruction quality
```bash
python3 volume_quality_check.py "/path/to/Salicornia_5.nhdr"
```

### 3. Export a 16-bit TIFF stack
```bash
python3 vol_to_tiff.py "/path/to/Salicornia_5.nhdr" Salicornia_5_stack.tif
```
Convert to 8-bit in Fiji (Image › Type › 8-bit, default display range) for the
Random Forest step; see `run_commands_reference.sh`, section 0d.

### 4. Estimate anatomical boundaries
```bash
python3 radial_boundary_estimation.py --stack Salicornia_1_stack.tif --slice 500 \
    --voxel-um 2.5 --center-y 520 --center-x 591 --candidate-radii 90 190 250
```

### 5. (If needed) Denoise a low-contrast volume
```bash
python3 nlm_denoise_crop.py --stack Salicornia_1_stack.tif --slice 500 \
    --center-y 520 --center-x 591 --half-width 400 --voxel-um 2.5 --known-radius-um 87.5
```

### 6. Build training labels
Edit the centre / radii / slice-list constants at the top of
`build_label_stack.py` for the sample being processed (values for the four
manuscript samples are in `run_commands_reference.sh`), then:
```bash
python3 build_label_stack.py
```

### 7. Train and predict (leaf-traits-microct)
See `run_commands_reference.sh`, section 4, for the exact per-sample invocations.

### 8. Identify predicted classes
Edit the `samples` dictionary at the top of `class_identification.py` with your
prediction file paths, then:
```bash
python3 class_identification.py
```

## Data

Raw microCT reconstructions (`.vgl`/`.vol` with detached NRRD headers), 16-bit
and 8-bit TIFF stacks, 3D Slicer annotation files, geometrically generated
training label stacks, trained Random Forest models, full-stack prediction
volumes, the 12 manually annotated anatomical reference images, and the original
per-sample scripts are archived at Zenodo: https://doi.org/10.5281/zenodo.XXXXXXX

## License

Released under the [MIT License](LICENSE). The third-party leaf-traits-microct
tool has its own license; consult the original repository before redistributing.

## Citation

If you use this pipeline, please cite the associated manuscript (full citation
to be updated upon publication) and the Zenodo code DOI above.
