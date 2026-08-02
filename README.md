# salicornia-microct-pipeline

Python scripts for converting VGStudio microCT files to an open, vendor-independent format (NRRD) and assessing reconstructed-volume image quality (contrast-to-noise ratio, CNR). Developed as part of a pilot study on microCT-based structural phenotyping of *Salicornia europaea* stems.

**Associated manuscript:** *A Pipeline for microCT-Based Structural Phenotyping and Preliminary Deep Learning Assessment in Salicornia: A Methodological Pilot Study* (Salman, Kováč, Ďurkovič — in preparation for Scientific Reports).

## Contents

| File | Purpose |
|---|---|
| `vgl_to_nrrd.py` | Converts a VGStudio project file (`.vgl`) into a detached NRRD header (`.nhdr`) that points to the original `.vol` data, so the volume can be opened directly in 3D Slicer (or any NRRD-aware tool) without needing VGStudio or a Windows VM. |
| `volume_quality_check.py` | Computes objective image-quality metrics (Otsu-based contrast-to-noise ratio, intensity statistics, edge sharpness) from a raw volume, to give a quantitative answer to "is this scan good enough to resolve internal tissue structures?" |

## Requirements

- Python 3.8+
- `numpy` (`pip install numpy`)
- No other dependencies — both scripts are self-contained.

## Usage

### 1. Convert a `.vgl` project to an open format

```bash
python3 vgl_to_nrrd.py "/path/to/Salicornia_5/Salicornia_5.vgl"
```

This writes a `.nhdr` file next to your `.vgl`/`.vol` files. Drag the resulting `.nhdr` file into **3D Slicer** (File → Add Data, or drag-and-drop) to view the volume.

> **Note:** the `.nhdr` file must stay in the same folder as the original `.vol` file — it references it by relative filename rather than copying the data.

### 2. Check reconstruction quality

```bash
python3 volume_quality_check.py "/path/to/Salicornia_5.nhdr"
```

Prints intensity statistics and an Otsu-based CNR estimate, with a rule-of-thumb interpretation:

- **CNR > 5** — structures clearly separable, good for AI/segmentation
- **CNR 2–5** — structures visible but noisy, denoising recommended
- **CNR < 2** — structures likely hard to distinguish reliably from noise

You can also compare multiple volumes in one call:

```bash
python3 volume_quality_check.py "vol1.nhdr" "vol2.nhdr"
```

## Data

Raw microCT volumes are not included in this repository due to file size. They are available from the corresponding author upon reasonable request.

## License

Released under the [MIT License](LICENSE).

## Citation

If you use this pipeline, please cite the associated manuscript (details above; full citation to be updated upon publication).
