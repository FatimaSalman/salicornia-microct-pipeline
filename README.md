# Sample-adaptive volumetric quantification of the central cylinder in *Salicornia europaea* stems

Analysis code for the X-ray microCT methodological study of central cylinder
(stele) quantification in *Salicornia europaea* L. stems.

The workflow combines sample-adaptive radial boundary detection with a random
forest tissue classifier, and reports how the choice of boundary criterion
propagates into volumetric estimates.

## Data

The reconstructed volumes and the curated reference segmentations are
deposited in the BioImage Archive under accession `S-BIADXXXXX`:

<https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIADXXXXX>

Derived outputs that are not image data (texture-filtered volumes, trained
classifier objects, radial profiles, volumetric logs) are deposited in Zenodo:
`10.5281/zenodo.XXXXXXX`

**This repository contains code only.** No image data is stored here.

## Samples

| ID | Population | NaCl | Acquisition | Reconstructed slices |
|---|---|---|---|---|
| I-0    | Inowroclaw  | 0 mM    | Normalscan, 4x averaging, 500 projections    | 1000 |
| I-1000 | Inowroclaw  | 1000 mM | Normalscan, 4x averaging, 500 projections    | 800  |
| C-0    | Ciechocinek | 0 mM    | Fastscan, 1x averaging, 1500-1600 projections | 1002 |
| C-1000 | Ciechocinek | 1000 mM | Fastscan, 1x averaging, 1500-1600 projections | 985  |

Isotropic voxel size 2.5 um. Phoenix v|tome|x L 240, 70 kV / 300 uA,
reconstruction in VGSTUDIO MAX 2.2.

## Layout

```
preprocessing/      format conversion, grid/phase preparation, denoising
centre_detection/   locating the stem axis in each slice
radial_profiling/   radial intensity and texture profiles
texture/            local texture volumes (streaming, memory bounded)
volumetric/         one boundary criterion per sample, see below
random_forest/      training stack preparation and class identification
validation/         independent checks of the published numbers
deposition/         preparation of the BioImage Archive upload
figures/            figure generation
```

## The four boundary criteria

Each sample is quantified with the criterion its contrast allows:

| Sample | Criterion | Script |
|---|---|---|
| I-0    | half maximum of the radial profile | `volumetric/I0_half_maximum.py` |
| I-1000 | fixed texture threshold (3500) + closing | `volumetric/I1000_texture_threshold.py` |
| C-0    | maximum gradient | `volumetric/C0_maximum_gradient.py` |
| C-1000 | Otsu on the texture volume (2841.73) | `volumetric/C1000_texture_otsu.py` |

## Reproducing the reported volumes

The central cylinder volumes in Table 2 can be recovered from the deposited
masks without re-running the pipeline:

```bash
python3 validation/check_mask_volume.py --expect 0.0735 I-0_cc_reference_mask.nrrd
```

Multiplying the labelled voxel count by the voxel volume reproduces each
reported value to within the rounding of the fourth decimal place.

| Sample | Labelled voxels | Recovered (mm3) | Table 2 |
|---|---|---|---|
| I-0    | 4,705,793  | 0.073530 | 0.0735 |
| I-1000 | 45,948,975 | 0.717970 | 0.7180 |
| C-0    | 17,803,942 | 0.278187 | 0.2782 |
| C-1000 | 96,832,051 | 1.513051 | 1.5131 |

## Configuring paths

Scripts read the data location from an environment variable:

```bash
export SALICORNIA_DATA=/path/to/your/data
```

## Installation

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Random forest classifier

The tissue classifier builds on the published `leaf-traits-microct` pipeline.
That code is **not** redistributed here; see `random_forest/README.md` for the
upstream repository, the version used, and the scripts specific to this study.

## Citation

See [`CITATION.cff`](CITATION.cff).

## Licence

MIT. See [`LICENSE`](LICENSE).
