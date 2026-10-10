# Random forest tissue classification

The classifier used in this study is the `leaf-traits-microct` pipeline:

- Upstream repository: https://github.com/plant-microct-tools/leaf-traits-microct
- Reference: Theroux-Rancourt, G. et al. (2020)
  leaves is limited by the scaling of cell size and genome size.*
- Version used in this study: <commit hash or release tag>
- Upstream licence: <licence of the upstream repository>

**That code is not redistributed here.** Clone it from the link above.

## What this directory contains

Only the scripts written for this study:

| Script | Purpose |
|---|---|
| `prepare_S1.py`, `prepare_S4.py`, `prepare_S5.py`, `prepare_S8.py` | build the per-sample inputs the pipeline expects |
| `build_label_stack.py` | assemble the sparse manual training labels |
| `identify_prediction_classes.py` | map predicted class indices onto tissue types |

## Label convention

Training labels use the values `{0, 51, 102, 153}`, where `153` denotes the
central cylinder. The deposited reference masks are binary, with label `1`
denoting the central cylinder.

## Caching

The upstream pipeline silently reuses `*_2x-smaller.tif` and
`*_RF_model.joblib` if they are present in the working directory. Delete them
before any run that is meant to start from scratch, otherwise the results of a
previous configuration are reused without warning.
