# Compatibility patches for the leaf-traits-microct pipeline

The random forest segmentation used in this study is the third-party
`leaf-traits-microct` pipeline:

- Repository: https://github.com/plant-microct-tools/leaf-traits-microct
- Reference: Theroux-Rancourt, G. et al. (2020)
- Version used: <commit hash or release tag — FILL IN>

That pipeline is **not** redistributed here. It was written against older
releases of Python and the scientific stack, and does not run unmodified on
the versions used in this study. The changes required are documented below so
that the analysis can be reproduced.

## Environment used

| Package | Version |
|---|---|
| Python | 3.14 |
| NumPy | <fill in: `python3 -c "import numpy; print(numpy.__version__)"`> |
| SciPy | <fill in> |
| scikit-image | <fill in> |
| scikit-learn | <fill in> |
| tifffile | <fill in> |

## Patches

> Fill one block per change. Keep the original line and the replacement so a
> reader can apply the patch to a fresh clone without guessing.

### 1. `<file name in the upstream repository>`

**Symptom**

```
<the exact error message>
```

**Cause**

<one sentence: which API changed>

**Change**

```python
# before
<original line>

# after
<replacement line>
```

---

### 2. `<file name>`

**Symptom**

```
<error>
```

**Cause**

<...>

**Change**

```python
# before
<...>

# after
<...>
```

---

## Behaviour to be aware of

The pipeline silently reuses cached intermediates if they are present in the
working directory:

- `*_2x-smaller.tif` and `*_4x-smaller.tif` — the rescaled stacks
- `*_RF_model.joblib` — the trained classifier

If either exists, it is loaded instead of being recomputed, and a run that was
meant to test a new configuration will reproduce the previous one without any
warning. Delete these files before a clean run.
