# Compatibility patches for the leaf-traits-microct pipeline

The random forest segmentation used in this study is the third-party
`leaf-traits-microct` pipeline:

- Repository: https://github.com/plant-microct-tools/leaf-traits-microct
- Reference: Theroux-Rancourt, G., Jenkins, M.R., Brodersen, C.R., McElrone, A.,
  Forrestel, E.J. and Earles, J.M. (2020) Digitally deconstructing leaves in 3D
  using X-ray microcomputed tomography and machine learning.
  *Applications in Plant Sciences*, 8, e11380.
- Version used: `master` branch, downloaded as a ZIP archive; the source files
  carry an upstream modification date of 2019-11-06. No release tag or commit
  hash was recorded at download time.
- Entry points used: `Leaf_Segmentation_py3.py` and
  `Leaf_Segmentation_Functions_py3.py`

That pipeline is **not** redistributed here. It was written against the 2019
scientific Python stack and does not run unmodified on the versions used in
this study. Several APIs it calls have since been removed. The changes required
are documented below so that the analysis can be reproduced: download the
upstream repository, apply these edits, and the pipeline runs.

## Environment used

| Package | Version |
|---|---|
| Python | 3.14.4 |
| NumPy | 2.5.0 |
| SciPy | 1.18.0 |
| scikit-image | 0.26.0 |
| scikit-learn | 1.9.0 |
| tifffile | 2026.6.1 |
| joblib | 1.5.3 |
| matplotlib | 3.11.0 |
| pynrrd | 1.1.3 |

macOS, Apple silicon.

## Patches

Changes 1–5 are pure API updates and do not alter the computation. Change 6
does alter values and is described in detail. Change 7 affects only how the
output is written to disk.

### 1. `Leaf_Segmentation_py3.py` — joblib import

**Symptom**

```
ModuleNotFoundError: No module named 'sklearn.externals'
```

**Cause**

`sklearn.externals.joblib` was deprecated in scikit-learn 0.21 and removed in
0.23. `joblib` is now an independent package.

**Change**

```python
# before
from sklearn.externals import joblib

# after
import joblib
```

---

### 2. `Leaf_Segmentation_Functions_py3.py` — `scipy.ndimage.filters`

**Symptom**

```
AttributeError: module 'scipy.ndimage' has no attribute 'filters'
```

**Cause**

The `scipy.ndimage.filters` namespace was deprecated in SciPy 1.8 and removed
in SciPy 1.15. The functions now live directly in `scipy.ndimage`, which the
module already imports as `spim`.

**Change**

```python
# before
def minFilter(img):
    filtered = sp.ndimage.filters.minimum_filter(img, size=(3, 1, 1))
    return filtered

# after
def minFilter(img):
    filtered = spim.minimum_filter(img, size=(3, 1, 1))
    return filtered
```

---

### 3. `Leaf_Segmentation_Functions_py3.py` — NumPy functions re-exported by SciPy

**Symptom**

```
AttributeError: module 'scipy' has no attribute 'unique'
```

**Cause**

SciPy used to re-export the NumPy namespace (`scipy.unique`, `scipy.around`,
`scipy.zeros_like`, …). These aliases were deprecated in SciPy 1.0 and removed
in 2.0. They must be called from NumPy directly.

**Change** — in `local_thickness`:

```python
# before
sizes = sp.unique(sp.around(dt, decimals=0))
im_new = sp.zeros_like(im)

# after
sizes = np.unique(np.around(dt, decimals=0))
im_new = np.zeros_like(im)
```

---

### 4. `Leaf_Segmentation_Functions_py3.py` — `np.bool`

**Symptom**

```
AttributeError: module 'numpy' has no attribute 'bool'
```

**Cause**

The deprecated aliases `np.bool`, `np.int`, `np.float` and `np.object` were
removed in NumPy 1.24. The builtin `bool` is the intended replacement.

**Change** — in `Threshold_GridPhase_invert_down`:

```python
# before
tmp = np.zeros(grid_img.shape, dtype=np.bool)

# after
tmp = np.zeros(grid_img.shape, dtype=bool)
```

---

### 5. `Leaf_Segmentation_Functions_py3.py` — `img_as_ubyte` on out-of-range input

**Symptom**

```
ValueError: Images of type float must be between -1 and 1.
```

**Cause**

`skimage.img_as_ubyte` enforces the range convention for floating point images
strictly. Several call sites in the pipeline pass arrays that are already in
0–255, or integer arrays cast to `uint64`, which earlier scikit-image releases
accepted silently. A local replacement reproduces the behaviour the pipeline
was written against.

**Change** — added near the top of the module:

```python
def safe_ubyte(arr):
    """Robust replacement for skimage's img_as_ubyte. Scales float arrays
    in [0,1] up to [0,255] (matching skimage's original behavior), and
    clips/casts arrays already in [0,255] directly."""
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        amin, amax = arr.min(), arr.max()
        if amax <= 1.0 and amin >= 0.0:
            arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)
```

Every call to `img_as_ubyte` was then replaced by `safe_ubyte`, in
`RFPredictCTStack`, `make_QC_images`, `localthick_load_and_resize`,
`Threshold_GridPhase_invert_down`, `Trim_Individual_Stack` and
`Load_Resize_and_Save_Stack`.

---

### 6. `Leaf_Segmentation_Functions_py3.py` — `transform.resize` and `preserve_range`

**This change affects the values, not only the API, and is the one patch that
matters for reproducibility.**

**Symptom**

Resized stacks came out effectively blank: label and threshold volumes that
should have held values in 0–255 contained only 0 and 1 after resizing.

**Cause**

`skimage.transform.resize` normalises its input to the 0–1 float range unless
`preserve_range=True` is passed. The pipeline then converts the result back to
8-bit. Under the older scikit-image behaviour the original range survived; from
scikit-image 0.19 onward it does not, so an 8-bit label stack is squashed to
0–1 and the subsequent cast discards the label values.

**Change** — `preserve_range=True` added to all three `resize` calls, in
`localthick_load_and_resize`, `Threshold_GridPhase_invert_down` and
`Load_Resize_and_Save_Stack`:

```python
# before
localthick_stack = transform.resize(
    localthick_small, [...], order=0, anti_aliasing=False)

# after
localthick_stack = transform.resize(
    localthick_small, [...], order=0, anti_aliasing=False,
    preserve_range=True)
```

Anyone reproducing this analysis should apply this patch; without it the
segmentation runs to completion but on degenerate input.

---

### 7. `Leaf_Segmentation_py3.py` — compressed writing of the prediction stack

**Symptom**

The full-stack prediction write failed on the larger volumes with a disk-space
error, after the prediction itself had completed.

**Cause**

`io.imsave` writes an uncompressed TIFF. For the volumes in this study the
prediction stacks are large and highly redundant (four label values), so an
uncompressed write is both unnecessary and, in our case, impossible.

**Change** — at both call sites (batch mode and single-scan mode):

```python
# before
io.imsave(folder_name + sample_name + "fullstack_prediction.tif",
          RFPredictCTStack_out)

# after
import tifffile as _tf
...
_tf.imwrite(
    folder_name + sample_name + "fullstack_prediction.tif",
    RFPredictCTStack_out,
    compression='zlib'
)
```

The pixel values are unchanged; only the file encoding differs.

---

## Behaviour to be aware of

The pipeline silently reuses cached intermediates if they are present in the
working directory:

- `*_2x-smaller.tif` and `*_4x-smaller.tif` — the rescaled stacks
- `*_RF_model.joblib` — the trained classifier
- `*GridPhase_invert_ds.tif` and `*local_thick.tif` — the thresholded and
  local-thickness volumes

If any of these exists, it is loaded instead of being recomputed, and a run
that was meant to test a new configuration will reproduce the previous one
without any warning. Delete these files before a clean run.

The training and test slices are also shuffled with an unseeded
`np.random.shuffle`, so repeated runs on the same data will not give bit-identical
classifiers.
