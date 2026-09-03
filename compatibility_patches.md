# Compatibility patches for leaf-traits-microct (Python 3.14 / NumPy 2.x / SciPy 1.x)

Original tool: https://github.com/plant-microct-tools/leaf-traits-microct
(Théroux-Rancourt et al. 2020, Applications in Plant Sciences — MIT License)

These are minimal compatibility fixes only; no changes to the classifier
logic, feature layers, or training methodology were made.

| # | File | Original | Patched | Reason |
|---|---|---|---|---|
| 1 | Leaf_Segmentation_py3.py | `from sklearn.externals import joblib` | `import joblib` | Removed in scikit-learn ≥0.23 |
| 2 | Leaf_Segmentation_Functions_py3.py | `dtype=np.bool` | `dtype=bool` | np.bool removed in NumPy ≥1.24 |
| 3 | Leaf_Segmentation_Functions_py3.py | `img_as_ubyte(...)` | custom `safe_ubyte()` | skimage's img_as_ubyte now rejects float arrays outside [-1,1]; safe_ubyte scales [0,1]→[0,255] or clips [0,255] directly |
| 4 | Leaf_Segmentation_Functions_py3.py | `transform.resize(...)` | + `preserve_range=True` | Newer skimage normalizes to [0,1] by default, corrupting 8-bit intensity scale |
| 5 | Leaf_Segmentation_Functions_py3.py | `sp.unique`, `sp.around`, `sp.zeros_like` | `np.unique`, `np.around`, `np.zeros_like` | Removed from top-level scipy namespace |
| 6 | Leaf_Segmentation_Functions_py3.py | `sp.ndimage.filters.minimum_filter` | `spim.minimum_filter` | scipy.ndimage.filters submodule removed |