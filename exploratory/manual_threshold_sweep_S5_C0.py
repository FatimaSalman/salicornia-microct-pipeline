#!/usr/bin/env python3
"""manual_threshold_sweep_S5.py
مسح threshold يدوي خشن (خطوة 500) على شريحة S5 - نبحث عن النطاق الذي
يفصل حلقة cc عن باقي الأنسجة المحيطة (تشخيص نمط الاتصال الكاذب)."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, label, binary_fill_holes
import os

def parse_nhdr(nhdr_path):
    info = {}
    with open(nhdr_path, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' not in line or line.startswith('NRRD'):
                continue
            key, val = line.split(':', 1)
            info[key.strip()] = val.strip()
    return info

def load_single_slice(nhdr_path, slice_index):
    info = parse_nhdr(nhdr_path)
    sizes = [int(x) for x in info['sizes'].split()]
    nx, ny, nz = sizes
    dtype = np.uint16
    bytes_per_voxel = 2
    slice_bytes = nx * ny * bytes_per_voxel
    data_file = info['data file']
    vol_path = os.path.join(os.path.dirname(nhdr_path), data_file)
    with open(vol_path, 'rb') as f:
        f.seek(slice_index * slice_bytes)
        raw = f.read(slice_bytes)
    arr = np.frombuffer(raw, dtype=dtype).reshape(ny, nx)
    return arr

def local_std(image, window=9):
    img = image.astype(np.float64)
    mean = uniform_filter(img, size=window)
    mean_sq = uniform_filter(img**2, size=window)
    variance = np.maximum(mean_sq - mean**2, 0)
    return np.sqrt(variance)

if __name__ == "__main__":
    nhdr_path = "Salicornia_5.nhdr"
    slice_index = 501

    slice_raw = load_single_slice(nhdr_path, slice_index)
    texture = local_std(slice_raw, window=9)

    print(f"Texture value range: {texture.min():.1f} - {texture.max():.1f}")

    # مسح خشن بخطوة 500 حول Otsu (2453) وحتى قيم أعلى بكثير
    thresholds = [1500, 2000, 2453, 3000, 3500, 4000, 4500, 5000, 5500, 6000]

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    for i, thr in enumerate(thresholds):
        binary = texture > thr
        filled = binary_fill_holes(binary)
        labeled, n_comp = label(filled)
        sizes_comp = np.bincount(labeled.ravel())[1:]
        largest_frac = sizes_comp.max() / binary.size if len(sizes_comp) else 0

        axes[i].imshow(filled, cmap='gray')
        axes[i].set_title(f'thr={thr}\nn_comp={n_comp}, largest={largest_frac*100:.1f}%',
                           fontsize=10)
        axes[i].axis('off')

        print(f"thr={thr:5d} | n_comp(after fill)={n_comp:4d} | largest_frac={largest_frac*100:5.1f}%")

    plt.tight_layout()
    plt.savefig('threshold_sweep_S5_z501.png', dpi=150)
    plt.show()
    print("Saved: threshold_sweep_S5_z501.png")