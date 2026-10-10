#!/usr/bin/env python3
"""compute_texture_slice_test.py
اختبار سريع: حساب texture map (windowed std-dev) لشريحة واحدة من S8
قبل معالجة الفوليوم الكامل - للتأكد من النتيجة بصريًا أولاً."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
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
    """يقرأ شريحة واحدة فقط من الملف الخام - سريع، بدون تحميل الفوليوم كامل."""
    info = parse_nhdr(nhdr_path)
    sizes = [int(x) for x in info['sizes'].split()]  # X, Y, Z
    nx, ny, nz = sizes
    dtype = np.uint16  # unsigned short مؤكد من الهيدر
    bytes_per_voxel = 2
    slice_bytes = nx * ny * bytes_per_voxel

    data_file = info['data file']
    vol_path = os.path.join(os.path.dirname(nhdr_path), data_file)

    with open(vol_path, 'rb') as f:
        f.seek(slice_index * slice_bytes)
        raw = f.read(slice_bytes)

    arr = np.frombuffer(raw, dtype=dtype).reshape(ny, nx)  # NRRD: X fastest -> reshape (Y,X)
    return arr

def local_std(image, window=9):
    """حساب local std-dev بسرعة عبر uniform_filter (بدون loop بطيء)."""
    img = image.astype(np.float64)
    mean = uniform_filter(img, size=window)
    mean_sq = uniform_filter(img**2, size=window)
    variance = np.maximum(mean_sq - mean**2, 0)
    return np.sqrt(variance)

if __name__ == "__main__":
    nhdr_path = "Salicornia_8.nhdr"
    slice_index = 492  # منتصف المكدس، نفس الشريحة اللي كنا نشوفها بـSlicer

    print(f"Loading slice {slice_index} ...")
    slice_raw = load_single_slice(nhdr_path, slice_index)
    print(f"Slice shape: {slice_raw.shape}, dtype: {slice_raw.dtype}")
    print(f"Value range: {slice_raw.min()} - {slice_raw.max()}")

    print("Computing texture map (window=9x9)...")
    texture = local_std(slice_raw, window=9)

    # عرض جنب لجنب: الأصلية مقابل التكستشر
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(slice_raw, cmap='gray')
    axes[0].set_title('Original (raw intensity)')
    axes[1].imshow(texture, cmap='inferno')
    axes[1].set_title('Texture map (local std-dev, 9x9)')
    plt.tight_layout()
    plt.savefig('texture_test_slice492.png', dpi=150)
    plt.show()
    print("Saved: texture_test_slice492.png")