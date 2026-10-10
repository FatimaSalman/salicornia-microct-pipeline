#!/usr/bin/env python3
"""otsu_diagnostic_S5.py
اختبار Otsu + فحص المكونات المتصلة على شريحة S5 - خطوة تشخيصية أولى
لتحديد أي طريقة (Threshold أم Radial Profile) ستُستخدم."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, label, binary_fill_holes
from skimage.filters import threshold_otsu
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

    # Otsu على خريطة التكستشر
    otsu_val = threshold_otsu(texture)
    print(f"Otsu threshold (auto) = {otsu_val:.2f}")

    binary = texture > otsu_val
    labeled, n_comp = label(binary)
    print(f"عدد المكونات المتصلة (raw): {n_comp}")

    filled = binary_fill_holes(binary)
    labeled_f, n_comp_f = label(filled)
    print(f"عدد المكونات بعد fill_holes: {n_comp_f}")

    # حساب مساحة أكبر مكوّن (كنسبة من الصورة الكاملة) - فحص التشبع
    sizes_comp = np.bincount(labeled.ravel())[1:]  # استبعاد الخلفية
    largest_frac = sizes_comp.max() / binary.size if len(sizes_comp) else 0
    print(f"نسبة أكبر مكوّن من مساحة الصورة: {largest_frac*100:.1f}%")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(texture, cmap='inferno')
    axes[0].set_title('Texture map')
    axes[1].imshow(binary, cmap='gray')
    axes[1].set_title(f'Otsu binary (thr={otsu_val:.0f}), n_comp={n_comp}')
    axes[2].imshow(filled, cmap='gray')
    axes[2].set_title(f'After fill_holes, n_comp={n_comp_f}')
    plt.tight_layout()
    plt.savefig('otsu_test_S5_z501.png', dpi=150)
    plt.show()
    print("Saved: otsu_test_S5_z501.png")