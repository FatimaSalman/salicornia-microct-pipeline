#!/usr/bin/env python3
"""denoised_threshold_sweep_S5.py
إعادة اختبار Method A (threshold+closing) لكنها المرة على الصورة
المُنظّفة (Z-averaged, 31 شرائح) بدل الشريحة الخام المفردة الصاخبة.
الهدف: التحقق هل الشكل الحقيقي (غير الدائري) يُلتقط بنظافة كمساحة،
بدل افتراض نصف قطر ثابت غير مناسب لهذه البنية."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, label, binary_fill_holes, binary_closing
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

def load_single_slice(nhdr_path, slice_index, sizes):
    info = parse_nhdr(nhdr_path)
    nx, ny, nz = sizes
    slice_bytes = nx * ny * 2
    vol_path = os.path.join(os.path.dirname(nhdr_path), info['data file'])
    with open(vol_path, 'rb') as f:
        f.seek(slice_index * slice_bytes)
        raw = f.read(slice_bytes)
    return np.frombuffer(raw, dtype=np.uint16).reshape(ny, nx)

def local_std(image, window=9):
    img = image.astype(np.float64)
    mean = uniform_filter(img, size=window)
    mean_sq = uniform_filter(img**2, size=window)
    return np.sqrt(np.maximum(mean_sq - mean**2, 0))

if __name__ == "__main__":
    nhdr_path = "Salicornia_5.nhdr"
    sizes = [int(x) for x in parse_nhdr(nhdr_path)['sizes'].split()]
    nx, ny, nz = sizes

    zc = 501
    half_window = 15

    print(f"Z-averaging حول z={zc} (±{half_window})...")
    accum = np.zeros((ny, nx)); count = 0
    for z in range(zc-half_window, zc+half_window+1):
        if 0 <= z < nz:
            accum += load_single_slice(nhdr_path, z, sizes); count += 1
    denoised = accum / count

    texture = local_std(denoised, window=9)
    print(f"Texture range: {texture.min():.1f} - {texture.max():.1f}")

    # مسح threshold على الصورة المُنظّفة (بدل الخام الصاخبة)
    thresholds = [500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300]

    # نقيّد البحث بمنطقة قريبة من المركز (r<200px) لتفادي القشرة الخارجية
    # هذا ليس "قصاً جغرافياً تعسفياً" بل نافذة فحص فقط للتشخيص
    cy_approx, cx_approx = 320, 320
    yy, xx = np.mgrid[0:ny, 0:nx]
    dist_from_center = np.sqrt((yy-cy_approx)**2 + (xx-cx_approx)**2)
    roi_mask = dist_from_center < 200

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    print(f"\n{'thr':>6} | {'n_comp(raw)':>12} | {'n_comp(closed+fill)':>20} | {'largest_frac':>13}")
    for i, thr in enumerate(thresholds):
        binary = (texture > thr) & roi_mask
        closed = binary_closing(binary, structure=np.ones((3,3)))
        filled = binary_fill_holes(closed)
        labeled, n_comp = label(filled)

        sizes_comp = np.bincount(labeled.ravel())[1:]
        largest_frac = sizes_comp.max()/binary.size if len(sizes_comp) else 0
        n_comp_raw = label(binary)[1]

        print(f"{thr:6d} | {n_comp_raw:12d} | {n_comp:20d} | {largest_frac*100:12.1f}%")

        axes[i].imshow(filled, cmap='gray')
        axes[i].set_title(f'thr={thr}\nn_comp={n_comp}, largest={largest_frac*100:.1f}%', fontsize=9)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('denoised_threshold_sweep_S5.png', dpi=150)
    plt.show()
    print("\nSaved: denoised_threshold_sweep_S5.png")