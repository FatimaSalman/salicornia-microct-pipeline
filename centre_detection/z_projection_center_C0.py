#!/usr/bin/env python3
"""z_projection_center_S5.py
حساب Z-projection (متوسط texture عبر نطاق شرائح) لإيجاد المركز الحقيقي
لعينة S5 - تفادي فخ استخدام مركز الصورة الهندسي."""

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

def load_single_slice(nhdr_path, slice_index, sizes=None):
    info = parse_nhdr(nhdr_path)
    if sizes is None:
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
    info = parse_nhdr(nhdr_path)
    sizes = [int(x) for x in info['sizes'].split()]
    nx, ny, nz = sizes
    print(f"Volume size: X={nx}, Y={ny}, Z={nz}")

    # نطاق Z واسع نسبياً لتغطية جزء جيد من الساق (مشابه لمنطق S1: z=300-700 من 1000)
    z_start, z_end = 300, 700
    step = 10  # نأخذ كل 10 شرائح لتسريع الحساب (كافٍ لتقدير المركز)

    print(f"Computing Z-projection over z={z_start}-{z_end} (step={step}) ...")

    accum_texture = np.zeros((ny, nx), dtype=np.float64)
    count = 0

    for z in range(z_start, z_end, step):
        slice_raw = load_single_slice(nhdr_path, z, sizes=sizes)
        texture = local_std(slice_raw, window=9)
        accum_texture += texture
        count += 1

    avg_texture = accum_texture / count
    print(f"تم استخدام {count} شريحة بالمتوسط")

    # تقدير المركز: استخدام أعلى 20% من قيم التكستشر كأوزان (centroid موزون)
    threshold_pct = np.percentile(avg_texture, 80)
    mask = avg_texture >= threshold_pct
    ys, xs = np.nonzero(mask)
    weights = avg_texture[ys, xs]

    cy = np.average(ys, weights=weights)
    cx = np.average(xs, weights=weights)

    print(f"\n🎯 المركز المقدّر (weighted centroid, top 20%): y={cy:.1f}, x={cx:.1f}")
    print(f"   (مركز الصورة الهندسي كان سيكون: y={ny/2:.1f}, x={nx/2:.1f})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(avg_texture, cmap='inferno')
    axes[0].scatter([cx], [cy], c='cyan', s=100, marker='+', linewidths=3)
    axes[0].set_title(f'Z-averaged texture (z={z_start}-{z_end})\nCenter=({cx:.0f},{cy:.0f})')

    axes[1].imshow(mask, cmap='gray')
    axes[1].scatter([cx], [cy], c='red', s=100, marker='+', linewidths=3)
    axes[1].set_title('Top 20% mask used for centroid')

    plt.tight_layout()
    plt.savefig('z_projection_center_S5.png', dpi=150)
    plt.show()
    print("Saved: z_projection_center_S5.png")