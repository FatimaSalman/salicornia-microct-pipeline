#!/usr/bin/env python3
"""half_max_comparison_S5.py
مقارنة عدة طرق لحساب نصف القطر من البروفايل الشعاعي على نفس الشريحة،
لتحديد الطريقة الأنسب لبنية S5 (المختلفة عن S1) - بدقة 1px."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.signal import savgol_filter
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

def local_std(image, window=5):
    img = image.astype(np.float64)
    mean = uniform_filter(img, size=window)
    mean_sq = uniform_filter(img**2, size=window)
    variance = np.maximum(mean_sq - mean**2, 0)
    return np.sqrt(variance)

def estimate_center(texture_img, approx_cy, approx_cx, search_radius=100):
    ny, nx = texture_img.shape
    y0, y1 = max(0, int(approx_cy - search_radius)), min(ny, int(approx_cy + search_radius))
    x0, x1 = max(0, int(approx_cx - search_radius)), min(nx, int(approx_cx + search_radius))
    region = texture_img[y0:y1, x0:x1]
    thr = np.percentile(region, 80)
    mask = region >= thr
    ys, xs = np.nonzero(mask)
    weights = region[ys, xs]
    cy = np.average(ys, weights=weights) + y0
    cx = np.average(xs, weights=weights) + x0
    return cy, cx

def radial_profile(texture_img, cy, cx, max_r=300, n_angles=360):
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    radii = np.arange(0, max_r)
    profile = np.zeros(max_r)
    ny, nx = texture_img.shape
    for i, r in enumerate(radii):
        ys = (cy + r * np.sin(angles)).astype(int)
        xs = (cx + r * np.cos(angles)).astype(int)
        valid = (ys >= 0) & (ys < ny) & (xs >= 0) & (xs < nx)
        if valid.sum() > 0:
            profile[i] = texture_img[ys[valid], xs[valid]].mean()
        else:
            profile[i] = np.nan
    return radii, profile

if __name__ == "__main__":
    nhdr_path = "Salicornia_5.nhdr"
    info = parse_nhdr(nhdr_path)
    sizes = [int(x) for x in info['sizes'].split()]
    nx, ny, nz = sizes

    zc = 501
    half_window = 15

    accum = np.zeros((ny, nx), dtype=np.float64)
    count = 0
    for z in range(zc - half_window, zc + half_window + 1):
        if 0 <= z < nz:
            slice_raw = load_single_slice(nhdr_path, z, sizes=sizes)
            accum += slice_raw
            count += 1
    denoised = accum / count

    approx_cy, approx_cx = 321.8, 318.8
    texture_for_center = local_std(denoised, window=9)
    cy, cx = estimate_center(texture_for_center, approx_cy, approx_cx, search_radius=100)

    texture = local_std(denoised, window=5)
    radii, profile = radial_profile(texture, cy, cx, max_r=250)
    profile_smooth = savgol_filter(profile, window_length=9, polyorder=2)

    FIJI_R = 71.6

    # --- طريقة 1: القمة الخام مباشرة ضمن نطاق معقول ---
    search_range_1 = (50, 95)
    idx1 = np.arange(search_range_1[0], search_range_1[1])
    peak_idx_1 = idx1[np.argmax(profile_smooth[idx1])]
    r_method1 = peak_idx_1
    print(f"[طريقة 1] Raw peak في [50,95]: r={r_method1}px "
          f"(فرق عن Fiji: {(r_method1-FIJI_R)/FIJI_R*100:+.1f}%)")

    # --- طريقة 2: half-max مع خلفية محلية قريبة (بعد القمة مباشرة، قبل نتوء pc) ---
    peak_val = profile_smooth[peak_idx_1]
    bg_range_2 = (115, 135)
    bg_val_2 = profile_smooth[bg_range_2[0]:bg_range_2[1]].mean()
    half_max_2 = (peak_val + bg_val_2) / 2
    # البحث عن أول تقاطع بعد القمة
    r_method2 = None
    for r in range(peak_idx_1, bg_range_2[0]):
        if profile_smooth[r] <= half_max_2:
            r_method2 = r
            break
    print(f"[طريقة 2] Half-max (خلفية محلية {bg_range_2}, val={bg_val_2:.1f}): "
          f"half_max_thr={half_max_2:.1f}, r={r_method2}px "
          f"(فرق عن Fiji: {(r_method2-FIJI_R)/FIJI_R*100:+.1f}%)" if r_method2 else "لم يوجد تقاطع")

    # --- طريقة 3: half-max مع خلفية بعيدة (cortex خارجي حقيقي) ---
    bg_range_3 = (220, 250)
    bg_val_3 = profile_smooth[bg_range_3[0]:bg_range_3[1]].mean()
    half_max_3 = (peak_val + bg_val_3) / 2
    r_method3 = None
    for r in range(peak_idx_1, bg_range_3[0]):
        if profile_smooth[r] <= half_max_3:
            r_method3 = r
            break
    print(f"[طريقة 3] Half-max (خلفية بعيدة {bg_range_3}, val={bg_val_3:.1f}): "
          f"half_max_thr={half_max_3:.1f}, r={r_method3}px "
          f"(فرق عن Fiji: {(r_method3-FIJI_R)/FIJI_R*100:+.1f}%)" if r_method3 else "لم يوجد تقاطع")

    # رسم توضيحي لكل الطرق
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(radii, profile_smooth, linewidth=2, color='black', label='smoothed profile')
    ax.axvline(FIJI_R, color='cyan', linestyle='--', linewidth=2, label=f'Fiji r={FIJI_R}px')
    ax.axvline(r_method1, color='green', linestyle=':', linewidth=2, label=f'Method1 (raw peak)={r_method1}px')
    if r_method2:
        ax.axvline(r_method2, color='orange', linestyle=':', linewidth=2, label=f'Method2 (local half-max)={r_method2}px')
        ax.axhline(half_max_2, color='orange', alpha=0.3)
    if r_method3:
        ax.axvline(r_method3, color='red', linestyle=':', linewidth=2, label=f'Method3 (far half-max)={r_method3}px')
        ax.axhline(half_max_3, color='red', alpha=0.3)
    ax.set_xlabel('Radius (px)')
    ax.set_ylabel('Mean texture')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('مقارنة طرق حساب نصف قطر cc - S5 slice 501')
    plt.tight_layout()
    plt.savefig('half_max_comparison_S5.png', dpi=150)
    plt.show()
    print("Saved: half_max_comparison_S5.png")