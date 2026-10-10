#!/usr/bin/env python3
"""radial_profile_test_S5.py
اختبار البروتوكول الكامل على شريحة مركزية واحدة من S5:
Z-denoising + centroid دقيق + texture + radial profile
الهدف: معايرة نطاقات peak/background المناسبة لـS5 (مختلفة عن S1)."""

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

    zc = 501  # شريحة مركزية للاختبار
    half_window = 15  # ±15 شريحة = 31 شريحة إجمالي (نفس بروتوكول S1)

    print(f"Z-averaging raw intensity حول z={zc} (±{half_window} شرائح)...")
    accum = np.zeros((ny, nx), dtype=np.float64)
    count = 0
    for z in range(zc - half_window, zc + half_window + 1):
        if 0 <= z < nz:
            slice_raw = load_single_slice(nhdr_path, z, sizes=sizes)
            accum += slice_raw
            count += 1
    denoised = accum / count
    print(f"تم استخدام {count} شريحة بالمتوسط")

    print("حساب centroid دقيق محلياً...")
    approx_cy, approx_cx = 321.8, 318.8  # من نتيجة Z-projection السابقة
    texture_for_center = local_std(denoised, window=9)
    cy, cx = estimate_center(texture_for_center, approx_cy, approx_cx, search_radius=100)
    print(f"المركز الدقيق: y={cy:.1f}, x={cx:.1f}")

    print("حساب texture (window=5) على الصورة المُنظّفة...")
    texture = local_std(denoised, window=5)

    print("حساب radial profile (360 زاوية)...")
    max_r = 300  # نطاق أوسع من S1 لأن cc أكبر هنا
    radii, profile = radial_profile(texture, cy, cx, max_r=max_r)

    # تنعيم البروفايل
    profile_smooth = savgol_filter(profile, window_length=9, polyorder=2)

    # رسم النتائج للفحص البصري ومعايرة النطاقات
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    axes[0].imshow(denoised, cmap='gray')
    axes[0].scatter([cx], [cy], c='red', s=100, marker='+', linewidths=3)
    circle_guess = plt.Circle((cx, cy), 71.6, color='cyan', fill=False, linewidth=2, label='Fiji cc r=71.6px')
    axes[0].add_patch(circle_guess)
    axes[0].legend()
    axes[0].set_title(f'Denoised slice (z={zc}, avg of {count})')

    axes[1].imshow(texture, cmap='inferno')
    axes[1].scatter([cx], [cy], c='cyan', s=100, marker='+', linewidths=3)
    axes[1].set_title('Texture map (window=5) on denoised')

    axes[2].plot(radii, profile, alpha=0.4, label='raw profile')
    axes[2].plot(radii, profile_smooth, linewidth=2, label='smoothed')
    axes[2].axvline(71.6, color='cyan', linestyle='--', label='Fiji cc radius (71.6px)')
    axes[2].set_xlabel('Radius (px)')
    axes[2].set_ylabel('Mean texture')
    axes[2].legend()
    axes[2].set_title('Radial profile (full range)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('radial_profile_test_S5.png', dpi=150)
    plt.show()
    print("Saved: radial_profile_test_S5.png")

    # طباعة قيم البروفايل كل 10px للمساعدة بمعايرة النطاقات يدوياً
    print("\n--- قيم البروفايل كل 10px (للمعايرة اليدوية) ---")
    for r in range(0, max_r, 10):
        print(f"r={r:3d}px | texture={profile_smooth[r]:8.2f}")