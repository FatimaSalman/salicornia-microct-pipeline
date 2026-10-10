#!/usr/bin/env python3
"""compute_texture_slice_test.py
اختبار texture map على عدة شرائح من S1 + تشخيص Otsu أولي
مبني على نفس القالب المؤكد من S4/S8 (بدون تغيير المنطق الأساسي)."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, label
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
    """يقرأ شريحة واحدة فقط من الملف الخام - سريع، بدون تحميل الفوليوم كامل."""
    info = parse_nhdr(nhdr_path)
    sizes = [int(x) for x in info['sizes'].split()]  # X, Y, Z
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

def analyze_slice(nhdr_path, slice_index, out_dir):
    print(f"\n{'='*60}")
    print(f"🔍 شريحة z={slice_index}")
    print(f"{'='*60}")

    raw = load_single_slice(nhdr_path, slice_index)
    print(f"Shape: {raw.shape}, dtype: {raw.dtype}")
    print(f"Raw range: {raw.min()} - {raw.max()}, mean={raw.mean():.1f}, median={np.median(raw):.1f}")

    texture = local_std(raw, window=9)
    print(f"Texture range: {texture.min():.2f} - {texture.max():.2f}, "
          f"mean={texture.mean():.2f}, median={np.median(texture):.2f}")

    # ---- Otsu على خريطة التكستشر (الخطوة 2 من البروتوكول) ----
    otsu_th = threshold_otsu(texture)
    mask = texture > otsu_th
    coverage = mask.sum() / mask.size * 100
    labeled, n_comp = label(mask)
    if n_comp > 0:
        sizes_cc = np.bincount(labeled.ravel())[1:]
        largest = sizes_cc.max()
        largest_pct = largest / mask.sum() * 100
    else:
        largest, largest_pct = 0, 0

    print(f"🎯 Otsu threshold = {otsu_th:.2f}")
    print(f"   تغطية القناع: {coverage:.1f}%")
    print(f"   عدد المكوّنات: {n_comp}")
    print(f"   أكبر مكوّن: {largest_pct:.1f}% من القناع")
    verdict = "⚠️ فشل محتمل (مكوّن عملاق - زي S4)" if (n_comp <= 3 and largest_pct > 70) else "✅ محتمل ناجح (زي S8) - يحتاج فحص بصري"
    print(f"   الحكم الأولي: {verdict}")

    # ---- الرسومات ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(raw, cmap='gray')
    axes[0].set_title(f'Raw z={slice_index} (I-0)')
    axes[1].imshow(texture, cmap='inferno')
    axes[1].set_title('Texture map (std 9x9)')
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title(f'Otsu mask (th={otsu_th:.1f}, {coverage:.1f}%)')
    plt.tight_layout()
    img_path = os.path.join(out_dir, f"s1_slice_{slice_index:04d}_test.png")
    plt.savefig(img_path, dpi=130)
    plt.close()
    print(f"💾 {img_path}")

    # ---- هيستوغرام التكستشر (يفيد لاحقًا بالمسح اليدوي) ----
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(texture.ravel(), bins=200, log=True)
    ax2.axvline(otsu_th, color='r', linestyle='--', label=f'Otsu={otsu_th:.1f}')
    ax2.set_title(f'Texture histogram z={slice_index} (I-0)')
    ax2.legend()
    hist_path = os.path.join(out_dir, f"s1_slice_{slice_index:04d}_hist.png")
    plt.savefig(hist_path, dpi=130)
    plt.close()
    print(f"💾 {hist_path}")

    return otsu_th, coverage, n_comp, largest_pct


if __name__ == "__main__":
    nhdr_path = "Salicornia 1 scan 2.nhdr"
    out_dir = "s1_texture_test_output"
    os.makedirs(out_dir, exist_ok=True)

    # نختار: طرف قريب من البداية، منتصف، طرف قريب من النهاية (Z حجمها 1000)
    test_slices = [30, 500, 970]

    summary = []
    for z in test_slices:
        otsu_th, coverage, n_comp, largest_pct = analyze_slice(nhdr_path, z, out_dir)
        summary.append((z, otsu_th, coverage, n_comp, largest_pct))

    print(f"\n{'='*60}")
    print("📋 ملخص جميع الشرائح")
    print(f"{'='*60}")
    print(f"{'z':>6} | {'Otsu':>8} | {'coverage%':>10} | {'n_comp':>7} | {'largest%':>9}")
    for z, th, cov, n, lp in summary:
        print(f"{z:>6} | {th:>8.1f} | {cov:>10.1f} | {n:>7} | {lp:>9.1f}")

    print("\n✅ راجع الصور بمجلد:", out_dir)
    print("👉 بناءً على 'largest%' نقرر: إذا فوق 70% بمعظم الشرائح → نروح لمسح threshold يدوي (بروتوكول S4)")