#!/usr/bin/env python3
"""vessel_ring_detection_S5.py
اكتشاف الأوعية الفردية (xylem vessels) كمكوّنات متصلة عند threshold عالٍ،
ثم حساب توزيع بُعدها عن المركز - بديل أكثر متانة من radial profile
averaging عندما تكون البنية عبارة عن نقاط متفرقة وليست حلقة متصلة."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, label, center_of_mass
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

def estimate_center_blob(texture_img, approx_cy, approx_cx, search_radius=100):
    ny, nx = texture_img.shape
    y0, y1 = max(0, int(approx_cy-search_radius)), min(ny, int(approx_cy+search_radius))
    x0, x1 = max(0, int(approx_cx-search_radius)), min(nx, int(approx_cx+search_radius))
    region = texture_img[y0:y1, x0:x1]
    mask = region >= np.percentile(region, 80)
    ys, xs = np.nonzero(mask)
    w = region[ys, xs]
    return np.average(ys, weights=w)+y0, np.average(xs, weights=w)+x0

def detect_vessels(texture_img, threshold, roi_center, roi_radius, min_size=5, max_size=500):
    """يكتشف الأوعية كمكوّنات متصلة ضمن ROI دائري، يستبعد الضجيج الصغير
    جداً والكتل الكبيرة جداً (احتمال pith أو دمج خاطئ)."""
    ny, nx = texture_img.shape
    cy, cx = roi_center
    yy, xx = np.mgrid[0:ny, 0:nx]
    roi_mask = np.sqrt((yy-cy)**2 + (xx-cx)**2) < roi_radius

    binary = (texture_img > threshold) & roi_mask
    labeled, n_comp = label(binary)

    vessel_centroids = []
    for comp_id in range(1, n_comp+1):
        size = np.sum(labeled == comp_id)
        if min_size <= size <= max_size:
            cy_v, cx_v = center_of_mass(labeled == comp_id)
            vessel_centroids.append((cy_v, cx_v, size))
    return vessel_centroids

if __name__ == "__main__":
    nhdr_path = "Salicornia_5.nhdr"
    sizes = [int(x) for x in parse_nhdr(nhdr_path)['sizes'].split()]
    nx, ny, nz = sizes
    FIJI_R = 71.6
    approx_cy, approx_cx = 321.8, 318.8
    half_window = 15

    z_list = [150, 300, 501, 700, 850]
    VESSEL_THR = 1800  # من المسح السابق - نطاق مستقر نسبياً (n_comp~30)
    ROI_RADIUS = 180

    fig, axes = plt.subplots(2, len(z_list), figsize=(4.5*len(z_list), 9))

    print(f"{'z':>5} | {'n_vessels':>10} | {'mean_r':>8} | {'median_r':>9} | {'std_r':>7} | {'diff_mean_vs_Fiji':>18}")
    print("-"*80)

    all_mean_r = []
    for col, zc in enumerate(z_list):
        accum = np.zeros((ny, nx)); count = 0
        for z in range(zc-half_window, zc+half_window+1):
            if 0 <= z < nz:
                accum += load_single_slice(nhdr_path, z, sizes); count += 1
        den = accum/count

        texture = local_std(den, 9)
        cy_b, cx_b = estimate_center_blob(texture, approx_cy, approx_cx)

        vessels = detect_vessels(texture, VESSEL_THR, (cy_b, cx_b), ROI_RADIUS)

        if len(vessels) == 0:
            print(f"{zc:5d} | لا توجد أوعية مكتشفة!")
            continue

        vessel_r = np.array([np.hypot(cy_v-cy_b, cx_v-cx_b) for cy_v, cx_v, s in vessels])
        mean_r = vessel_r.mean()
        median_r = np.median(vessel_r)
        std_r = vessel_r.std()
        diff = (mean_r - FIJI_R)/FIJI_R*100

        print(f"{zc:5d} | {len(vessels):10d} | {mean_r:8.1f} | {median_r:9.1f} | {std_r:7.1f} | {diff:+17.1f}%")
        all_mean_r.append(mean_r)

        # لوحة علوية: الأوعية المكتشفة فوق الصورة
        ax = axes[0, col]
        pad = 150
        ax.imshow(den, cmap='gray', vmin=np.percentile(den,1), vmax=np.percentile(den,99))
        for cy_v, cx_v, s in vessels:
            ax.plot(cx_v, cy_v, 'r.', markersize=6)
        circle_mean = plt.Circle((cx_b, cy_b), mean_r, color='lime', fill=False, lw=2, label=f'mean_r={mean_r:.0f}')
        circle_fiji = plt.Circle((cx_b, cy_b), FIJI_R, color='cyan', fill=False, lw=2, ls='--', label=f'Fiji={FIJI_R}')
        ax.add_patch(circle_mean); ax.add_patch(circle_fiji)
        ax.plot(cx_b, cy_b, 'y+', ms=12, mew=2)
        ax.set_xlim(cx_b-pad, cx_b+pad); ax.set_ylim(cy_b+pad, cy_b-pad)
        ax.set_title(f'z={zc}: {len(vessels)} vessels'); ax.legend(fontsize=8)

        # لوحة سفلية: توزيع أبعاد الأوعية (histogram)
        ax = axes[1, col]
        ax.hist(vessel_r, bins=15, color='steelblue', edgecolor='black')
        ax.axvline(mean_r, color='green', lw=2, label=f'mean={mean_r:.0f}')
        ax.axvline(FIJI_R, color='cyan', ls='--', lw=2, label='Fiji')
        ax.set_xlabel('distance from center (px)'); ax.legend(fontsize=8)
        ax.set_title(f'vessel radius distribution z={zc}')

    plt.tight_layout()
    plt.savefig('vessel_ring_detection_S5.png', dpi=150)
    plt.show()

    if all_mean_r:
        arr = np.array(all_mean_r)
        print(f"\n📊 ملخص (n={len(arr)} شرائح):")
        print(f"   mean of means = {arr.mean():.1f}px | std = {arr.std():.1f}px ({arr.std()/arr.mean()*100:.1f}%)")
        print(f"   diff vs Fiji = {(arr.mean()-FIJI_R)/FIJI_R*100:+.1f}%")
    print("\nSaved: vessel_ring_detection_S5.png")