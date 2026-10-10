#!/usr/bin/env python3
"""vessel_ring_median_test_S5.py
تصحيح: استخدام median بدل mean، واختبار تصفية outliers (استبعاد أبعد
20% من الأوعية) لعزل حلقة cc الأساسية عن الأوعية الشاردة الأبعد."""

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

    z_list = np.linspace(50, nz-50, 15).astype(int)  # عينة أوسع للتأكد
    VESSEL_THR = 1800
    ROI_RADIUS = 180

    methods_results = {'mean_all': [], 'median_all': [],
                        'mean_trim20': [], 'median_trim20': [],
                        'mean_iqr': []}

    print(f"{'z':>5} | {'n_ves':>6} | {'mean':>6} | {'median':>7} | {'mean_tr20':>10} | {'med_tr20':>9} | {'mean_iqr':>9}")
    print("-"*80)

    for zc in z_list:
        accum = np.zeros((ny, nx)); count = 0
        for z in range(zc-half_window, zc+half_window+1):
            if 0 <= z < nz:
                accum += load_single_slice(nhdr_path, z, sizes); count += 1
        den = accum/count

        texture = local_std(den, 9)
        cy_b, cx_b = estimate_center_blob(texture, approx_cy, approx_cx)
        vessels = detect_vessels(texture, VESSEL_THR, (cy_b, cx_b), ROI_RADIUS)

        if len(vessels) < 5:
            continue

        r = np.array([np.hypot(cy_v-cy_b, cx_v-cx_b) for cy_v, cx_v, s in vessels])

        mean_all = r.mean()
        median_all = np.median(r)

        # تصفية 20% (استبعاد أعلى وأدنى 10% كل طرف - trimmed mean)
        r_sorted = np.sort(r)
        trim_n = int(len(r) * 0.1)
        r_trimmed = r_sorted[trim_n:len(r)-trim_n] if trim_n > 0 else r_sorted
        mean_trim20 = r_trimmed.mean()
        median_trim20 = np.median(r_trimmed)

        # IQR-based outlier removal
        q1, q3 = np.percentile(r, [25, 75])
        iqr = q3 - q1
        r_iqr = r[(r >= q1 - 1.5*iqr) & (r <= q3 + 1.5*iqr)]
        mean_iqr = r_iqr.mean() if len(r_iqr) > 0 else np.nan

        print(f"{zc:5d} | {len(vessels):6d} | {mean_all:6.1f} | {median_all:7.1f} | "
              f"{mean_trim20:10.1f} | {median_trim20:9.1f} | {mean_iqr:9.1f}")

        methods_results['mean_all'].append(mean_all)
        methods_results['median_all'].append(median_all)
        methods_results['mean_trim20'].append(mean_trim20)
        methods_results['median_trim20'].append(median_trim20)
        methods_results['mean_iqr'].append(mean_iqr)

    print(f"\n{'='*70}")
    print(f"📊 ملخص إحصائي (n={len(methods_results['mean_all'])} شريحة):")
    print(f"{'Method':<16}{'mean':>8}{'std':>8}{'std%':>8}{'diff_vs_Fiji%':>16}")
    for m, vals in methods_results.items():
        arr = np.array(vals)
        arr = arr[~np.isnan(arr)]
        mean, std = arr.mean(), arr.std()
        diff = (mean-FIJI_R)/FIJI_R*100
        print(f"{m:<16}{mean:8.1f}{std:8.1f}{std/mean*100:7.1f}%{diff:15.1f}%")