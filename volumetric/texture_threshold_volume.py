#!/usr/bin/env python3
"""
texture_threshold_volume.py
===========================
Texture-based three-dimensional threshold segmentation of the central
cylinder, as described in Materials and methods for the Inowroclaw
1000 mM and Ciechocinek 1000 mM samples.

WHY THIS SCRIPT EXISTS
----------------------
For these two samples the thresholding step was originally carried out
interactively in the 3D Slicer Segment Editor / Python console, so no
script existed in the repository. This file reproduces that step exactly
and non-interactively, so the published volumes can be regenerated from
the archived texture volumes by anyone.

Pipeline (per slice, matching the manuscript):
    texture volume  ->  threshold  ->  binary closing (radius 2)
                    ->  largest connected component
                    ->  binary_fill_holes
    volume = n_voxels * voxel_size^3

Usage
-----
    # Inowroclaw 1000 mM  (fixed threshold)
    python3 texture_threshold_volume.py \
        --texture-nhdr Salicornia_4_texture.nhdr \
        --threshold 3500 --label-out S4_cc_labelmap.npy

    # Ciechocinek 1000 mM (Otsu on the texture volume, as in Slicer)
    python3 texture_threshold_volume.py \
        --texture-nhdr Salicornia_8_texture.nhdr \
        --threshold otsu

    # reproduce Table 2 exactly, then compare:
    #   I-1000 published V = 0.7180 mm3
    #   C-1000 published V = 1.5131 mm3,  Slicer Otsu = 2841.73
"""
import argparse
import os
import time

import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops


# --------------------------------------------------------------------------
def parse_nhdr(nhdr_path):
    info = {}
    with open(nhdr_path) as f:
        for line in f:
            line = line.strip()
            if ':' not in line or line.startswith('NRRD'):
                continue
            k, v = line.split(':', 1)
            info[k.strip()] = v.strip()
    return info


class VolReader:
    def __init__(self, nhdr_path):
        self.info = parse_nhdr(nhdr_path)
        self.nx, self.ny, self.nz = [int(v) for v in self.info['sizes'].split()]
        self.spacing_mm = float(self.info['spacings'].split()[0])
        self.slice_bytes = self.nx * self.ny * 2
        base = os.path.dirname(os.path.abspath(nhdr_path))
        self.path = os.path.join(base, self.info['data file'])
        self.fh = open(self.path, 'rb')

    def slice(self, z):
        self.fh.seek(z * self.slice_bytes)
        return np.frombuffer(self.fh.read(self.slice_bytes),
                             dtype=np.uint16).reshape(self.ny, self.nx)

    def close(self):
        self.fh.close()


def otsu_from_histogram(counts):
    """Otsu's threshold from a 65536-bin uint16 histogram (Otsu 1979)."""
    counts = counts.astype(np.float64)
    bins = np.arange(counts.size, dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    w0 = np.cumsum(counts)
    w1 = total - w0
    sum_total = np.cumsum(counts * bins)
    grand = sum_total[-1]
    valid = (w0 > 0) & (w1 > 0)
    mu0 = np.zeros_like(w0); mu1 = np.zeros_like(w1)
    mu0[valid] = sum_total[valid] / w0[valid]
    mu1[valid] = (grand - sum_total[valid]) / w1[valid]
    between = np.zeros_like(w0)
    between[valid] = w0[valid] * w1[valid] * (mu0[valid] - mu1[valid]) ** 2
    return float(bins[int(np.argmax(between))])


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--texture-nhdr', required=True)
    ap.add_argument('--threshold', required=True,
                    help='numeric value, or "otsu" to compute it from the volume')
    ap.add_argument('--closing-radius', type=int, default=2)
    ap.add_argument('--z-start', type=int, default=0)
    ap.add_argument('--z-end', type=int, default=-1)
    ap.add_argument('--label-out', default=None,
                    help='optional .npy labelmap of the segmented central cylinder')
    ap.add_argument('--csv-out', default=None)
    ap.add_argument('--published-mm3', type=float, default=None)
    a = ap.parse_args()

    rd = VolReader(a.texture_nhdr)
    z0 = a.z_start
    z1 = rd.nz if a.z_end < 0 else a.z_end + 1
    vox_um = rd.spacing_mm * 1000.0
    print(f"Texture volume : {rd.path}")
    print(f"Dimensions     : {rd.nx} x {rd.ny} x {rd.nz}")
    print(f"Voxel size     : {vox_um:.6f} um")
    print(f"Evaluated range: {z0}..{z1-1}  ({z1-z0} slices)")

    # ---- threshold ----
    if str(a.threshold).lower() == 'otsu':
        print("\nComputing Otsu threshold from the full texture histogram ...")
        hist = np.zeros(65536, dtype=np.int64)
        for z in range(z0, z1):
            hist += np.bincount(rd.slice(z).ravel(), minlength=65536)
        thr = otsu_from_histogram(hist)
        print(f"  Otsu threshold = {thr:.2f}")
    else:
        thr = float(a.threshold)
        print(f"\nFixed threshold = {thr:.2f}")

    # ---- segment ----
    st = np.ones((2 * a.closing_radius + 1,) * 2, bool)
    labelmap = np.zeros((rd.nz, rd.ny, rd.nx), np.uint8) if a.label_out else None
    n_vox = 0
    rows, sols = [], []
    t0 = time.time()

    for z in range(z0, z1):
        m = rd.slice(z) > thr
        if m.any():
            if a.closing_radius > 0:
                m = ndimage.binary_closing(m, structure=st)
            lab, n = ndimage.label(m)
            if n > 1:
                sizes = np.bincount(lab.ravel()); sizes[0] = 0
                m = lab == sizes.argmax()
            elif n == 0:
                m = np.zeros_like(m)
            m = ndimage.binary_fill_holes(m)
        area = int(m.sum())
        n_vox += area
        if area:
            pr = regionprops(label(m))
            s = max(pr, key=lambda p: p.area).solidity if pr else np.nan
        else:
            s = np.nan
        sols.append(s)
        rows.append((z, area, area * vox_um ** 2, s))
        if labelmap is not None:
            labelmap[z] = m
        if (z - z0) % 100 == 0:
            print(f"  z={z}/{z1-1}  area={area:>8}px  ({time.time()-t0:.0f}s)", flush=True)

    V_mm3 = n_vox * (vox_um ** 3) * 1e-9
    L_um = (z1 - z0) * vox_um
    R_eq = np.sqrt(V_mm3 * 1e9 / (np.pi * L_um))
    sols_a = np.asarray([s for s in sols if not np.isnan(s)])

    print("\n" + "=" * 56)
    print(f"CC VOLUME            = {V_mm3:.4f} mm3   ({n_vox:,} voxels)")
    print(f"Evaluated length L   = {L_um/1000:.4f} mm")
    print(f"Equivalent radius    = {R_eq:.2f} um   -> diameter {2*R_eq:.1f} um")
    if sols_a.size:
        print(f"Mean solidity        = {sols_a.mean():.3f}")
        print(f"Slices solidity<0.8  = {100*(sols_a < 0.8).mean():.1f}%   "
              f"(manuscript reports 68% for I-1000)")
    if a.published_mm3:
        d = 100 * (V_mm3 - a.published_mm3) / a.published_mm3
        print(f"\nPublished volume     = {a.published_mm3:.4f} mm3")
        print(f"Difference           = {d:+.2f}%   "
              f"{'OK - reproduces Table 2' if abs(d) < 1 else '<-- INVESTIGATE'}")
    print("=" * 56)

    if a.csv_out:
        import csv
        with open(a.csv_out, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['z', 'area_px', 'area_um2', 'solidity'])
            w.writerows(rows)
        print(f"Saved {a.csv_out}")
    if labelmap is not None:
        np.save(a.label_out, labelmap)
        print(f"Saved {a.label_out}")
    rd.close()


if __name__ == '__main__':
    main()
