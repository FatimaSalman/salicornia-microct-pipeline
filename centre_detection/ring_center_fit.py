#!/usr/bin/env python3
"""
Stem center from the vascular ring, via radial "ring score" optimization.
No thresholds: find the centre that makes the texture ring strongest and most uniform in azimuth.
score = peak_mean_texture * (1 - angular_CV)
"""

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import argparse
from scipy.ndimage import uniform_filter

def local_std(img, w):
    imgf = img.astype(np.float64)
    m1 = uniform_filter(imgf, size=w)
    m2 = uniform_filter(imgf**2, size=w)
    return np.sqrt(np.maximum(m2 - m1**2, 0))

def bilinear_sample(img, xs, ys):
    H, W = img.shape
    x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int)
    x1 = np.clip(x0+1, 0, W-1);   y1 = np.clip(y0+1, 0, H-1)
    x0 = np.clip(x0, 0, W-1);     y0 = np.clip(y0, 0, H-1)
    fx = xs - x0; fy = ys - y0
    return (img[y0,x0]*(1-fx)*(1-fy) + img[y0,x1]*fx*(1-fy) +
            img[y1,x0]*(1-fx)*fy   + img[y1,x1]*fx*fy)

def ring_eval(tex, cx, cy, radii):
    th = np.linspace(0, 2*np.pi, 64, endpoint=False)
    cos, sin = np.cos(th), np.sin(th)
    means = np.array([bilinear_sample(tex, cx+r*cos, cy+r*sin).mean()
                      for r in radii])
    k = int(means.argmax()); r_best = radii[k]; peak = means[k]
    thf = np.linspace(0, 2*np.pi, 180, endpoint=False)
    vals = bilinear_sample(tex, cx+r_best*np.cos(thf), cy+r_best*np.sin(thf))
    cv = vals.std() / max(vals.mean(), 1e-9)
    score = peak * (1.0 - min(cv, 1.0))
    return score, r_best, peak, cv

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--grid', required=True)
    p.add_argument('--slice', type=int, required=True)
    p.add_argument('--cx0', type=float, required=True)
    p.add_argument('--cy0', type=float, required=True)
    p.add_argument('--rmin', type=int, default=40)
    p.add_argument('--rmax', type=int, default=140)
    p.add_argument('--search', type=int, default=30)
    p.add_argument('--window', type=int, default=9)
    p.add_argument('--save-plot', default=None)
    args = p.parse_args()

    with tifffile.TiffFile(args.grid) as tif:
        img = tif.pages[args.slice].asarray()
    print(f"Slice {args.slice}, shape {img.shape}, dtype {img.dtype}")

    tex = local_std(img, args.window)
    radii = np.arange(args.rmin, args.rmax + 1, 2, dtype=float)

    # ---- coarse search: grid of +/-search px, step 3 ----
    offs = np.arange(-args.search, args.search + 1, 3)
    best = (-1, None)
    for dy in offs:
        for dx in offs:
            s, r, pk, cv = ring_eval(tex, args.cx0+dx, args.cy0+dy, radii)
            if s > best[0]:
                best = (s, (args.cx0+dx, args.cy0+dy, r, pk, cv))
    s0, (cx, cy, r, pk, cv) = best
    print(f"Coarse: score={s0:.1f} at ({cx:.0f},{cy:.0f}), r={r:.0f}px, peak={pk:.1f}, CV={cv:.2f}")

    # ---- refinement: +/-3 px, step 1 ----
    best = (-1, None)
    for dy in np.arange(-3, 4, 1):
        for dx in np.arange(-3, 4, 1):
            s, r, pk, cv = ring_eval(tex, cx+dx, cy+dy, radii)
            if s > best[0]:
                best = (s, (cx+dx, cy+dy, r, pk, cv))
    s, (cx, cy, r, pk, cv) = best

    print("\n=== RESULT (ring-score fit) ===")
    print(f"Center (x, y)   = ({cx:.2f}, {cy:.2f})")
    print(f"Ring radius     = {r:.0f} px  ({r*2.5:.0f} µm)")
    print(f"Peak / CV       = {pk:.1f} / {cv:.2f}   (low CV = azimuthally complete ring)")
    print(f"Offset from prim= ({cx-args.cx0:+.1f}, {cy-args.cy0:+.1f}) px")

    if args.save_plot:
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(tex, cmap='magma')
        th = np.linspace(0, 2*np.pi, 300)
        ax.plot(cx+r*np.cos(th), cy+r*np.sin(th), 'w--', lw=1.2, alpha=0.8)
        ax.plot(cx, cy, '+', color='yellow', ms=16, mew=2.5)
        ax.plot(args.cx0, args.cy0, 'x', color='cyan', ms=12, mew=2)
        ax.set_title(f'Ring fit: ({cx:.0f},{cy:.0f}) r={r:.0f}px CV={cv:.2f}')
        plt.savefig(args.save_plot, dpi=150, bbox_inches='tight')
        print(f"Saved: {args.save_plot}")

if __name__ == '__main__':
    main()