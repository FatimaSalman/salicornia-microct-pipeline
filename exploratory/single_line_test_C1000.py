#!/usr/bin/env python3
"""
اختبار: مقارنة نصف القطر عبر خط أفقي واحد فقط 
مقابل الـ azimuthal average (المتوسط الدائري)
"""

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True, help='مسار ملف tif الخام')
    parser.add_argument('--slice', type=int, required=True, help='رقم الشريحة')
    parser.add_argument('--cx', type=float, required=True, help='إحداثي مركز x')
    parser.add_argument('--cy', type=float, required=True, help='إحداثي مركز y')
    parser.add_argument('--voxel-size-um', type=float, required=True)
    parser.add_argument('--window', type=int, default=9)
    parser.add_argument('--max-radius-um', type=float, default=650)
    parser.add_argument('--direction', type=str, default='right', 
                         choices=['right', 'left', 'up', 'down'],
                         help='اتجاه الخط: right/left/up/down من المركز')
    parser.add_argument('--save-plot', type=str, default=None)
    args = parser.parse_args()

    print(f"Loading slice {args.slice} from {args.raw} ...")
    with tifffile.TiffFile(args.raw) as tif:
        raw_slice = tif.pages[args.slice].asarray()
    print(f"Slice shape: {raw_slice.shape}, dtype: {raw_slice.dtype}")

    cx, cy = args.cx, args.cy
    voxel_size = args.voxel_size_um
    window = args.window
    max_radius_px = int(args.max_radius_um / voxel_size)

    row = int(round(cy))
    col = int(round(cx))

    # اختيار الخط حسب الاتجاه
    if args.direction == 'right':
        line_profile = raw_slice[row, col:col + max_radius_px]
    elif args.direction == 'left':
        line_profile = raw_slice[row, max(0, col - max_radius_px):col][::-1]
    elif args.direction == 'down':
        line_profile = raw_slice[row:row + max_radius_px, col]
    elif args.direction == 'up':
        line_profile = raw_slice[max(0, row - max_radius_px):row, col][::-1]

    print(f"Line length: {len(line_profile)} px, direction: {args.direction}")

    # حساب local std dev (windowed texture) على الخط 1D
    texture_1d = []
    for i in range(len(line_profile) - window):
        texture_1d.append(np.std(line_profile[i:i + window]))
    texture_1d = np.array(texture_1d)

    distances_um = np.arange(len(texture_1d)) * voxel_size

    # إيجاد أعلى قمة
    peak_idx = np.argmax(texture_1d)
    peak_r = distances_um[peak_idx]
    peak_val = texture_1d[peak_idx]

    print(f"\n{'='*50}")
    print(f"Peak along single {args.direction} line: r = {peak_r:.1f} µm "
          f"(texture = {peak_val:.1f})")
    print(f"{'='*50}\n")

    # رسم بياني للمقارنة البصرية
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(distances_um, texture_1d, 'b-', linewidth=1.5)
    ax.axvline(peak_r, color='red', linestyle='--', 
               label=f'Peak = {peak_r:.1f} µm')
    ax.set_xlabel('Distance from center (µm)')
    ax.set_ylabel('Local texture (std dev) — single line')
    ax.set_title(f'Single-line radial profile (direction: {args.direction})')
    ax.legend()
    ax.grid(alpha=0.3)

    if args.save_plot:
        plt.savefig(args.save_plot, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {args.save_plot}")
    else:
        plt.show()

if __name__ == '__main__':
    main()