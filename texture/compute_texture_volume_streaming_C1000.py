#!/usr/bin/env python3
"""compute_texture_volume_streaming.py
Fully memory-light version: reads and writes slice by slice directly from and to disk,
never holding the whole volume in RAM, to avoid the out-of-memory failure
that occurred with sitk.WriteImage on the full version."""

import numpy as np
from scipy.ndimage import uniform_filter
import os
import time

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

def write_nhdr(out_nhdr_path, sizes, spacing, data_file_name, dtype_str='unsigned short'):
    """Writes a new .nhdr identical to the original but pointing at the new texture file."""
    with open(out_nhdr_path, 'w') as f:
        f.write("NRRD0004\n")
        f.write(f"type: {dtype_str}\n")
        f.write("dimension: 3\n")
        f.write(f"sizes: {sizes[0]} {sizes[1]} {sizes[2]}\n")
        f.write(f"spacings: {spacing[0]} {spacing[1]} {spacing[2]}\n")
        f.write("endian: little\n")
        f.write("encoding: raw\n")
        f.write(f"data file: {data_file_name}\n")

def compute_texture_streaming(nhdr_path, window=9,
                                out_nhdr_name="Salicornia_8_texture.nhdr",
                                out_vol_name="Salicornia_8_texture.vol"):
    info = parse_nhdr(nhdr_path)
    sizes = [int(x) for x in info['sizes'].split()]  # X, Y, Z
    nx, ny, nz = sizes
    spacing = [float(x) for x in info['spacings'].split()]
    dtype = np.uint16
    slice_bytes = nx * ny * 2  # 2 bytes/voxel

    base_dir = os.path.dirname(nhdr_path) or '.'
    in_vol_path = os.path.join(base_dir, info['data file'])
    out_vol_path = os.path.join(base_dir, out_vol_name)
    out_nhdr_path = os.path.join(base_dir, out_nhdr_name)

    print(f"Input volume : {in_vol_path}")
    print(f"Output volume: {out_vol_path}")
    print(f"Dimensions (X,Y,Z): {nx},{ny},{nz}  |  window={window}x{window}")
    print("Processing slice-by-slice (low memory mode)...")

    t0 = time.time()
    max_seen = 0.0

    with open(in_vol_path, 'rb') as fin, open(out_vol_path, 'wb') as fout:
        for z in range(nz):
            raw = fin.read(slice_bytes)
            s = np.frombuffer(raw, dtype=dtype).reshape(ny, nx).astype(np.float64)

            mean = uniform_filter(s, size=window)
            mean_sq = uniform_filter(s**2, size=window)
            var = np.maximum(mean_sq - mean**2, 0)
            tex = np.sqrt(var)

            max_seen = max(max_seen, tex.max())
            tex_u16 = np.clip(tex, 0, 65535).astype(np.uint16)
            fout.write(tex_u16.tobytes())

            # Explicit release (optional; Python garbage-collects anyway)
            del s, mean, mean_sq, var, tex, tex_u16

            if z % 100 == 0:
                print(f"  Slice {z}/{nz} done ({time.time()-t0:.1f}s elapsed)")

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print(f"Max texture value seen: {max_seen:.1f}")

    write_nhdr(out_nhdr_path, sizes=[nx, ny, nz], spacing=spacing,
               data_file_name=out_vol_name)
    print(f"\nDone. Load this file directly in Slicer:\n   {out_nhdr_path}")

if __name__ == "__main__":
    compute_texture_streaming("Salicornia_8.nhdr", window=9)