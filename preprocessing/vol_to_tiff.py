#!/usr/bin/env python3
"""Convert a .vol (+ .nhdr header) volume into a TIFF stack for leaf-traits-microct."""
import numpy as np
import tifffile
import os
import sys

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

def load_vol(nhdr_path):
    info = parse_nhdr(nhdr_path)
    sizes = [int(x) for x in info['sizes'].split()]  # X Y Z order in NRRD
    dtype_map = {
        'unsigned char': np.uint8, 'signed char': np.int8,
        'unsigned short': np.uint16, 'short': np.int16,
        'unsigned int': np.uint32, 'int': np.int32,
        'float': np.float32, 'double': np.float64,
    }
    dtype = dtype_map[info['type']]
    endian = '<' if info.get('endian', 'little') == 'little' else '>'
    data_file = info['data file']
    vol_path = os.path.join(os.path.dirname(nhdr_path), data_file)

    print(f"Loading {vol_path} ...")
    print(f"Sizes (X Y Z): {sizes}, dtype: {dtype}")

    arr = np.fromfile(vol_path, dtype=np.dtype(dtype).newbyteorder(endian))
    expected = sizes[0] * sizes[1] * sizes[2]
    if arr.size != expected:
        print(f"WARNING: expected {expected} voxels, got {arr.size}")
    # NRRD sizes are X Y Z (fastest-varying first); reshape then transpose to Z,Y,X for TIFF stack
    arr = arr.reshape(sizes[2], sizes[1], sizes[0])  # already Z,Y,X since numpy is C-order (last axis fastest)
    return arr

if __name__ == "__main__":
    nhdr_path = sys.argv[1] if len(sys.argv) > 1 else "Salicornia_8.nhdr"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "Salicornia_8_stack.tif"

    arr = load_vol(nhdr_path)
    print(f"Final array shape (Z,Y,X): {arr.shape}, dtype: {arr.dtype}")
    print(f"Value range: {arr.min()} - {arr.max()}")

    tifffile.imwrite(out_path, arr, photometric='minisblack')
    print(f"Saved TIFF stack to {out_path}")