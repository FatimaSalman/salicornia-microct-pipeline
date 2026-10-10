#!/usr/bin/env python3
"""
npy_to_nrrd.py
==============
Converts the C-0 (S5) central-cylinder labelmap produced by the corrected
S5_full_volume_grad_method.py from .npy (Z, Y, X) to .nrrd (X, Y, Z), so it
can be used by build_rf_labels_from_final_masks.py in place of the old mask.

    python3 npy_to_nrrd.py \
        --npy  Salicornia_5/S5_cc_labelmap.npy \
        --like Salicornia_5/S5_cc_labelmap_correct.nrrd \
        --out  Salicornia_5/S5_cc_labelmap_bufferfix.nrrd
"""
import argparse
import numpy as np
import nrrd

ap = argparse.ArgumentParser()
ap.add_argument('--npy', required=True)
ap.add_argument('--like', required=True, help='existing .nrrd to copy the header from')
ap.add_argument('--out', required=True)
ap.add_argument('--voxel-um', type=float, default=2.50000064)
a = ap.parse_args()

m = np.load(a.npy)                      # (Z, Y, X)
print('npy shape (Z,Y,X):', m.shape, '| unique:', np.unique(m))
v = m.sum() * a.voxel_um ** 3 / 1e9
print(f'volume from npy    : {v:.4f} mm3')

_, hdr = nrrd.read(a.like)
old, oldhdr = nrrd.read(a.like)
print('reference nrrd shape (X,Y,Z):', old.shape)
print(f'volume of OLD mask : {old.sum() * a.voxel_um ** 3 / 1e9:.4f} mm3')

out = np.transpose(m, (2, 1, 0)).astype(old.dtype)   # (Z,Y,X) -> (X,Y,Z)
assert out.shape == old.shape, f'shape mismatch: {out.shape} vs {old.shape}'
out = (out > 0).astype(old.dtype)                    # ensure binary 0/1

nrrd.write(a.out, out, hdr)
chk, _ = nrrd.read(a.out)
print(f'volume of NEW mask : {chk.sum() * a.voxel_um ** 3 / 1e9:.4f} mm3')
print('unique values      :', np.unique(chk))
print('\nWrote', a.out)
print('Now point build_rf_labels_from_final_masks.py at this file for S5.')
