
import os

# Root of the data tree. Override with:  export SALICORNIA_DATA=/path
DATA = os.environ.get("SALICORNIA_DATA", ".")

import tifffile
import numpy as np

pred = tifffile.imread(os.path.join(DATA, 'Salicornia 4, Inowroclaw 1000 mM/image_folder/Salicornia4_/MLresults/Salicornia4_fullstack_prediction.tif'))
print("Shape:", pred.shape)
print("Unique raw values:", np.unique(pred))

# voxel size: X,Y rescaled by 4 (2.5um -> 10um), Z unchanged (2.5um)
voxel_volume_mm3 = 10.0 * 10.0 * 2.5 * 1e-9

label_map = {
    0: "background",
    85: "pt",
    -86: "pc",     # true value 170
    -1: "cc_pi",   # true value 255
}

print(f"\n{'Tissue':<12} {'Voxels':>12} {'Volume (mm3)':>15}")
print("-" * 42)
for val, name in label_map.items():
    count = int((pred == val).sum())
    vol_mm3 = count * voxel_volume_mm3
    print(f"{name:<12} {count:>12} {vol_mm3:>15.4f}")