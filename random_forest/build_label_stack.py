import tifffile
import numpy as np
import os

# --- Fixed parameters ---
com_y, com_x = 520, 591          # full-frame coordinates (not crop-relative)
voxel_um = 2.5

r_cc_pc = 87.5 / voxel_um        # cc/pc boundary radius, in voxels
r_pc_pt = 350 / voxel_um         # pc/pt boundary radius, in voxels
r_pt_out = 450 / voxel_um        # pt/outside boundary radius, in voxels

reference_slices = [80, 200, 350, 500, 650, 800]  # spread across 1000 slices

output_dir = "image_folder/Salicornia1_"
os.makedirs(output_dir, exist_ok=True)

label_slices = []
with tifffile.TiffFile("Salicornia_1_stack.tif") as tif:
    for z in reference_slices:
        s_img = tif.pages[z].asarray().astype(float)

        # Build the radius map fresh for THIS slice's actual shape,
        # rather than reusing a crop-sized r from a different slice.
        yy, xx = np.mgrid[0:s_img.shape[0], 0:s_img.shape[1]]
        r_full = np.sqrt((yy - com_y) ** 2 + (xx - com_x) ** 2)

        lbl = np.zeros(s_img.shape, dtype=np.uint8)
        lbl[r_full <= r_cc_pc] = 153
        lbl[(r_full > r_cc_pc) & (r_full <= r_pc_pt)] = 102
        lbl[(r_full > r_pc_pt) & (r_full <= r_pt_out)] = 51
        label_slices.append(lbl)
        print(f"Processed slice {z}")

label_stack = np.stack(label_slices, axis=0)
print("Shape:", label_stack.shape, "Unique:", np.unique(label_stack))

out_path = os.path.join(output_dir, "labelled-stack.tif")
tifffile.imwrite(out_path, label_stack, photometric='minisblack')
print(f"Saved {out_path}")