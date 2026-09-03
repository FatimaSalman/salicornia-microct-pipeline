import tifffile
import numpy as np

reference_slices = [50, 130, 220, 310, 400, 500, 600, 700, 780, 850, 920, 970]  # 12 slices spread across 1000

r_cc_pc = 90 / 2.5
r_pc_pt = 190 / 2.5
r_pt_out = 250 / 2.5

com_y, com_x = 520, 591
yy, xx = np.mgrid[0:900, 0:900]
r = np.sqrt((yy-com_y)**2 + (xx-com_x)**2)

label_slices = []
with tifffile.TiffFile("/path/to/Salicornia_1_stack.tif") as tif:
    for z in reference_slices:
        s_img = tif.pages[z].asarray().astype(float)
        lbl = np.zeros(s_img.shape, dtype=np.uint8)
        lbl[r <= r_cc_pc] = 153
        lbl[(r > r_cc_pc) & (r <= r_pc_pt)] = 102
        lbl[(r > r_pc_pt) & (r <= r_pt_out)] = 51
        label_slices.append(lbl)
        print(f"Processed slice {z}")

label_stack = np.stack(label_slices, axis=0)
print("Shape:", label_stack.shape, "Unique:", np.unique(label_stack))

tifffile.imwrite("/path/to/image_folder/Salicornia1_/labelled-stack.tif", label_stack, photometric='minisblack')
print("Saved labelled-stack.tif (12 slices)")