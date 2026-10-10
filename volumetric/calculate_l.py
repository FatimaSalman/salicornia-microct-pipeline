#!/usr/bin/env python3
"""
calculate_l.py
==============
Derived pc / pt quantities from Ján's single-slice manual Fiji measurements.

Conventions (Ján's table):
  cc  = FULL diameter (µm)
  pc  = one-sided thickness of parenchyma cortex (µm)
  pt  = one-sided thickness of palisade tissue (µm)

  R_cc     = cc / 2
  R_pc_out = R_cc + pc
  R_pt_out = R_pc_out + pt

Areas (circular-equivalent assumption):
  A_cc = pi * R_cc^2
  A_pc = pi * (R_pc_out^2 - R_cc^2)
  A_pt = pi * (R_pt_out^2 - R_pc_out^2)

Per-sample analysed length L is INFERRED from Table 2 of the manuscript:
  L = V_cc / (pi * R_eq^2)
(to be verified against the actual number of slices analysed).

Derived volumes = A * L. They are NOT comparable across samples because L
differs between samples (analysis length, not biology). Use areas and
within-sample ratios as primary numbers.
"""
import csv
import numpy as np

VOXEL_UM = 2.5
OUT_CSV = "pc_pt_derived_table.csv"

# Manual measurements (Ján) + Table 2 values
samples = {
    "I-0":    dict(cc=176.5, pc=153.8, pt=81.9,  V_cc=0.0735, R_eq=98.2),
    "I-1000": dict(cc=643.2, pc=512.0, pt=147.2, V_cc=0.7180, R_eq=338.0),
    "C-0":    dict(cc=357.9, pc=150.4, pt=109.0, V_cc=0.2660, R_eq=186.8),
    "C-1000": dict(cc=841.3, pc=271.8, pt=210.9, V_cc=1.5131, R_eq=442.0),
}

res = {}
for name, s in samples.items():
    # inferred analysed length
    L = s["V_cc"] * 1e9 / (np.pi * s["R_eq"] ** 2)      # mm^3 -> um^3, / um^2 -> um
    n_slices = L / VOXEL_UM

    # radii
    r_cc = s["cc"] / 2.0
    r_pc = r_cc + s["pc"]
    r_pt = r_pc + s["pt"]

    # areas (um^2)
    A_cc = np.pi * r_cc ** 2
    A_pc = np.pi * (r_pc ** 2 - r_cc ** 2)
    A_pt = np.pi * (r_pt ** 2 - r_pc ** 2)

    # sanity: the three areas must add up to the total disc
    assert np.isclose(A_cc + A_pc + A_pt, np.pi * r_pt ** 2), "area sum mismatch"

    # derived volumes (mm^3)
    V_cc_manual = A_cc * L / 1e9
    V_pc = A_pc * L / 1e9
    V_pt = A_pt * L / 1e9

    res[name] = dict(
        L=L, n_slices=n_slices,
        R_cc=r_cc, R_pc_out=r_pc, R_pt_out=r_pt,
        A_cc=A_cc, A_pc=A_pc, A_pt=A_pt,
        V_cc_manual=V_cc_manual, V_cc_table2=s["V_cc"],
        V_pc=V_pc, V_pt=V_pt,
        pc_over_cc=A_pc / A_cc,
        pt_over_cc=A_pt / A_cc,
        total_diam=2.0 * r_pt,
    )

# ---------------- printout ----------------
print("=" * 104)
print("1) Inferred analysed length L (verify against actual slice count!)")
print("=" * 104)
print(f"{'sample':8s} {'L (um)':>9s} {'slices@2.5um':>14s}")
for n, r in res.items():
    print(f"{n:8s} {r['L']:9.0f} {r['n_slices']:14.1f}")

print("\n" + "=" * 104)
print("2) Radii, areas and derived volumes")
print("=" * 104)
print(f"{'sample':8s} {'R_cc':>7s} {'R_pc,out':>9s} {'R_pt,out':>9s} | "
      f"{'A_cc':>9s} {'A_pc':>9s} {'A_pt':>9s} (um^2) | "
      f"{'V_cc*':>7s} {'V_pc':>7s} {'V_pt':>7s} (mm^3)")
print("-" * 104)
for n, r in res.items():
    print(f"{n:8s} {r['R_cc']:7.1f} {r['R_pc_out']:9.1f} {r['R_pt_out']:9.1f} | "
          f"{r['A_cc']:9.0f} {r['A_pc']:9.0f} {r['A_pt']:9.0f}         | "
          f"{r['V_cc_manual']:7.4f} {r['V_pc']:7.4f} {r['V_pt']:7.4f}")
print("* V_cc recomputed from the MANUAL diameter (self-check); it differs from "
      "Table 2 by the (manual/computed diameter)^2 ratio.")

print("\n" + "=" * 104)
print("3) Within-sample ratios and total stem diameter (PRIMARY descriptive numbers)")
print("=" * 104)
print(f"{'sample':8s} {'A_pc/A_cc':>10s} {'A_pt/A_cc':>10s} {'total diam (um)':>16s}")
for n, r in res.items():
    print(f"{n:8s} {r['pc_over_cc']:10.2f} {r['pt_over_cc']:10.2f} {r['total_diam']:16.0f}")

print("\n" + "=" * 104)
print("4) Salt / control ratios (n = 1 per condition, DESCRIPTIVE ONLY - not inferential)")
print("=" * 104)
for a, b in (("I-1000", "I-0"), ("C-1000", "C-0")):
    ra, rb = res[a], res[b]
    print(f"{a}/{b}:  A_cc={ra['A_cc']/rb['A_cc']:.2f}  "
          f"A_pc={ra['A_pc']/rb['A_pc']:.2f}  "
          f"A_pt={ra['A_pt']/rb['A_pt']:.2f}  "
          f"total_diam={ra['total_diam']/rb['total_diam']:.2f}")
print("NOTE: sites differ (Inowroclaw vs Ciechocinek) and plants are single "
      "individuals; do not present as a salinity effect.")

# ---------------- CSV export ----------------
cols = ["sample", "L_um", "slices_at_2.5um", "R_cc_um", "R_pc_out_um", "R_pt_out_um",
        "A_cc_um2", "A_pc_um2", "A_pt_um2", "V_pc_mm3", "V_pt_mm3",
        "A_pc_over_A_cc", "A_pt_over_A_cc", "total_diam_um"]
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for n, r in res.items():
        w.writerow([n, f"{r['L']:.1f}", f"{r['n_slices']:.1f}",
                    f"{r['R_cc']:.2f}", f"{r['R_pc_out']:.2f}", f"{r['R_pt_out']:.2f}",
                    f"{r['A_cc']:.0f}", f"{r['A_pc']:.0f}", f"{r['A_pt']:.0f}",
                    f"{r['V_pc']:.4f}", f"{r['V_pt']:.4f}",
                    f"{r['pc_over_cc']:.3f}", f"{r['pt_over_cc']:.3f}",
                    f"{r['total_diam']:.1f}"])
print(f"\nSaved table to {OUT_CSV}")