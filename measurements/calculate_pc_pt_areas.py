#!/usr/bin/env python3
"""
calculate_pc_pt_areas.py
========================
Derived parenchyma cortex (pc) and palisade tissue (pt) quantities from the
single-slice manual measurements, reported in Table 5.

Conventions
-----------
  cc  full diameter of the central cylinder (um)
  pc  one-sided thickness of the parenchyma cortex (um)
  pt  one-sided thickness of the palisade tissue (um)

  R_cc     = cc / 2
  R_pc_out = R_cc + pc
  R_pt_out = R_pc_out + pt

Areas, under a circular-equivalent assumption:

  A_cc = pi * R_cc^2
  A_pc = pi * (R_pc_out^2 - R_cc^2)
  A_pt = pi * (R_pt_out^2 - R_pc_out^2)

The measurements are taken on one slice per sample and are DESCRIPTIVE. The
samples come from two collection sites and were acquired with two different
protocols, with one individual per condition; the ratios below must not be
read as a salinity effect.

Analysed length L is the number of evaluated slices times the voxel size, not
an inferred quantity. The volume recomputed from the manual diameter is
reported only as a consistency check against Table 2.
"""
import csv
import numpy as np

VOXEL_UM = 2.5
OUT_CSV = "pc_pt_derived_table.csv"

# Manual measurements (final), with the Table 2 values for cross-checking.
# measurement slices, 0-based: I-0 522, I-1000 572, C-0 454, C-1000 750
samples = {
    "I-0":    dict(cc=179.7, pc=183.5, pt=109.1,
                   pc_sd=52.6, pt_sd=18.9,
                   V_cc=0.0735, R_eq=98.2,  n_slices=970),
    "I-1000": dict(cc=678.4, pc=533.5, pt=212.6,
                   pc_sd=65.6, pt_sd=33.8,
                   V_cc=0.7180, R_eq=338.0, n_slices=800),
    "C-0":    dict(cc=386.2, pc=131.4, pt=90.2,
                   pc_sd=31.4, pt_sd=8.1,
                   V_cc=0.2782, R_eq=191.0, n_slices=971),
    "C-1000": dict(cc=902.7, pc=298.1, pt=162.9,
                   pc_sd=68.7, pt_sd=12.3,
                   V_cc=1.5131, R_eq=442.3, n_slices=985),
}

res = {}
for name, s in samples.items():
    L = s["n_slices"] * VOXEL_UM                     # analysed length, um
    L_inferred = s["V_cc"] * 1e9 / (np.pi * s["R_eq"] ** 2)

    r_cc = s["cc"] / 2.0
    r_pc = r_cc + s["pc"]
    r_pt = r_pc + s["pt"]

    A_cc = np.pi * r_cc ** 2
    A_pc = np.pi * (r_pc ** 2 - r_cc ** 2)
    A_pt = np.pi * (r_pt ** 2 - r_pc ** 2)
    A_total = np.pi * r_pt ** 2
    assert np.isclose(A_cc + A_pc + A_pt, A_total), "area sum mismatch"

    res[name] = dict(
        L=L, L_inferred=L_inferred,
        R_cc=r_cc, R_pc_out=r_pc, R_pt_out=r_pt,
        A_cc=A_cc, A_pc=A_pc, A_pt=A_pt,
        V_cc_manual=A_cc * L / 1e9, V_cc_table2=s["V_cc"],
        V_pc=A_pc * L / 1e9, V_pt=A_pt * L / 1e9,
        pc_over_cc=A_pc / A_cc, pt_over_cc=A_pt / A_cc,
        cc_share=A_cc / A_total,
        total_diam=2.0 * r_pt,
        pc_sd=s["pc_sd"], pt_sd=s["pt_sd"],
    )

line = "=" * 92

print(line)
print("Table 5 as reported")
print(line)
print(f"{'sample':8s} {'cc':>7s} {'pc':>14s} {'pt':>14s} "
      f"{'A_pc/A_cc':>10s} {'A_pt/A_cc':>10s} {'cc share':>9s} {'outer d':>8s}")
for n, r in res.items():
    s = samples[n]
    print(f"{n:8s} {s['cc']:7.1f} {s['pc']:7.1f}+-{r['pc_sd']:<5.1f} "
          f"{s['pt']:7.1f}+-{r['pt_sd']:<5.1f} "
          f"{r['pc_over_cc']:10.2f} {r['pt_over_cc']:10.2f} "
          f"{r['cc_share']*100:8.1f}% {r['total_diam']:8.0f}")

print("\n" + line)
print("Radii and areas")
print(line)
print(f"{'sample':8s} {'R_cc':>7s} {'R_pc,out':>9s} {'R_pt,out':>9s} | "
      f"{'A_cc':>10s} {'A_pc':>10s} {'A_pt':>10s}   (um^2)")
for n, r in res.items():
    print(f"{n:8s} {r['R_cc']:7.1f} {r['R_pc_out']:9.1f} {r['R_pt_out']:9.1f} | "
          f"{r['A_cc']:10.0f} {r['A_pc']:10.0f} {r['A_pt']:10.0f}")

print("\n" + line)
print("Consistency check: analysed length, and volume from the manual diameter")
print(line)
print(f"{'sample':8s} {'L (um)':>9s} {'L inferred':>11s} {'diff':>7s} | "
      f"{'V manual':>9s} {'V Table 2':>10s} {'diff':>8s}")
for n, r in res.items():
    dL = (r['L_inferred'] - r['L']) / r['L'] * 100
    dV = (r['V_cc_manual'] - r['V_cc_table2']) / r['V_cc_table2'] * 100
    print(f"{n:8s} {r['L']:9.0f} {r['L_inferred']:11.0f} {dL:6.1f}% | "
          f"{r['V_cc_manual']:9.4f} {r['V_cc_table2']:10.4f} {dV:7.1f}%")
print("The volume recomputed from the manual diameter differs from Table 2 by")
print("the square of the manual-to-computed diameter ratio; this is the same")
print("discrepancy reported in Table 2 and is expected.")

print("\n" + line)
print("Ratios between conditions (n = 1 per condition, DESCRIPTIVE ONLY)")
print(line)
for a, b in (("I-1000", "I-0"), ("C-1000", "C-0")):
    ra, rb = res[a], res[b]
    print(f"{a}/{b}:  A_cc {ra['A_cc']/rb['A_cc']:.2f}  "
          f"A_pc {ra['A_pc']/rb['A_pc']:.2f}  "
          f"A_pt {ra['A_pt']/rb['A_pt']:.2f}  "
          f"outer diameter {ra['total_diam']/rb['total_diam']:.2f}")
print("Collection sites and acquisition protocols differ between the two")
print("populations, and each condition is a single individual. These ratios")
print("are not evidence of a salinity effect.")

cols = ["sample", "cc_um", "pc_um", "pc_sd_um", "pt_um", "pt_sd_um",
        "L_um", "n_slices", "R_cc_um", "R_pc_out_um", "R_pt_out_um",
        "A_cc_um2", "A_pc_um2", "A_pt_um2", "V_pc_mm3", "V_pt_mm3",
        "A_pc_over_A_cc", "A_pt_over_A_cc", "cc_share", "outer_diam_um"]
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for n, r in res.items():
        s = samples[n]
        w.writerow([n, f"{s['cc']:.1f}", f"{s['pc']:.1f}", f"{r['pc_sd']:.1f}",
                    f"{s['pt']:.1f}", f"{r['pt_sd']:.1f}",
                    f"{r['L']:.1f}", s["n_slices"],
                    f"{r['R_cc']:.2f}", f"{r['R_pc_out']:.2f}", f"{r['R_pt_out']:.2f}",
                    f"{r['A_cc']:.0f}", f"{r['A_pc']:.0f}", f"{r['A_pt']:.0f}",
                    f"{r['V_pc']:.4f}", f"{r['V_pt']:.4f}",
                    f"{r['pc_over_cc']:.3f}", f"{r['pt_over_cc']:.3f}",
                    f"{r['cc_share']:.4f}", f"{r['total_diam']:.1f}"])
print(f"\nSaved {OUT_CSV}")
