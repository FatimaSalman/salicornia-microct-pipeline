"""
prepare_bia_upload.py
Verify and (optionally) rename the raw microCT volumes before depositing them
in the BioImage Archive.

What it does
------------
1. Reads every .nhdr header and prints: dimensions, voxel spacing, data type,
   and the .vol file it points to.
2. Flags a spacing that is not 0.0025 mm  -- the error we already hit once.
3. Checks that the referenced .vol actually exists and that its size matches
   what the header implies.
4. With --rename, copies each sample into a clean upload tree named with the
   manuscript sample IDs, and rewrites the "data file:" line inside the .nhdr
   so the pair stays valid.

Dry run by default. Nothing is written unless you pass --rename.

Usage
-----
    python3 prepare_bia_upload.py  --root ~/Desktop/Salicornia_Project
    python3 prepare_bia_upload.py  --root ~/Desktop/Salicornia_Project --rename
"""
import argparse
import os
import re
import shutil
from pathlib import Path

# folder name in your project  ->  sample ID used in the manuscript
SAMPLE_MAP = {
    "Salicornia 1, Inowroclaw 0 mM": "I-0",
    "Salicornia 4, Inowroclaw 1000 mM": "I-1000",
    "Salicornia_5": "C-0",
    "Salicornia_8": "C-1000",
}

EXPECTED_SPACING_MM = 0.0025
REL_TOL = 0.01      # 1 %. The failure we guard against is a factor of 1000
                    # (micrometres written as millimetres), not the 8th decimal.

DTYPE_BYTES = {
    "short": 2, "signed short": 2, "int16": 2,
    "ushort": 2, "unsigned short": 2, "uint16": 2,
    "float": 4, "double": 8,
    "uchar": 1, "unsigned char": 1, "uint8": 1,
}


def parse_nhdr(path):
    """Return a dict of the NRRD header fields we care about."""
    info = {"path": path, "raw": {}}
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if ":=" in line:                       # key-value pair
                k, v = line.split(":=", 1)
            elif ":" in line:                      # field
                k, v = line.split(":", 1)
            else:
                continue
            info["raw"][k.strip()] = v.strip()
    r = info["raw"]

    info["sizes"] = [int(x) for x in r.get("sizes", "").split()] or None
    info["type"] = r.get("type")
    info["datafile"] = r.get("data file") or r.get("datafile")

    # spacing from "space directions" or from "spacings"
    spacing = None
    sd = r.get("space directions")
    if sd:
        vecs = re.findall(r"\(([^)]*)\)", sd)
        if vecs:
            spacing = []
            for v in vecs:
                comps = [float(c) for c in v.split(",")]
                spacing.append(max(abs(c) for c in comps))
    elif r.get("spacings"):
        spacing = [float(x) for x in r["spacings"].split()]
    info["spacing"] = spacing
    return info


def report(info):
    p = info["path"]
    problems = []
    print(f"\n  {p.name}")
    print(f"    dimensions : {info['sizes']}")
    print(f"    type       : {info['type']}")
    print(f"    spacing    : {info['spacing']}")

    if info["spacing"]:
        bad = [s for s in info["spacing"]
               if abs(s - EXPECTED_SPACING_MM) / EXPECTED_SPACING_MM > REL_TOL]
        if bad:
            factor = bad[0] / EXPECTED_SPACING_MM
            problems.append(
                f"spacing {info['spacing']} is {factor:.0f}x the expected "
                f"{EXPECTED_SPACING_MM} mm -- micrometres written as millimetres"
            )
        else:
            dev = max(abs(s - EXPECTED_SPACING_MM) for s in info["spacing"])
            print(f"    spacing ok : within {dev/EXPECTED_SPACING_MM*100:.4f}% "
                  f"of 0.0025 mm")
    else:
        problems.append("no spacing found in header")

    df = info["datafile"]
    print(f"    data file  : {df}")
    if not df:
        problems.append("header has no 'data file:' line")
    else:
        vol = (p.parent / df)
        if not vol.exists():
            problems.append(f"referenced volume not found: {df}")
        else:
            actual = vol.stat().st_size
            print(f"    volume size: {actual / 1e9:.3f} GB")
            if info["sizes"] and info["type"]:
                nb = DTYPE_BYTES.get(info["type"].lower())
                if nb:
                    expect = nb
                    for s in info["sizes"]:
                        expect *= s
                    if abs(expect - actual) > 4096:
                        problems.append(
                            f"size mismatch: header implies {expect/1e9:.3f} GB, "
                            f"file is {actual/1e9:.3f} GB"
                        )
    # BioImage Archive dislikes spaces and non-ASCII in file names
    for name in (p.name, info.get("datafile") or ""):
        if name and (" " in name or any(ord(c) > 127 for c in name)):
            print(f"    -- note: '{name}' contains spaces or non-ASCII; "
                  f"--rename will clean this")

    for msg in problems:
        print(f"    !! {msg}")
    return problems


LABEL_HINTS = ("labelmap", "label", "mask", "seg", "cc_lm")


def is_annotation(info):
    """A label map, not a raw volume: 8-bit and/or named like a mask."""
    name = info["path"].name.lower()
    if any(h in name for h in LABEL_HINTS):
        return True
    t = (info["type"] or "").lower()
    return t in ("uchar", "unsigned char", "uint8")


def place(src, dst, use_copy):
    """Hard-link when possible (costs no extra disk), else copy."""
    if dst.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {dst}")
    if not use_copy:
        try:
            os.link(src, dst)
            return "linked"
        except OSError:
            pass
    free = shutil.disk_usage(dst.parent).free
    need = src.stat().st_size
    if free < need * 1.05:
        raise OSError(
            f"not enough free space: need {need/1e9:.2f} GB, "
            f"have {free/1e9:.2f} GB. Re-run without --copy to hard-link, "
            f"or point --out at another disk."
        )
    shutil.copy2(src, dst)
    return "copied"


def do_rename(root, out_root, folder, sample_id, info, use_copy=False):
    src_dir = root / folder
    annotation = is_annotation(info)

    if annotation:
        dest = out_root / "annotations"
        stem = f"{sample_id}_cc_reference_mask"
    else:
        dest = out_root / sample_id
        stem = sample_id
    dest.mkdir(parents=True, exist_ok=True)

    new_vol = f"{stem}.vol"
    new_hdr = f"{stem}.nhdr"

    how = place(src_dir / info["datafile"], dest / new_vol, use_copy)

    # rewrite the data file line so the pair stays valid
    text = info["path"].read_text(errors="replace")
    text = re.sub(r"(?mi)^(data\s*file\s*:\s*).*$", r"\g<1>" + new_vol, text)
    (dest / new_hdr).write_text(text)

    kind = "annotation" if annotation else "raw volume"
    print(f"    -> {kind}: {dest/new_hdr}  ({how})")

    if not annotation:
        for vgl in src_dir.glob("*.vgl"):
            try:
                place(vgl, dest / f"{sample_id}.vgl", use_copy)
            except FileExistsError:
                pass
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="folder containing Salicornia_1, _4, _5, _8")
    ap.add_argument("--out", default="BIA_upload",
                    help="output tree for the renamed copies")
    ap.add_argument("--rename", action="store_true",
                    help="actually write the clean upload tree")
    ap.add_argument("--copy", action="store_true",
                    help="force real copies instead of hard links "
                         "(only needed if --out is on another disk)")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    out_root = Path(args.out).expanduser()
    all_problems = 0

    for folder, sample_id in SAMPLE_MAP.items():
        print(f"\n=== {folder}  ->  {sample_id} ===")
        d = root / folder
        if not d.is_dir():
            print(f"    !! folder not found: {d}")
            all_problems += 1
            continue
        headers = sorted(d.glob("*.nhdr"))
        headers = [h for h in headers if "texture" not in h.name.lower()]
        if not headers:
            print("    !! no .nhdr found")
            all_problems += 1
            continue
        for h in headers:
            info = parse_nhdr(h)
            probs = report(info)
            all_problems += len(probs)
            if args.rename and not probs:
                try:
                    do_rename(root, out_root, folder, sample_id, info,
                              use_copy=args.copy)
                except (OSError, FileExistsError) as e:
                    print(f"    !! {e}")
                    all_problems += 1
            elif args.rename and probs:
                print("    -> skipped rename: fix the problems above first")

    print("\n" + "=" * 52)
    if all_problems:
        print(f"{all_problems} problem(s) found. Do not upload until these are resolved.")
    else:
        print("All headers verified: spacing, dimensions and data file references are consistent.")
        if not args.rename:
            print("Re-run with --rename to write the clean upload tree.")
            print("Hard links are used by default, so no extra disk space is consumed.")


if __name__ == "__main__":
    main()
