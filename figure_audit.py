#!/usr/bin/env python3
"""
Audit figure files against The Plant Journal's requirements.

    python3 figure_audit.py ~/Desktop/fig_audit/from_docx
    python3 figure_audit.py ~/Desktop/figures_final

For every image it reports the pixel size, the embedded resolution, and the
widest page column the file can fill at 300 dpi. TPJ asks for 300 dpi for
photographs, 600 dpi for line art and composites, a width of either 80 mm
(one column) or 169 mm (full page), and no lettering below 8 pt once the
figure has been scaled to that width.

Nothing is modified. This only reports.

Needs Pillow:  pip3 install pillow
"""

import sys, os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

COL_MM, PAGE_MM = 80.0, 169.0
MM_PER_INCH = 25.4

def mm_at(px, dpi):
    return px / dpi * MM_PER_INCH

def audit(path):
    try:
        im = Image.open(path)
    except Exception as e:
        return None, f"cannot open ({e})"
    w, h = im.size
    dpi = im.info.get("dpi", (None, None))[0]
    try:
        dpi = float(dpi) if dpi else None
    except Exception:
        dpi = None

    # Width achievable at the two target resolutions
    w300, w600 = mm_at(w, 300), mm_at(w, 600)

    if   w300 >= PAGE_MM: verdict, note = "OK",    f"fills {PAGE_MM:.0f} mm at 300 dpi"
    elif w300 >= COL_MM:  verdict, note = "COLUMN", f"only {w300:.0f} mm at 300 dpi - one column, not full page"
    else:                 verdict, note = "TOO SMALL", f"only {w300:.0f} mm at 300 dpi - must be regenerated"

    return dict(file=os.path.basename(path), mode=im.mode, fmt=im.format,
                px=f"{w}x{h}", dpi=(f"{dpi:.0f}" if dpi else "-"),
                w300=f"{w300:.0f}", w600=f"{w600:.0f}",
                verdict=verdict, note=note), None

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    root = os.path.expanduser(sys.argv[1])
    exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".emf", ".wmf")
    files = sorted(
        os.path.join(dp, f)
        for dp, _, fs in os.walk(root) for f in fs
        if f.lower().endswith(exts) and not f.startswith(".")
    )
    if not files:
        print(f"No images found under {root}"); sys.exit(1)

    print(f"\nAuditing {len(files)} file(s) under {root}")
    print(f"Targets: 300 dpi photographs, 600 dpi line art, width 80 mm or {PAGE_MM:.0f} mm\n")
    hdr = f"{'file':<34}{'format':<7}{'mode':<7}{'pixels':<14}{'dpi':<6}{'mm@300':<8}{'mm@600':<8}verdict"
    print(hdr); print("-" * len(hdr))

    counts = {}
    for p in files:
        r, err = audit(p)
        if err:
            print(f"{os.path.basename(p):<34}{err}"); continue
        counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
        print(f"{r['file']:<34}{str(r['fmt']):<7}{r['mode']:<7}{r['px']:<14}"
              f"{r['dpi']:<6}{r['w300']:<8}{r['w600']:<8}{r['verdict']}  - {r['note']}")

    print()
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print("""
How to read this
  OK         wide enough to run across the full page at 300 dpi.
  COLUMN     usable, but only at single-column width. Fine for a simple
             plot, not for a six-panel composite.
  TOO SMALL  must be exported again from whatever produced it. Upscaling
             an existing file adds pixels but no detail, and reviewers
             can see the difference.

Resolution is only half the job. Open each figure at the width it will be
printed and check that the abbreviations (cc, pc, pt, pi, xy) are still
readable. If they are not, enlarge the text in the source file - never by
scaling the whole image, which makes the labels blurry along with it.
""")

if __name__ == "__main__":
    main()
