#!/usr/bin/env python3
"""
vgl_to_nrrd.py
==============
Reads a VGStudio project file (.vgl) and generates a detached NRRD header
(.nhdr) that points directly at the raw .vol volume data, so it can be
opened in 3D Slicer (or any NRRD-aware tool) WITHOUT needing myVGL,
VGStudio, or a Windows VM.

How it works
------------
A .vgl file is actually a gzip-compressed XML document. This script:
  1. Decompresses it (if needed).
  2. Parses the XML to find every raw-volume import block
     (class="VGLSampleGridImportRaw").
  3. Extracts: the referenced .vol filename, voxel grid size, data type,
     byte order, header skip, and voxel spacing.
  4. Writes a small .nhdr text file next to your .vol file. This .nhdr
     is NOT a copy of your data -- it just tells Slicer how to read the
     existing .vol file (dimensions, type, endianness, spacing).

Usage
-----
    python3 vgl_to_nrrd.py "/path/to/Salicornia 5/Salicornia 5.vgl"

This will create:
    /path/to/Salicornia 5/Salicornia 5.nhdr

Then in 3D Slicer: File > Add Data... and select the .nhdr file
(or just drag & drop it into the Slicer window).

Notes
-----
- The .nhdr file must stay in the SAME FOLDER as the .vol file, because
  it references it by relative filename.
- If your .vgl references multiple volumes (rare, but possible for
  multi-part / multi-channel scans), one .nhdr is written per volume.
- Tested against real VGStudio MAX project files where the .vol sits
  next to the .vgl (the common "single volume" project layout).
"""

import xml.etree.ElementTree as ET
import zlib
import os
import sys
import argparse


# VGStudio -> NRRD data type mapping
DATATYPE_MAP = {
    "UInt8":   "unsigned char",
    "Int8":    "signed char",
    "UInt16":  "unsigned short",
    "Int16":   "short",
    "UInt32":  "unsigned int",
    "Int32":   "int",
    "Float32": "float",
    "Float64": "double",
}

BYTEORDER_MAP = {
    "LittleEndian": "little",
    "BigEndian": "big",
}


def load_vgl_xml(path):
    """Load a .vgl file, transparently handling gzip compression.
    Uses raw zlib decompression (gzip-wrapper, wbits=16+MAX_WBITS) and
    stops at the first complete member, since some .vgl files have
    extra trailing bytes after the gzip stream that would otherwise
    confuse Python's gzip module."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] == b"\x1f\x8b":  # gzip magic bytes
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        data = decompressor.decompress(raw)
        data += decompressor.flush()
    else:
        data = raw
    return ET.fromstring(data)


def get_prop(obj_elem, name):
    for p in obj_elem.findall("property"):
        if p.get("name") == name:
            return p
    return None


def prop_text(prop_elem):
    if prop_elem is None:
        return None
    children = list(prop_elem)
    if not children:
        return None
    return children[0].text


def extract_volumes(vgl_path):
    """Return a list of dicts describing every raw volume referenced
    in the project."""
    root = load_vgl_xml(vgl_path)
    results = []

    for obj in root.iter("object"):
        if obj.get("class") != "VGLSampleGridImportRaw":
            continue

        filename = prop_text(get_prop(obj, "FileName"))
        byteorder = prop_text(get_prop(obj, "ByteOrder"))
        headerskip = prop_text(get_prop(obj, "HeaderSkip"))

        gridsize = None
        datatype = None
        spacing = None

        fi_prop = get_prop(obj, "ImportSettingsFileInfo")
        if fi_prop is not None:
            fi_obj = fi_prop.find('.//object[@class="VGLVolumeImportSettings::FileInfo"]')
            if fi_obj is not None:
                gridsize = prop_text(get_prop(fi_obj, "GridSize"))
                datatype = prop_text(get_prop(fi_obj, "SampleDataType"))
                tm_prop = get_prop(fi_obj, "TransformMatrixList")
                if tm_prop is not None:
                    m4 = tm_prop.find("matrix4")
                    if m4 is not None and m4.text:
                        vals = [float(v) for v in m4.text.split()]
                        # diagonal of the 4x4 transform = voxel spacing (mm)
                        spacing = (vals[0], vals[5], vals[10])

        if filename and gridsize and datatype:
            results.append({
                "filename": filename,
                "gridsize": gridsize,
                "datatype": datatype,
                "byteorder": byteorder or "LittleEndian",
                "headerskip": headerskip or "0 0 0 0",
                "spacing": spacing or (1.0, 1.0, 1.0),
            })

    return results


def resolve_vol_path(vgl_path, stored_filename):
    """The .vgl stores an absolute Windows-style path (e.g.
    'S:/CT_Data/.../Salicornia 5.vol') that won't exist on your machine,
    and the file may also have been renamed (e.g. spaces replaced with
    underscores) since the project was created. We try, in order:
      1. The exact basename stored in the .vgl, in the .vgl's folder.
      2. A case/whitespace/underscore-insensitive match in that folder.
      3. If exactly one .vol file exists in that folder, use it.
    """
    stored_basename = stored_filename.replace("\\", "/").split("/")[-1]
    folder = os.path.dirname(os.path.abspath(vgl_path))

    exact = os.path.join(folder, stored_basename)
    if os.path.exists(exact):
        return exact, stored_basename

    def normalize(name):
        return name.lower().replace(" ", "").replace("_", "").replace("-", "")

    target_norm = normalize(stored_basename)
    candidates = [f for f in os.listdir(folder) if f.lower().endswith(".vol")]

    for f in candidates:
        if normalize(f) == target_norm:
            return os.path.join(folder, f), f

    if len(candidates) == 1:
        print(f"  [i] Note: '{stored_basename}' not found as-is, but found "
              f"a single .vol file in the folder: '{candidates[0]}'. Using that.")
        return os.path.join(folder, candidates[0]), candidates[0]

    # give up, return the originally expected (possibly wrong) path so the
    # caller can print a clear warning
    return exact, stored_basename


def write_nhdr(vgl_path, volume_info, index=0):
    vol_full_path, vol_basename = resolve_vol_path(vgl_path, volume_info["filename"])

    nrrd_type = DATATYPE_MAP.get(volume_info["datatype"])
    if nrrd_type is None:
        raise ValueError(f"Unsupported/unknown data type: {volume_info['datatype']}")

    nrrd_endian = BYTEORDER_MAP.get(volume_info["byteorder"], "little")

    size_vals = [int(float(v)) for v in volume_info["gridsize"].split()]
    # gridsize is X Y Z [Channels] -- drop the trailing channel count if it's 1
    if len(size_vals) == 4 and size_vals[3] == 1:
        size_vals = size_vals[:3]
    sizes_str = " ".join(str(v) for v in size_vals)

    sx, sy, sz = volume_info["spacing"]
    spacings_str = f"{sx} {sy} {sz}"

    header_skip = [int(float(v)) for v in volume_info["headerskip"].split()]
    byte_skip = header_skip[0] if header_skip else 0

    if not os.path.exists(vol_full_path):
        print(f"  [!] Warning: expected raw data file not found next to .vgl:")
        print(f"      {vol_full_path}")
        print(f"      Make sure '{vol_basename}' is in the same folder as the .vgl file.")

    vgl_dir = os.path.dirname(os.path.abspath(vgl_path))
    vgl_stem = os.path.splitext(os.path.basename(vgl_path))[0]
    suffix = "" if index == 0 else f"_{index}"
    nhdr_path = os.path.join(vgl_dir, f"{vgl_stem}{suffix}.nhdr")

    with open(nhdr_path, "w") as f:
        f.write("NRRD0004\n")
        f.write(f"type: {nrrd_type}\n")
        f.write(f"dimension: {len(size_vals)}\n")
        f.write(f"sizes: {sizes_str}\n")
        f.write(f"spacings: {spacings_str}\n")
        f.write(f"endian: {nrrd_endian}\n")
        f.write("encoding: raw\n")
        if byte_skip:
            f.write(f"byte skip: {byte_skip}\n")
        f.write(f"data file: {vol_basename}\n")

    return nhdr_path


def main():
    parser = argparse.ArgumentParser(description="Convert a VGStudio .vgl project into a Slicer-ready .nhdr header.")
    parser.add_argument("vgl_path", help="Path to the .vgl project file")
    args = parser.parse_args()

    if not os.path.exists(args.vgl_path):
        print(f"File not found: {args.vgl_path}")
        sys.exit(1)

    print(f"Reading {args.vgl_path} ...")
    volumes = extract_volumes(args.vgl_path)

    if not volumes:
        print("No raw volume import blocks found in this project.")
        print("(The project might already reference a pre-processed / different format.)")
        sys.exit(1)

    print(f"Found {len(volumes)} volume(s) in the project.\n")

    for i, vol in enumerate(volumes):
        print(f"Volume {i+1}:")
        print(f"  Data file : {vol['filename'].split('/')[-1]}")
        print(f"  Grid size : {vol['gridsize']}")
        print(f"  Data type : {vol['datatype']}")
        print(f"  Byte order: {vol['byteorder']}")
        print(f"  Spacing   : {vol['spacing']} mm")
        nhdr_path = write_nhdr(args.vgl_path, vol, index=i)
        print(f"  -> Wrote: {nhdr_path}\n")

    print("Done. Drag the .nhdr file(s) above into 3D Slicer to load the volume.")


if __name__ == "__main__":
    main()
