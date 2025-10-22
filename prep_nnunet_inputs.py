#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import nibabel as nib
from nibabel.orientations import (
    aff2axcodes, io_orientation, ornt_transform,
    apply_orientation, inv_ornt_aff, axcodes2ornt
)

IN_DIR  = Path("~/Desktop/niftis").expanduser()
OUT_DIR = IN_DIR  # generate them in the same folder
GLOB    = "case_*.nii.gz"           # which inputs to process
SKIP_IF_HAS_0000_SUFFIX = True      # skip files already like *_0000.nii.gz
DESIRED_AXCODES  = ('L','P','S')    # Arnaud’s convention
DESIRED_SPACING  = (0.37, 0.37, 1.0)

def _rgb_struct_to_gray(arr: np.ndarray) -> np.ndarray:
    """Convert structured/void dtype (packed RGB/RGBA) to grayscale float32."""
    arr_c = np.ascontiguousarray(arr)       # ensure contiguous for viewing bytes
    bpp = arr_c.dtype.itemsize              # bytes per voxel (3=RGB, 4=RGBA)
    if bpp < 3:
        raise RuntimeError(f"Unexpected bytes/voxel {bpp} for RGB-like NIfTI.")
    u8 = arr_c.view(np.uint8).reshape(arr_c.shape + (bpp,))
    rgb = u8[..., :3].astype(np.float32, copy=False)  # ignore alpha/extra
    gray = 0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]
    return gray.astype(np.float32, copy=False)

def to_float32_single_channel(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Return scalar float32 NIfTI; converts RGB/RGBA to grayscale."""
    dataobj = img.dataobj  # array proxy
    arr = np.asanyarray(dataobj)  # raw; avoid get_fdata() on RGB
    if arr.dtype.kind == 'V':     # structured/void → RGB/RGBA
        arr = _rgb_struct_to_gray(arr)
    else:
        arr = np.asarray(arr, dtype=np.float32)

    # Ensure 3D (H, W, T). If 2D, add a singleton third dim.
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim > 3:
        raise ValueError(f"Unexpected ndim {arr.ndim}; expected 2D/3D after conversion.")

    out = nib.Nifti1Image(arr, img.affine, header=img.header)
    out.set_data_dtype(np.float32)
    return out

def reorient_to_axcodes(img: nib.Nifti1Image, target_axcodes=('L','P','S')) -> nib.Nifti1Image:
    """Apply orientation transform to match target axcodes (no resampling)."""
    src_ornt = io_orientation(img.affine)
    tgt_ornt = axcodes2ornt(target_axcodes)
    xform = ornt_transform(src_ornt, tgt_ornt)
    data = apply_orientation(np.asanyarray(img.dataobj), xform)
    new_aff = img.affine @ inv_ornt_aff(xform, img.shape)
    hdr = img.header.copy()
    hdr.set_data_shape(data.shape)
    out = nib.Nifti1Image(data.astype(np.float32, copy=False), new_aff, header=hdr)
    out.set_data_dtype(np.float32)
    return out

def set_spacings_on_affine(aff: np.ndarray, spacings=(0.37,0.37,1.0)) -> np.ndarray:
    """
    Scale each affine column so its length equals the desired spacing,
    preserving axis directions/orientation.
    """
    A = aff.copy()
    for i, sp in enumerate(spacings):
        v = A[:3, i]
        n = float(np.linalg.norm(v))
        if n == 0.0:
            # fallback: set pure axis with spacing on diagonal (rare)
            A[:3, i] = 0.0
            A[i, i] = sp
        else:
            A[:3, i] = (v / n) * sp
    return A

def sanitize_header_zooms(hdr: nib.Nifti1Header, desired_spacings):
    # nib expects you to set shape first
    shp = hdr.get_data_shape()
    z = list(hdr.get_zooms())
    if len(z) < len(shp):
        z = list(z) + [1.0] * (len(shp) - len(z))
    z[:3] = desired_spacings
    hdr.set_zooms(tuple(z))

def make_out_path(p: Path) -> Path:
    stem = p.stem  # drops .gz, so handle .nii.gz carefully
    # If endswith .nii.gz Path.stem gives 'case_1_0000.nii', use p.name instead:
    name = p.name
    if name.endswith(".nii.gz"):
        base = name[:-7]  # strip ".nii.gz"
    elif name.endswith(".nii"):
        base = name[:-4]
    else:
        base = name
    # enforce *_0000 suffix for nnU-Net images
    if not base.endswith("_0000"):
        base = base + "_0000"
    return (OUT_DIR / (base + ".nii.gz")).resolve()

def process_one(p: Path):
    img0 = nib.load(str(p))
    before_ax = aff2axcodes(img0.affine)
    before_zooms = tuple(float(x) for x in img0.header.get_zooms()[:3])

    # 1) single-channel float32
    img = to_float32_single_channel(img0)

    # 2) reorient to LPS
    if aff2axcodes(img.affine) != DESIRED_AXCODES:
        img = reorient_to_axcodes(img, DESIRED_AXCODES)

    # 3) set target spacings on affine & header (no resampling)
    new_aff = set_spacings_on_affine(img.affine, DESIRED_SPACING)
    hdr = img.header.copy()
    hdr.set_data_shape(img.shape)
    sanitize_header_zooms(hdr, DESIRED_SPACING)
    out_img = nib.Nifti1Image(np.asanyarray(img.dataobj).astype(np.float32, copy=False), new_aff, header=hdr)
    out_img.set_data_dtype(np.float32)
    # keep both sform & qform consistent
    out_img.set_sform(new_aff, code=1)
    out_img.set_qform(new_aff, code=1)

    after_ax = aff2axcodes(out_img.affine)
    after_zooms = tuple(float(x) for x in out_img.header.get_zooms()[:3])

    out_p = make_out_path(p)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_p))

    print(f"[OK] {p.name}  ax {before_ax}->{after_ax}  zooms {before_zooms}->{after_zooms}  -> {out_p.name}")

def main():
    paths = sorted(IN_DIR.glob(GLOB))
    if not paths:
        print(f"[WARN] No inputs found in {IN_DIR} matching {GLOB}")
        sys.exit(0)
    for p in paths:
        if SKIP_IF_HAS_0000_SUFFIX and p.name.endswith("_0000.nii.gz"):
            continue
        try:
            process_one(p)
        except Exception as e:
            print(f"[ERR] {p.name}: {e}")

if __name__ == "__main__":
    main()

