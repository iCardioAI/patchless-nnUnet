#!/usr/bin/env python3
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.image import resample_to_img
from nibabel.orientations import (
    aff2axcodes, io_orientation, ornt_transform,
    apply_orientation, inv_ornt_aff
)

BASE = "nifti_output/case_1e_0000.nii.gz"
MASK = "inference_raw/case_1e.nii.gz"

FLIP_LR  = False
FLIP_UD  = False
ROTATE_K = 1  # 0..3

USE_MANUAL_REORIENT = False


def _rgb_struct_to_gray(arr: np.ndarray) -> np.ndarray:
    """Convert structured/void dtype (packed RGB/RGBA) to grayscale float32."""
    arr_c = np.ascontiguousarray(arr)     # ensure bytes are contiguous
    bpp   = arr_c.dtype.itemsize          # bytes per voxel (3=RGB, 4=RGBA)
    if bpp < 3:
        raise RuntimeError(f"Unexpected bytes/voxel {bpp} for RGB-like NIfTI.")
    u8  = arr_c.view(np.uint8).reshape(arr_c.shape + (bpp,))
    rgb = u8[..., :3].astype(np.float32, copy=False)  # ignore alpha if present
    gray = 0.299*rgb[..., 0] + 0.587*rgb[..., 1] + 0.114*rgb[..., 2]
    return gray.astype(np.float32, copy=False)

def to_scalar_img(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Return scalar float32 NIfTI; converts RGB/RGBA to gray; preserves shape."""
    arr = np.asanyarray(img.dataobj)  # avoid get_fdata() (fails on void dtype)
    if arr.dtype.kind == 'V':         # structured/void → RGB/RGBA
        arr = _rgb_struct_to_gray(arr)
    else:
        arr = np.asarray(arr, dtype=np.float32)

    # Ensure 3D (H, W, Z). If 2D, add a singleton Z.
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim > 3:
        raise ValueError(f"Unexpected ndim {arr.ndim}; expected 2D/3D.")

    out = nib.Nifti1Image(arr, img.affine, header=img.header)
    out.set_data_dtype(np.float32)
    return out

def reorient_like(img: nib.Nifti1Image, ref: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorient img so its axes match ref (no resampling). Robust to RGB."""
    src_ornt = io_orientation(img.affine)
    ref_ornt = io_orientation(ref.affine)
    xform = ornt_transform(src_ornt, ref_ornt)

    arr = np.asanyarray(img.dataobj)
    if arr.dtype.kind == 'V':
        arr = _rgb_struct_to_gray(arr)
    else:
        arr = np.asarray(arr, dtype=np.float32)

    data = apply_orientation(arr, xform)
    new_aff = img.affine @ inv_ornt_aff(xform, img.shape)
    hdr = img.header.copy()
    hdr.set_data_shape(data.shape)
    out = nib.Nifti1Image(data.astype(np.float32, copy=False), new_aff, header=hdr)
    out.set_data_dtype(np.float32)
    return out

def align_mask_to_base(mask_img: nib.Nifti1Image, base_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    If shapes match, adopt base affine/header (index-space overlay).
    Otherwise, resample mask → base with nearest interpolation.
    """
    if mask_img.shape == base_img.shape:
        data = np.asanyarray(mask_img.dataobj)
        out = nib.Nifti1Image(data, base_img.affine, header=base_img.header)
        out.set_data_dtype(data.dtype)
        return out
    else:
        return resample_to_img(mask_img, base_img, interpolation="nearest",
                               force_resample=True, copy_header=True)

def normalize_mask_dims(mm: np.ndarray) -> np.ndarray:
    """Collapse 4D channel maps or threshold probability maps → label mask."""
    if mm.ndim == 4:
        mm = mm.argmax(axis=-1).astype(np.uint8)
        return mm
    vals = np.unique(mm[: min(200000, mm.size)])  # sampled
    if (vals.size > 10) or (mm.max() <= 1.0 and np.any((vals != 0) & (vals != 1))):
        mm = (mm > 0.5).astype(np.uint8)
    return mm

def show_overlay(base_img, mask_on_base, z=None, flip_lr=False, flip_ud=False, rotate_k=0):
    """Display overlay using base's native pixel grid (no physical stretching)."""
    base = base_img.get_fdata()
    mask = mask_on_base.get_fdata()

    if z is None:
        if mask.ndim == 3 and mask.size > 0:
            z = int(np.argmax(mask.reshape(-1, mask.shape[2]).sum(axis=0)))
        else:
            z = 0

    base_slice = base[:, :, z]
    mask_slice = mask[:, :, z]

    if rotate_k:
        base_slice = np.rot90(base_slice, k=rotate_k)
        mask_slice = np.rot90(mask_slice, k=rotate_k)
    if flip_lr:
        base_slice = np.fliplr(base_slice)
        mask_slice = np.fliplr(mask_slice)
    if flip_ud:
        base_slice = np.flipud(base_slice)
        mask_slice = np.flipud(mask_slice)

    plt.figure(figsize=(12, 10))
    plt.imshow(base_slice, cmap="gray", origin="lower", aspect="equal", interpolation="nearest")
    overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
    plt.imshow(overlay, cmap="autumn", alpha=0.5, origin="lower", aspect="equal", interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


base_img = to_scalar_img(nib.load(BASE))
mask_img = to_scalar_img(nib.load(MASK))

print("Base axcodes:", aff2axcodes(base_img.affine), "zooms:", base_img.header.get_zooms()[:3])
print("Mask axcodes:", aff2axcodes(mask_img.affine), "zooms:", mask_img.header.get_zooms()[:3])
print("Base affine:\n", np.array_str(base_img.affine, precision=3, suppress_small=True))
print("Mask affine:\n", np.array_str(mask_img.affine, precision=3, suppress_small=True))

# 0) Raw mask stats (before any transforms)
raw = np.asanyarray(mask_img.dataobj)
print("[RAW] mask shape:", raw.shape,
      "| unique (up to 20):", np.unique(raw)[:20],
      "| nonzeros:", int(np.count_nonzero(raw)))

# 1) Optional manual reorient (OFF by default)
if USE_MANUAL_REORIENT:
    mask_work = reorient_like(mask_img, base_img)
else:
    mask_work = mask_img

# 2) Align mask → base (prefer header adoption when shapes match)
mask_on_base = align_mask_to_base(mask_work, base_img)
mm = np.asanyarray(mask_on_base.dataobj)

# 3) Normalize mask dims/content if needed (probs/4D)
mm = normalize_mask_dims(mm)

# Ensure integer labels (nearest should keep ints, but be safe)
if mm.dtype != np.uint8:
    mm = np.rint(mm).astype(np.uint8)

print("[post-align] shape:", mm.shape, "| matches base?", mask_on_base.shape == base_img.shape)
u = np.unique(mm)
print("[post-align] unique (first 20):", u[:20], "| nonzeros:", int(np.count_nonzero(mm)))
if mm.max() == 0:
    print("Mask is all zeros after alignment. Likely affine mismatch; "
          "adopted base affine if shapes matched, otherwise check input spacings/origins.")

# 4) Visualize overlay (auto-pick a labeled slice)
z = int(np.argmax((mm != 0).reshape(-1, mm.shape[2]).sum(0))) if mm.ndim == 3 and mm.max() > 0 else 0

mask_on_base = nib.Nifti1Image(mm, base_img.affine, base_img.header)
show_overlay(base_img, mask_on_base, z=z, flip_lr=FLIP_LR, flip_ud=FLIP_UD, rotate_k=ROTATE_K)

