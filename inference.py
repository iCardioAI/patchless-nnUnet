"""
End-to-end runner:
DICOM (.dcm or folder) -> NIfTI (dcm2niix) -> preprocess -> patchless-nnUNet -> PNG overlays

Requirements:
  - dcm2niix installed and in PATH (macOS: `brew install dcm2niix`)
  - Python pkgs: nibabel, numpy, matplotlib, nilearn
  - Access to the patchless-nnUnet repo + checkpoint

Usage:
  python run_end_to_end.py \
    --input /path/to/dicom_or_dir \
    --patchless_dir /home/users/<you>/patchless-nnUnet \
    --ckpt /home/users/<you>/patchless-nnUnet/actor_3d_ANAT_LM_BEST_1.ckpt \
    --workdir /tmp/echo_run \
    --apply-eq-hist \
    --show

Outputs:
  - <workdir>/nifti_raw/          (converted NIfTI)
  - <workdir>/data/               (preprocessed NIfTI, *_0000.nii.gz)
  - <workdir>/output/             (model outputs from patchless-nnUnet)
  - <workdir>/overlays/*.png      (saved overlays)
"""

import os, sys, shutil, subprocess, argparse, textwrap
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.image import resample_to_img
from nibabel.orientations import (
    aff2axcodes, io_orientation, ornt_transform,
    apply_orientation, inv_ornt_aff, axcodes2ornt
)

DESIRED_AXCODES  = ('L','P','S')
DESIRED_SPACING  = (0.37, 0.37, 1.0)

def _rgb_struct_to_gray(arr: np.ndarray) -> np.ndarray:
    """Convert structured/void dtype (packed RGB/RGBA) to grayscale float32."""
    arr_c = np.ascontiguousarray(arr)
    bpp   = arr_c.dtype.itemsize  # bytes per voxel (3=RGB, 4=RGBA)
    if bpp < 3:
        raise RuntimeError(f"Unexpected bytes/voxel {bpp} for RGB-like NIfTI.")
    u8  = arr_c.view(np.uint8).reshape(arr_c.shape + (bpp,))
    rgb = u8[..., :3].astype(np.float32, copy=False)
    gray = 0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]
    return gray.astype(np.float32, copy=False)

def to_float32_single_channel(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Return scalar float32 NIfTI; converts RGB/RGBA to grayscale; ensures 3D."""
    arr = np.asanyarray(img.dataobj)  # avoid get_fdata() on void dtype
    if arr.dtype.kind == 'V':
        arr = _rgb_struct_to_gray(arr)
    else:
        arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim > 3:
        raise ValueError(f"Unexpected ndim {arr.ndim}; expected 2D/3D.")
    out = nib.Nifti1Image(arr, img.affine, header=img.header)
    out.set_data_dtype(np.float32)
    return out

def reorient_to_axcodes(img: nib.Nifti1Image, target_axcodes=('L','P','S')) -> nib.Nifti1Image:
    src_ornt = io_orientation(img.affine)
    tgt_ornt = axcodes2ornt(target_axcodes)
    xform = ornt_transform(src_ornt, tgt_ornt)
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

def set_spacings_on_affine(aff: np.ndarray, spacings=(0.37,0.37,1.0)) -> np.ndarray:
    """Scale affine columns so their norms equal desired spacings (preserve direction)."""
    A = aff.copy()
    for i, sp in enumerate(spacings):
        v = A[:3, i]
        n = float(np.linalg.norm(v))
        if n == 0.0:
            A[:3, i] = 0.0
            A[i, i] = sp
        else:
            A[:3, i] = (v / n) * sp
    return A

def sanitize_header_zooms(hdr: nib.Nifti1Header, desired_spacings):
    shp = hdr.get_data_shape()
    z = list(hdr.get_zooms())
    if len(z) < len(shp):
        z = list(z) + [1.0] * (len(shp) - len(z))
    z[:3] = desired_spacings
    hdr.set_zooms(tuple(z))

def ensure_0000_suffix(name: str) -> str:
    base = name
    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]
    if not base.endswith("_0000"):
        base += "_0000"
    return base + ".nii.gz"

def preprocess_nifti(in_path: Path, out_dir: Path) -> Path:
    """Preprocess a single NIfTI and save as *_0000.nii.gz in out_dir."""
    img0 = nib.load(str(in_path))
    img  = to_float32_single_channel(img0)
    if aff2axcodes(img.affine) != DESIRED_AXCODES:
        img = reorient_to_axcodes(img, DESIRED_AXCODES)
    new_aff = set_spacings_on_affine(img.affine, DESIRED_SPACING)
    hdr = img.header.copy()
    hdr.set_data_shape(img.shape)
    sanitize_header_zooms(hdr, DESIRED_SPACING)
    out_img = nib.Nifti1Image(np.asanyarray(img.dataobj).astype(np.float32, copy=False), new_aff, header=hdr)
    out_img.set_data_dtype(np.float32)
    out_img.set_sform(new_aff, code=1)
    out_img.set_qform(new_aff, code=1)

    out_name = ensure_0000_suffix(in_path.name)
    out_path = out_dir / out_name
    nib.save(out_img, str(out_path))
    return out_path

def align_mask_to_base(mask_img: nib.Nifti1Image, base_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Adopt base affine if shapes match; else resample (nearest)."""
    if mask_img.shape == base_img.shape:
        data = np.asanyarray(mask_img.dataobj)
        return nib.Nifti1Image(data, base_img.affine, header=base_img.header)
    from nilearn.image import resample_to_img
    return resample_to_img(mask_img, base_img, interpolation="nearest",
                           force_resample=True, copy_header=True)

def save_overlay_png(base_img: nib.Nifti1Image, mask_img: nib.Nifti1Image, out_png: Path, rotate_k=0):
    base = base_img.get_fdata()
    mask = mask_img.get_fdata()
    # pick a slice with labels if any
    if mask.ndim == 3 and mask.max() > 0:
        z = int(np.argmax((mask != 0).reshape(-1, mask.shape[2]).sum(0)))
    else:
        z = base.shape[2] // 2
    base_slice = base[:, :, z]
    mask_slice = mask[:, :, z]
    if rotate_k:
        base_slice = np.rot90(base_slice, k=rotate_k)
        mask_slice = np.rot90(mask_slice, k=rotate_k)
    plt.figure(figsize=(10, 8))
    plt.imshow(base_slice, cmap="gray", origin="lower", aspect="equal", interpolation="nearest")
    overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
    plt.imshow(overlay, cmap="autumn", alpha=0.5, origin="lower", aspect="equal", interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def run_dcm2niix(input_path: Path, out_dir: Path):
    """Run dcm2niix on a single DICOM file or a directory of DICOMs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # dcm2niix only takes directories; if a .dcm file was provided, use its parent
    src_dir = input_path if input_path.is_dir() else input_path.parent
    cmd = ["dcm2niix", "-z", "y", "-f", "%p_%s", "-o", str(out_dir), str(src_dir)]
    print("[dcm2niix]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_patchless_predict(patchless_dir: Path, input_folder: Path, output_folder: Path,
                          ckpt_path: Path, apply_eq_hist: bool, extra_args=None):
    """
    Call: python -m patchless_nnunet.predict model=patchless_nnunet input_folder=... output_folder=...
    """
    env = os.environ.copy()
    # Run from inside repo so hydra configs resolve
    cmd = [
        sys.executable, "-m", "patchless_nnunet.predict",
        "model=patchless_nnunet",
        f"input_folder={str(input_folder)}",
        f"output_folder={str(output_folder)}",
        f"ckpt_path={str(ckpt_path)}",
        f"apply_eq_hist={'true' if apply_eq_hist else 'false'}",
        "tta=false",
        "num_workers=2",
        "pin_memory=false",
        "trainer.accelerator=gpu",
        "trainer.devices=1",
        "trainer.precision=32",
    ]
    if extra_args:
        cmd.extend(extra_args)
    print("[predict]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(patchless_dir), env=env, check=True)

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="DICOM -> NIfTI -> preprocess -> patchless-nnUNet -> PNG overlays"
    )
    ap.add_argument("--input", required=True, help="Path to a DICOM .dcm file OR a directory of DICOMs")
    ap.add_argument("--patchless_dir", required=True, help="Path to patchless-nnUnet repository root")
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint (.ckpt)")
    ap.add_argument("--workdir", default="./work_echo_run", help="Working directory for intermediates/outputs")
    ap.add_argument("--rotate-k", type=int, default=1, help="Rotate display k*90° in overlay (default 1)")
    ap.add_argument("--apply-eq-hist", action="store_true", help="Pass apply_eq_hist=true to predictor")
    ap.add_argument("--show", action="store_true", help="Also show overlays interactively")
    args = ap.parse_args()

    input_path   = Path(args.input).resolve()
    patchless_dir= Path(args.patchless_dir).resolve()
    ckpt_path    = Path(args.ckpt).resolve()
    workdir      = Path(args.workdir).resolve()

    # Check dcm2niix
    if shutil.which("dcm2niix") is None:
        print("ERROR: dcm2niix not found in PATH. Install it (macOS: `brew install dcm2niix`).")
        sys.exit(1)

    # Prepare folders
    nifti_raw = workdir / "nifti_raw"
    prepped   = workdir / "data"          # patchless expects a folder of *_0000.nii.gz
    out_dir   = workdir / "output"        # model output folder
    overlays  = workdir / "overlays"
    for d in (nifti_raw, prepped, out_dir, overlays):
        d.mkdir(parents=True, exist_ok=True)

    # 1) DICOM -> NIfTI
    run_dcm2niix(input_path, nifti_raw)

    # 2) Preprocess all NIfTI from nifti_raw -> prepped (ensure *_0000.nii.gz)
    produced = []
    for p in sorted(nifti_raw.glob("*.nii*")):
        try:
            out_p = preprocess_nifti(p, prepped)
            produced.append(out_p)
            print(f"[preprocess] {p.name} -> {out_p.name}  ax={aff2axcodes(nib.load(str(out_p)).affine)}  zooms={nib.load(str(out_p)).header.get_zooms()[:3]}")
        except Exception as e:
            print(f"[preprocess][ERR] {p.name}: {e}")

    if not produced:
        print("ERROR: No preprocessed NIfTI produced; check your DICOM input.")
        sys.exit(2)

    # 3) Run patchless-nnUNet
    run_patchless_predict(
        patchless_dir=patchless_dir,
        input_folder=prepped,
        output_folder=out_dir,
        ckpt_path=ckpt_path,
        apply_eq_hist=args.apply_eq_hist,
    )

    # 4) Find outputs (assumes predictor writes to output/inference_raw/*.nii.gz)
    inference_raw = out_dir / "inference_raw"
    mask_files = sorted(inference_raw.glob("*.nii.gz"))
    if not mask_files:
        print("WARNING: No mask outputs found in output/inference_raw.")
        sys.exit(3)

    # 5) For each produced base, try to match a mask (by stem ignoring _0000)
    for base_p in produced:
        base = nib.load(str(base_p))
        base_key = base_p.stem.replace("_0000", "")
        # find first matching mask containing the key (simple heuristic)
        match = None
        for m in mask_files:
            if base_key in m.stem:
                match = m
                break
        if match is None:
            # fallback: just take first mask
            match = mask_files[0]

        mask = nib.load(str(match))
        # Align mask to base (adopt base affine if shapes match)
        if mask.shape == base.shape:
            mask_aligned = nib.Nifti1Image(np.asanyarray(mask.dataobj), base.affine, header=base.header)
        else:
            mask_aligned = resample_to_img(mask, base, interpolation="nearest", force_resample=True, copy_header=True)

        # Ensure integer labels
        mm = np.asanyarray(mask_aligned.dataobj)
        if mm.ndim == 4:
            mm = mm.argmax(axis=-1).astype(np.uint8)
        else:
            mm = np.rint(mm).astype(np.uint8)
        mask_aligned = nib.Nifti1Image(mm, base.affine, header=base.header)

        png_path = overlays / f"{base_key}_overlay.png"
        save_overlay_png(base, mask_aligned, png_path, rotate_k=args.rotate_k)
        print(f"[overlay] saved {png_path}")

        if args.show:
            # if showing interactively, also display last saved image
            import matplotlib.image as mpimg
            img = mpimg.imread(png_path)
            plt.figure(figsize=(10,8)); plt.imshow(img); plt.axis("off"); plt.tight_layout(); plt.show()

    print("\nDone.")
    print(f"- NIfTI:   {nifti_raw}")
    print(f"- Prepped: {prepped}")
    print(f"- Output:  {out_dir}")
    print(f"- Overlays:{overlays}")
if __name__ == "__main__":
    main()
