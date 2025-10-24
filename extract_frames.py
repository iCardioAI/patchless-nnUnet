#!/usr/bin/env python3
"""
extract_four_frames.py

Usage:
    python extract_four_frames.py input_multiframe.dcm [--start N]

Creates (in ./four_frame_dicoms):
    frames_<start>_to_<start+3>.dcm  (a 4-frame DICOM)

Notes:
- Forces UNCOMPRESSED transfer syntax (Explicit VR Little Endian).
- Copies basic Patient/Study context and geometry where sensible.
- Handles grayscale and RGB frames (all 4 must match shape/channels).
"""

import os
import sys
import argparse
from typing import List, Tuple
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import (
    generate_uid,
    SecondaryCaptureImageStorage,
    ExplicitVRLittleEndian,
)

# ---- default start frame (0-based) ----
DEFAULT_START = 10
FRAME_COUNT = 10


def _to_uint_dtype(frame: np.ndarray) -> Tuple[np.ndarray, int, int, int, int]:
    """
    Ensure frame is an unsigned integer type and return:
    (frame_as_uint, bits_allocated, bits_stored, high_bit, pixel_representation)
    """
    if np.issubdtype(frame.dtype, np.integer):
        if frame.dtype.kind == "u":
            arr = frame
            bits = arr.dtype.itemsize * 8
            return arr, bits, bits, bits - 1, 0  # unsigned
        else:
            f = frame
            minv = int(f.min())
            if minv < 0:
                f = f - minv
            maxv = int(f.max()) if f.size else 0
            bits = max(8, int(np.ceil(np.log2(maxv + 1))) if maxv > 0 else 8)
            if bits <= 8:
                arr = f.astype(np.uint8)
                return arr, 8, 8, 7, 0
            else:
                arr = f.astype(np.uint16)
                return arr, 16, 16, 15, 0
    else:
        f = frame.astype(np.float32)
        fmin, fmax = float(np.min(f)), float(np.max(f))
        if fmax > fmin:
            f = (f - fmin) / (fmax - fmin)
        else:
            f = np.zeros_like(f)
        arr = (f * 65535.0 + 0.5).astype(np.uint16)
        return arr, 16, 16, 15, 0


def _channel_last(frame: np.ndarray) -> np.ndarray:
    """Ensure channel-last if input is (C,H,W) or (C,H,W,??)."""
    if frame.ndim == 3 and frame.shape[0] in (3, 4) and frame.shape[-1] not in (3, 4):
        frame = np.moveaxis(frame, 0, -1)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]
    return frame


def _make_four_frame_ds(
    src_ds: FileDataset, frames: List[np.ndarray], start_idx: int, series_uid: str
) -> FileDataset:
    """
    Build a 4-frame DICOM from a list of 4 frames (np.ndarrays).
    Uses Secondary Capture Image Storage. UNCOMPRESSED (Explicit VR Little Endian).
    """
    assert len(frames) == 10, "Expected exactly 4 frames"

    # Normalize shape (channel-last), convert dtype for each frame
    frames_proc = []
    first_uint = None
    bits_alloc = bits_store = high_bit = pixel_rep = None

    for f in frames:
        f = _channel_last(f)
        fu, ba, bs, hb, pr = _to_uint_dtype(f)
        if first_uint is None:
            first_uint = fu
            bits_alloc, bits_store, high_bit, pixel_rep = ba, bs, hb, pr
            target_dtype = fu.dtype
        else:
            fu = fu.astype(target_dtype, copy=False)
        frames_proc.append(fu)

    # Validate consistent spatial dims and channels
    shapes = [fp.shape for fp in frames_proc]
    if len(set(shapes)) != 1:
        raise ValueError(f"Inconsistent frame shapes for 4-frame output: {shapes}")

    h, w = frames_proc[0].shape[:2]
    is_rgb = (len(frames_proc[0].shape) == 3 and frames_proc[0].shape[-1] == 3)

    # --- file meta ---
    file_meta = Dataset()
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.10.543.1"

    # --- main dataset ---
    out = FileDataset(
        filename_or_obj=None,
        dataset=Dataset(),
        file_meta=file_meta,
        preamble=b"\x00" * 128,
    )

    # Patient
    for tag in ("PatientName", "PatientID", "PatientBirthDate", "PatientSex"):
        if tag in src_ds:
            out[tag] = src_ds[tag]

    # Study
    for tag in (
        "StudyInstanceUID",
        "StudyDate",
        "StudyTime",
        "AccessionNumber",
        "ReferringPhysicianName",
        "StudyID",
    ):
        if tag in src_ds:
            out[tag] = src_ds[tag]

    # Series (new Series for extracted pack)
    out.SeriesInstanceUID = series_uid
    out.SeriesNumber = getattr(src_ds, "SeriesNumber", 999)
    out.Modality = getattr(src_ds, "Modality", "OT")

    # SOP
    out.SOPClassUID = SecondaryCaptureImageStorage
    out.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    # Instance info
    out.InstanceNumber = int(start_idx)

    # Geometry (copy if available; applies to all frames)
    if hasattr(src_ds, "ImagePositionPatient"):
        out.ImagePositionPatient = src_ds.ImagePositionPatient
    if hasattr(src_ds, "ImageOrientationPatient"):
        out.ImageOrientationPatient = src_ds.ImageOrientationPatient
    if hasattr(src_ds, "PixelSpacing"):
        out.PixelSpacing = src_ds.PixelSpacing
    if hasattr(src_ds, "SliceThickness"):
        out.SliceThickness = src_ds.SliceThickness

    # Photometric / samples-per-pixel
    if is_rgb:
        out.PhotometricInterpretation = "RGB"
        out.SamplesPerPixel = 3
        out.PlanarConfiguration = 0  # interleaved by pixel
    else:
        out.PhotometricInterpretation = getattr(
            src_ds, "PhotometricInterpretation", "MONOCHROME2"
        )
        out.SamplesPerPixel = 1

    out.Rows = int(h)
    out.Columns = int(w)
    out.NumberOfFrames = FRAME_COUNT

    # Pixel data characteristics
    out.BitsAllocated = bits_alloc
    out.BitsStored = bits_store
    out.HighBit = high_bit
    out.PixelRepresentation = pixel_rep  # 0 unsigned, 1 signed

    # Keep rescale if present (applies globally)
    if "RescaleIntercept" in src_ds:
        out.RescaleIntercept = src_ds.RescaleIntercept
    if "RescaleSlope" in src_ds:
        out.RescaleSlope = src_ds.RescaleSlope

    # Remove per-frame/multi-frame attributes we're not populating
    for tag in (
        "PerFrameFunctionalGroupsSequence",
        "SharedFunctionalGroupsSequence",
        "FrameIncrementPointer",
        "FrameTime",
        "FrameTimeVector",
    ):
        if tag in out:
            del out[tag]

    # Stack frames into a single multi-frame pixel buffer
    if is_rgb:
        # (F,H,W,3) -> bytes
        stack = np.stack(frames_proc, axis=0)  # shape (4, H, W, 3)
    else:
        # (F,H,W) -> bytes
        stack = np.stack(frames_proc, axis=0)  # shape (4, H, W)

    out.PixelData = stack.tobytes()

    # Typical Secondary Capture attributes
    out.ConversionType = "WSD"
    out.SecondaryCaptureDeviceManufacturer = "generated-by-script"

    # Explicit VR Little Endian
    out.is_little_endian = True
    out.is_implicit_VR = False

    return out


def _detect_multiframe_and_accessor(ds: FileDataset, px: np.ndarray):
    """
    Return (num_frames, accessor) where accessor(i) gives the i-th frame (C-last if RGB).
    """
    if px.ndim == 2:
        raise ValueError("This DICOM appears to be single-frame; no multi-frame index to extract.")
    elif px.ndim == 3:
        # (frames, rows, cols) or (rows, cols, samples)
        if getattr(ds, "NumberOfFrames", None) == px.shape[0]:
            num_frames = px.shape[0]
            get_frame = lambda i: px[i]
        elif px.shape[-1] in (3, 4) and getattr(ds, "NumberOfFrames", 1) > 1:
            # Rare (rows, cols, samples) but says multi-frame—unsupported layout
            raise ValueError("Unsupported multi-frame layout (rows, cols, samples) without leading frame dim.")
        else:
            raise ValueError("Unable to detect multi-frame dimension reliably.")
    elif px.ndim == 4:
        # (frames, rows, cols, channels)
        num_frames = px.shape[0]
        get_frame = lambda i: px[i]
    else:
        raise ValueError(f"Unsupported pixel array shape: {px.shape}")
    return num_frames, get_frame


def extract_four_consecutive_frames(dcm_path: str, start_index: int) -> None:
    ds = pydicom.dcmread(dcm_path)
    px = ds.pixel_array  # pydicom/GDCM will decompress if needed

    num_frames, get_frame = _detect_multiframe_and_accessor(ds, px)

    end_index = start_index + FRAME_COUNT - 1
    if start_index < 0 or end_index >= num_frames:
        raise ValueError(
            f"Requested 4 frames starting at {start_index} exceed bounds (0..{num_frames-1})."
        )

    frames = [get_frame(i) for i in range(start_index, start_index + FRAME_COUNT)]

    out_dir = "four_frame_dicoms"
    os.makedirs(out_dir, exist_ok=True)

    # New Series UID for this 4-pack extraction (keeps 4-frame groups together)
    new_series_uid = generate_uid()

    out_ds = _make_four_frame_ds(
        src_ds=ds, frames=frames, start_idx=start_index, series_uid=new_series_uid
    )
    out_name = os.path.join(out_dir, f"frames_{start_index}_to_{start_index+FRAME_COUNT-1}.dcm")
    out_ds.save_as(out_name)
    print(f"Saved 10-frame DICOM ({start_index}..{start_index+FRAME_COUNT-1}) → {out_name}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to input multi-frame DICOM")
    ap.add_argument("--start", type=int, default=DEFAULT_START, help="0-based starting frame index")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_four_consecutive_frames(args.input, args.start)
