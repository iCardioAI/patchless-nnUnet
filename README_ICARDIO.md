## iCardio dicom preprocessing -> Patchless nnU-Net Inference

### Steps:

1. Convert dicoms to NIfTI with dcm2niix

2. Preprocess NIfTI to match Arnaud’s model expectations (LPS orientation, voxel sizes, scalar channel)

3. Run inference with patchless-nnUNet

4. Visualize the predicted segmentation overlay on the input volumes

---

## Details

### Pick your input dicom folder and an output NIfTI folder:

```
# example paths
DICOM_DIR=~/Desktop/dicoms
NIFTI_DIR=~/Desktop/nnunet_data

mkdir -p "$NIFTI_DIR"
dcm2niix -z y -f %p_%s -o "$NIFTI_DIR" "$DICOM_DIR"
```
After this step, you have NIfTI files in ~/Desktop/nnunet_data.

**Note: The preprocessing script expects NIfTI inputs (not dicom).**

### Preprocess NIfTI for patchless-nnUNet

**Run prep_nnunet_inputs.py to:**
- Force single-channel float32 (handles RGB/void dtype)

- Reorient to LPS axis codes

- Set voxel spacing to (0.37, 0.37, 1.0) in the header

- Output files with the *_0000.nii.gz suffix (required by nnU-Net style)


**Key settings inside the script:**

- IN_DIR = Path("~/Desktop/dicoms").expanduser() → Set this to your NIfTI folder, e.g. ~/Desktop/nnunet_data

- OUT_DIR = IN_DIR → Outputs go next to inputs (change if you want a separate dir)

- GLOB = "case_*.nii.gz" → Adjust the glob if your files have different names

- Produces files ending with _0000.nii.gz

Run the script: `python prep_nnunet_inputs.py`


### Run patchless-nnUNet inference

- Go to dev branch: `git checkout dev`

- Put the nifti files in `data/` folder.

- Install its dependencies in a virtual env

```
cd /home/users/aleksandar.s/patchless-nnUnet
source venv/bin/activate  # or your environment activation

python -m patchless_nnunet.predict \
  model=patchless_nnunet \
  input_folder=data \
  output_folder=output \
  ckpt_path=actor_3d_ANAT_LM_BEST_1.ckpt \
  apply_eq_hist=true tta=false \
  num_workers=2 pin_memory=false \
  trainer.accelerator=gpu trainer.devices=1 trainer.precision=32
```

Results will appear under `output/` (e.g., output/inference_raw/…).

**Note: The nifti inputs must have *_0000.nii.gz suffix!**

### Visualize base + predicted mask overlay

Use the script `visualize_nifti_overlay.py` to display overlays of model predictions on input images.

The script:

- Converts both NIfTIs (base + mask) to float32

- Aligns them (resamples or adopts base affine if shapes match)

- Overlays the mask with transparency for quick inspection


Example configuration (inside the script):

```
BASE = "nifti_input/case_1_0000.nii.gz"
MASK = "inference_raw/case_1.nii.gz"
```


## TL;DR

- brew install dcm2niix

- dcm2niix -z y -o ~/Desktop/nnunet_data ~/Desktop/dicoms

- Update IN_DIR in prep_nnunet_inputs.py to ~/Desktop/nnunet_data and run it

- Copy _0000.nii.gz files to patchless-nnUnet/data on the server

- `git checkout dev`

- Run the patchless_nnunet.predict command above

- Visualize base and predicted mask
