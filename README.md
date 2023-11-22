# patchless-nnUnet

Code repository for patchless version of the nnUnet.

(Inspired by https://github.com/creatis-myriad/ASCENT/)

# Description

This code base allows you to train a 3D Unet for a multi-classs segmentation task with images stored in nifti format.

## Install

1. Download the repository:
   ```bash
   # clone project
   git clone https://github.com/arnaudjudge/patchless-nnUnet
   cd patchless_nnunet
   ```
2. Create a virtual environment (Conda is strongly recommended):
   ```bash
   # create conda environment
   conda create -n ascent python=3.10
   conda activate ascent
   ```
3. Install [PyTorch](https://pytorch.org/get-started/locally/) according to instructions. Grab the one with GPU for faster training:
   ```bash
   # example for linux or Windows
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```
4. Install the project in editable mode and its dependencies:
   ```bash
   pip install -e .
   ```

# Data

The images used with this project are 3d (H x W x timestep) and are saved in the nifti format.
The dataloader assumes a file hierarchy format associated with a csv dataframe.
Each image is saved in a folder associated with it's study id, and view.
The filename corresponds to the dicom uuid of the image.
Images and their respective ground truth labels are stored in separate, by identically formatted folders ('img' and 'segmentation') residing one next to another.
Images and their segmentation map that are to be used in the training must be identified with the 'valid_segmentation' column in the dataframe.
* Currently, images are required to have the _0000 suffix.
## Example:

dataframe.csv:

| index | dicom_uuid        | ... | study  | view | valid_segmentation |
|-------|-------------------|-----|--------|------|--------------------|
| 0     | di-1234-ABCD-5678 | ... | study1 | a2c  | True               |
| 1     | di-5678-EFGH-1234 | ... | study2 | a4c  | True               |
| ...   | ...               | ... | ...    | ...  | ...                |
| 500   | di-WXYZ-6789-ABCD | ... | study3 | a4c  | True               |

The associated files would be:
```
├── dataframe.csv
├── img
│    ├── study1
│    │   └── a2c
│    │      └──di-1234-ABCD-5678_0000.nii.gz
│    ├── study2
│    │   └── a4c
│    │       └──di-5678-EFGH-1234_0000.nii.gz
│    ├── ...
│    └── study3
│        └── a4c
│            └──di-WXYZ-6789-ABCD_0000.nii.gz
├── segmentation
│    ├── study1
│    │   └── a2c
│    │      └──di-1234-ABCD-5678.nii.gz
│    ├── study2
│    │   └── a4c
│    │       └──di-5678-EFGH-1234.nii.gz
│    ├── ...
│    └── study3
│        └── a4c
│            └──di-WXYZ-6789-ABCD.nii.gz
```
# Training

To train the network, use the following command:
```bash
python runner.py experiment=patchless
```
Many options are available through hydra CLI override syntax or through modification/addition of config files.

For example:
```bash
python runner.py experiment=patchless trainer.max_epochs=20
```

# Inference

To run inference on a set of new images, use the following command. Make sure to specify input and output folder as well as model checkpoint to use.
```bash
python predictor.py input_path=<INPUT_PATH> output_path=<OUTPUT_PATH> ckpt_path=<CKPT_PATH>
```
