import json
from pathlib import Path

from scipy.ndimage import gaussian_filter

from patchless_nnunet.datamodules.patchless_nnunet_datamodule import PatchlessnnUnetDataModule, get_img_subpath, \
    PatchlessnnUnetDataset
from torch.utils.data import Dataset

import nibabel as nib
import numpy as np
import torch
import torchio as tio


class LandmarkDataset(PatchlessnnUnetDataset):
    def __init__(self,
                 df,
                 data_path,
                 common_spacing=None,
                 max_window_len=None,
                 use_dataset_fraction=1.0,
                 max_batch_size=None,
                 max_tensor_volume=5000000,
                 shape_divisible_by=(32, 32, 4),
                 test=False,
                 landmark_path=None,
                 landmark_qc_path=None,
                 *args, **kwargs):
        super().__init__(df, data_path, common_spacing, max_window_len, use_dataset_fraction, max_batch_size,
                         max_tensor_volume, shape_divisible_by, test)

        self.landmark_path = landmark_path
        print(len(self.df))

    def __getitem__(self, idx):
        # Get paths and open images
        sub_path = get_img_subpath(self.df.iloc[idx])
        img_nifti = nib.load(self.data_path + '/img/' + sub_path)
        img = img_nifti.get_fdata() / 255
        landmark_points = np.load(self.landmark_path + sub_path.replace("_0000", "").replace(".nii.gz", ".npy"))
        original_shape = np.asarray(list(img.shape))

        # limit size of tensor so it can fit on GPU
        if not self.test:
            if img.shape[0] * img.shape[1] * img.shape[2] > self.max_tensor_volume:
                time_len = int(self.max_tensor_volume // (img.shape[0] * img.shape[1]))
                img = img[..., :time_len]
                landmark_points = landmark_points[:time_len]

        # transforms and resampling
        if self.common_spacing is None:
            raise Exception("COMMON SPACING IS NONE!")
        transform = tio.Resample(self.common_spacing)
        resampled = transform(tio.ScalarImage(tensor=np.expand_dims(img, 0), affine=img_nifti.affine))

        croporpad = tio.CropOrPad(self.get_desired_size(resampled.shape[1:]))
        resampled_cropped = croporpad(resampled)
        resampled_affine = resampled_cropped.affine
        img = resampled_cropped.tensor

        # convert points to heatmaps
        landmarks = np.zeros_like(img).repeat(repeats=2, axis=0)
        landmark_points_norm = landmark_points.copy()
        for i in range(img.shape[-1]):
            for j, point in enumerate(landmark_points[i]):
                y = int(point[0] / original_shape[0] * img.shape[1])
                x = int(point[1] / original_shape[1] * img.shape[2])

                landmark_points_norm[i, j, 0] = (point[0] / original_shape[0] * 2) - 1
                landmark_points_norm[i, j, 1] = (point[1] / original_shape[1] * 2) - 1

                landmarks[j, x, y, i] = 1
                landmarks[j, ..., i] = gaussian_filter(landmarks[j, ..., i], sigma=10)
                landmarks[idx, ..., i] = (landmarks[idx, ..., i] - np.min(landmarks[idx, ..., i])) / \
                                         (np.max(landmarks[idx, ..., i]) - np.min(landmarks[idx, ..., i]))
        # from matplotlib import pyplot as plt
        # plt.figure()
        # plt.imshow(landmarks[0, :, :, 0])
        # plt.show()
        landmarks = torch.tensor(landmarks)
        landmark_points_norm = torch.tensor(landmark_points_norm)

        if not self.test:
            if self.max_window_len:
                # use partial time window, create as many batches as possible with it unless self.max_batch_size not set
                dynamic_batch_size = max(img.shape[-1] // self.max_window_len, 1) \
                    if not self.max_batch_size or not (self.max_batch_size > 0 and
                                                       (self.max_batch_size * self.max_window_len) < img.shape[-1]) \
                    else self.max_batch_size
                b_img = []
                b_landmarks = []
                b_points = []
                for _ in range(dynamic_batch_size):
                    start_idx = np.random.randint(low=0, high=max(img.shape[-1] - self.max_window_len, 1))
                    b_img += [img[..., start_idx:start_idx + self.max_window_len]]
                    b_landmarks += [landmarks[..., start_idx:start_idx + self.max_window_len]]
                    b_points += [landmark_points_norm[start_idx:start_idx + self.max_window_len, ...]]
                img = torch.stack(b_img)
                landmarks = torch.stack(b_landmarks)
                landmark_points_norm = torch.stack(b_points)
            else:
                # use entire available time window
                # must unsqueeze to accommodate code in train/val step
                img = img.unsqueeze(0)
                landmarks = landmarks.unsqueeze(0)
                landmark_points_norm = landmark_points_norm.unsqueeze(0)

        return {'image': img.type(torch.float32),
                'label': landmarks.type(torch.float32),
                'landmark_points': landmark_points_norm.type(torch.float32),  # in original coordinate system
                'image_meta_dict': {'case_identifier': self.df.iloc[idx]['dicom_uuid'],
                                    'original_shape': original_shape,
                                    'original_spacing': img_nifti.header['pixdim'][1:4],
                                    'original_affine': img_nifti.affine,
                                    'resampled_affine': resampled_affine,
                                    }
                }


class LandmarkDataModule(PatchlessnnUnetDataModule):
    def __init__(self, dataset=LandmarkDataset, landmark_qc_path=None, *args, **kwargs):
        super().__init__(dataset=dataset, *args, **kwargs)
        self.df = self.df[self.df['dicom_uuid'].isin(self.get_valid_landmark_sequences(landmark_qc_path))]

    def get_valid_landmark_sequences(self, json_files_path):
        if not json_files_path:
            return []
        json_dir = Path(json_files_path)
        seq_dict = {}
        for p in json_dir.glob("*.json"):
            with open(p, 'r') as f:
                data = json.load(f)
            for i in range(len(data[1]['data'])):
                seq = data[1]['data'][i]

                dicom = seq['filename'].split("_")[0]
                seq_dict[dicom] = seq_dict.get(dicom, []) + [True] if "Pass" in seq['status'] else [False]

        for key, value in seq_dict.items():
            seq_dict[key] = all(value)
        return [k for k, v in seq_dict.items() if v]


if __name__ == "__main__":
    import pyrootutils
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    root = pyrootutils.setup_root(__file__, pythonpath=True)

    dl = LandmarkDataModule(data_dir="/data/icardio/processed/",
                            common_spacing=(0.37, 0.37, 1.0),
                            max_window_len=4,
                            max_batch_size=None,
                            dataset_name='',
                            csv_file_name='subset_csv/merged_subset_w_split_0_1-18.csv',
                            splits_column='splits_0',
                            num_workers=1,
                            batch_size=1,
                            use_dataset_fraction=0.1,
                            landmark_path='/data/icardio/processed/valve_landmarks_2/',
                            landmark_qc_path='/home/local/USHERBROOKE/juda2901/dev/qc/landmarks2/')
    dl.setup()
    for batch in iter(dl.val_dataloader()):
        bimg = batch['image'].squeeze(0)
        blabel = batch['label'].squeeze(0)
        print(bimg.shape)
        print(blabel.shape)
        plt.figure()
        plt.imshow(bimg[0, 0, :, :, 1].T)

        plt.figure()
        plt.imshow(blabel[0, 0, :, :, 1].T)

        plt.figure()
        plt.imshow(blabel[0, 1, :, :, 1].T)
        plt.show()
