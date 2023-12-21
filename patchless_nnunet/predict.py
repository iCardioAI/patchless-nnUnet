import os
from pathlib import Path
from typing import Tuple

import hydra
import nibabel as nib
import numpy as np
import torch
import torchio as tio
from lightning import LightningModule
from lightning import Trainer
from monai import transforms
from monai.data import DataLoader, ArrayDataset, MetaTensor
from monai.transforms import MapTransform
from monai.transforms import ToTensord
from omegaconf import DictConfig

from patchless_nnunet import utils, setup_root

log = utils.get_pylogger(__name__)


class PatchlessPreprocessd(MapTransform):
    """Load and preprocess data path given in dictionary keys.

    Dictionary must contain the following key(s): "image" and/or "label".
    """

    def __init__(
        self, keys, common_spacing,
    ) -> None:
        """Initialize class instance.

        Args:
            keys: Keys of the corresponding items to be transformed.
            common_spacing: Common spacing to resample the data.
        """
        super().__init__(keys)
        self.keys = keys
        self.common_spacing = np.array(common_spacing)

    def __call__(self, data: dict[str, str]):
        # load data
        d = dict(data)
        image = d["image"]

        image_meta_dict = {"case_identifier": os.path.basename(image._meta["filename_or_obj"]),
                           "original_shape": np.array(image.shape[1:]),
                           "original_spacing": np.array(image._meta["pixdim"][1:4].tolist())}
        original_affine = np.array(image._meta["original_affine"].tolist())
        image_meta_dict["original_affine"] = original_affine

        image = image.cpu().detach().numpy()
        # transforms and resampling
        if self.common_spacing is None:
            raise Exception("COMMON SPACING IS NONE!")
        transform = tio.Resample(self.common_spacing)
        resampled = transform(tio.ScalarImage(tensor=image, affine=original_affine))

        croporpad = tio.CropOrPad(self.get_desired_size(resampled.shape[1:]))
        resampled_cropped = croporpad(resampled)
        resampled_affine = resampled_cropped.affine

        d["image"] = resampled_cropped.numpy().astype(np.float32)

        image_meta_dict['resampled_affine'] = resampled_affine

        d["image_meta_dict"] = image_meta_dict
        return d

    def get_desired_size(self, current_shape, divisible_by=(32, 32, 4)):
        # get desired closest divisible bigger shape
        x = int(np.ceil(current_shape[0] / divisible_by[0]) * divisible_by[0])
        y = int(np.ceil(current_shape[1] / divisible_by[1]) * divisible_by[1])
        z = current_shape[2]
        return x, y, z


class PatchlessnnUnetPredictor:
    @classmethod
    def main(cls) -> None:
        """Runs the requested experiment."""
        # Set up the environment
        cls.pre_run_routine()

        # Run the system with config loaded by @hydra.main
        cls.run_system()

    @classmethod
    def pre_run_routine(cls) -> None:
        """Sets-up the environment before running the training/testing."""
        # Load environment variables from `.env` file if it exists
        # Load before hydra main to allow for setting environment variables with ${oc.env:ENV_NAME}
        setup_root()

    @staticmethod
    def load_and_prepare_nifti_img(img_path: Path):
        nifti_img = nib.load(img_path)

        data = np.expand_dims(nifti_img.get_fdata(), 0)
        hdr = nifti_img.header
        aff = nifti_img.affine

        meta = {"filename_or_obj": img_path.stem.split('.')[0].strip("_0000"),
                "pixdim": hdr['pixdim'],
                "original_affine": aff}
        return {'image': MetaTensor(torch.tensor(data, dtype=torch.float32), meta=meta)}

    @staticmethod
    def get_array_dataset(input_path):
        tensor_list = []
        # find all nifti files in input_path
        # open and get relevant information
        # add to list of data
        input_path = Path(input_path)
        if input_path.is_file():
            log.info(f"Predicting on file: {input_path}")
            tensor_list += [PatchlessnnUnetPredictor.load_and_prepare_nifti_img(input_path)]
        else:
            log.info(f"Predicting on file list found recursively in folder: {input_path}")
            for nifti_file_p in input_path.rglob('*.nii.gz'):
                tensor_list += [PatchlessnnUnetPredictor.load_and_prepare_nifti_img(nifti_file_p)]
        return tensor_list

    @staticmethod
    @hydra.main(version_base="1.3", config_path="configs", config_name="predict")
    @utils.task_wrapper
    def run_system(cfg: DictConfig) -> Tuple[dict, dict]:
        """Predict unseen cases with a given checkpoint.

        Currently, this method only supports inference for nnUNet models.

        This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
        failure. Useful for multiruns, saving info about the crash, etc.

        Args:
            cfg (DictConfig): Configuration composed by Hydra.

        Returns:
            Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.

        Raises:
            ValueError: If the checkpoint path is not provided.
        """
        # apply extra utilities
        # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
        utils.extras(cfg)

        if not cfg.ckpt_path:
            raise ValueError("ckpt_path must not be empty!")

        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

        object_dict = {
            "cfg": cfg,
            "model": model,
            "trainer": trainer,
        }

        preprocessed = PatchlessPreprocessd(keys='image', common_spacing=cfg.model.common_spacing)
        tf = transforms.compose.Compose([preprocessed, ToTensord(keys="image", track_meta=True)])

        numpy_arr_data = PatchlessnnUnetPredictor.get_array_dataset(cfg.input_folder)
        dataset = ArrayDataset(img=numpy_arr_data, img_transform=tf)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            shuffle=False,
        )

        log.info("Starting predicting!")
        log.info(f"Using checkpoint: {cfg.ckpt_path}")
        trainer.predict(model=model, dataloaders=dataloader, ckpt_path=cfg.ckpt_path)

        metric_dict = trainer.callback_metrics

        return metric_dict, object_dict


def main():
    """Run the script."""
    PatchlessnnUnetPredictor.main()


if __name__ == '__main__':
    PatchlessnnUnetPredictor.main()
