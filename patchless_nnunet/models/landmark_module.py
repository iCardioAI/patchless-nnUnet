import functools
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal, Optional, Union, Dict, List

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from einops.einops import rearrange
from lightning import LightningModule
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from monai.data import MetaTensor
from lightning.pytorch.loggers import TensorBoardLogger, CometLogger
from torch import Tensor
from torch.nn.functional import pad
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.transforms.functional import adjust_contrast, rotate

from patchless_nnunet.utils.inferers import SlidingWindowInferer
from patchless_nnunet.utils.softmax import softmax_helper
from patchless_nnunet.utils.tensor_utils import sum_tensor
from patchless_nnunet.utils import custom_dsnt
from patchless_nnunet.utils.coordinate_utils import CoordinateExtractor

import torchio as tio


class nnUNetPatchlessLitModule(LightningModule):
    """`nnUNet` training, evaluation and test strategy converted to PyTorch Lightning.

    nnUNetLitModule includes all nnUNet key features, including the test time augmentation, sliding
    window inference etc. Currently only 2D and 3D_fullres nnUNet are supported.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.nn.Module,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        optimizer_monitor: str = None,
        tta: bool = True,
        reg_lambda: float = 1.0,
        sliding_window_len: int = 4,
        sliding_window_overlap: float = 0.5,
        sliding_window_importance_map: bool = "gaussian",
        common_spacing: [float] = None,
        save_predictions: bool = True,
        save_npz: bool = False,
        name: str = "patchless_nnunet",
    ):
        """Saves the system's configuration in `hparams`. Initialize variables for training and
        validation loop.

        Args:
            net: Network architecture.
            optimizer: Optimizer.
            loss: Loss function.
            scheduler: Scheduler for training.
            tta: Whether to use the test time augmentation, i.e. flip.
            sliding_window_overlap: Minimum overlap for sliding window inference.
            sliding_window_importance_map: Importance map used for sliding window inference.
            save_prediction: Whether to save the test predictions.
            name: Name of the network.
        """
        super().__init__()
        # ignore net and loss as they are nn.module and will be saved automatically
        self.save_hyperparameters(logger=False, ignore=["net", "loss"])

        self.net = net

        # loss function
        self.loss = loss

        # parameter alpha for calculating moving average eval metrics
        # MA_metric = alpha * old + (1-alpha) * new
        self.val_eval_criterion_alpha = 0.9

        # current moving average dice
        self.val_eval_criterion_MA = None

        # best moving average dice
        self.best_val_eval_criterion_MA = None

        # list to store all the moving average dice during the training
        self.all_val_eval_metrics = []

        # list to store the metrics computed during evaluation steps
        self.online_eval_foreground_dc = []

        # we consider all the evaluation batches as a single element and only compute the global
        # foreground dice at the end of the evaluation epoch
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

        # store validation/test steps output as we can no longer receive steps output in
        # `on_validation_epoch_end` and `on_test_epoch_end`
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.coords_extractor = CoordinateExtractor(threshold=0.1)

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D102
        # to initialize some class variables that depend on the model
        self.threeD = len(self.net.patch_size) == 3
        self.patch_size = list(self.net.patch_size)
        self.num_classes = self.net.num_classes

        # create a dummy input to display model summary
        self.example_input_array = torch.rand(
            1, self.net.in_channels, *self.patch_size, device=self.device
        )

    def forward(self, img: Union[Tensor, MetaTensor]):  # noqa: D102
        unnormalized_heatmaps = self.net(img)
        return self.extract_coords(unnormalized_heatmaps)

    def extract_coords(self, unnormalized_heatmaps):
        heatmaps = custom_dsnt.flat_softmax(unnormalized_heatmaps)
        coords = custom_dsnt.dsnt(heatmaps)
        # Permute, do not invert last dim, as it changes ordering of values
        coords = coords.permute((0, 2, 1, 3))
        return coords, heatmaps

    def norm_to_coord(self, coords, img_shape):
        return (0.5 * (coords + 1)) * torch.tensor(img_shape[-3:-1]).flip(dims=(0,)).to(self.device)

    def reg_term(self, coords):
        # v1
        reg = torch.zeros_like(coords)
        winlen = 4
        for i in range(coords.shape[1]):
            reg[0, i] = (coords.squeeze(0)[max(0, i-winlen):i+winlen].mean(dim=0) - coords.squeeze(0)[i])**2

        # v2
        # shift_l = torch.roll(coords.clone(), 1, dims=-1)
        # shift_l[0, -1] = coords[0, -1]
        # shift_r = torch.roll(coords.clone(), -1, dims=-1)
        # shift_r[0, 0] = coords[0, 0]
        # reg = torch.abs(shift_r - coords) + torch.abs(shift_l - coords)

        return reg

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:  # noqa: D102
        # make sure to squeeze first dimension since batch comes from only one image
        img, label, lm_coords = batch["image"].squeeze(0), batch["label"].squeeze(0), \
                                            batch['landmark_coords'].squeeze(0)

        coords, heatmaps = self.forward(img)
        coords = self.norm_to_coord(coords, img.shape)
        mse = self.loss(coords, lm_coords)
        # reg = custom_dsnt.js_reg_losses(heatmaps, label)
        # euc = custom_dsnt.euclidean_losses(coords, lm_coords)
        # loss = reg.mean() + euc.mean()

        reg = self.reg_term(coords)

        loss = mse + self.hparams.reg_lambda * reg.mean()

        self.x_train_distances_r += list((lm_coords[..., 1, 1] - coords[..., 1, 1]).cpu().detach().numpy().flatten())
        self.y_train_distances_r += list((lm_coords[..., 1, 0] - coords[..., 1, 0]).cpu().detach().numpy().flatten())
        self.x_train_distances_l += list((lm_coords[..., 0, 1] - coords[..., 0, 1]).cpu().detach().numpy().flatten())
        self.y_train_distances_l += list((lm_coords[..., 0, 0] - coords[..., 0, 0]).cpu().detach().numpy().flatten())

        self.log(
            "train/reg_term",
            reg.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
            sync_dist=True,
        )
        self.log(
            "train/mse",
            mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
            sync_dist=True,
        )
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
            sync_dist=True,
        )
        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        # clear memory to avoid OOM when close to limit
        torch.cuda.empty_cache()
        fig = plt.figure()
        plt.scatter(self.x_train_distances_r, self.y_train_distances_r, c='b')
        plt.scatter(self.x_train_distances_l, self.y_train_distances_l, c='r')
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        self.trainer.logger.experiment.log_figure("Train", fig, step=self.current_epoch)
        plt.close()

    def on_train_epoch_start(self) -> None:
        self.x_train_distances_r = []
        self.y_train_distances_r = []
        self.x_train_distances_l = []
        self.y_train_distances_l = []

    def on_validation_epoch_start(self) -> None:
        self.x_val_distances_r = []
        self.y_val_distances_r = []
        self.x_val_distances_l = []
        self.y_val_distances_l = []

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:  # noqa: D102
        # squeeze first dim since batch comes from only one image
        img, label, lm_coords = batch["image"].squeeze(0), batch["label"].squeeze(0), \
                                batch['landmark_coords'].squeeze(0)

        coords, heatmaps = self.forward(img)
        coords = self.norm_to_coord(coords, img.shape)
        mse = self.loss(coords, lm_coords)
        # reg = custom_dsnt.js_reg_losses(heatmaps, label)
        # euc = custom_dsnt.euclidean_losses(coords, lm_coords)
        # loss = reg.mean() + euc.mean()

        reg = self.reg_term(coords)

        loss = mse + self.hparams.reg_lambda * reg.mean()

        # log spacing from center
        self.x_val_distances_r += list((lm_coords[..., 1, 1] - coords[..., 1, 1]).cpu().detach().numpy().flatten())
        self.y_val_distances_r += list((lm_coords[..., 1, 0] - coords[..., 1, 0]).cpu().detach().numpy().flatten())
        self.x_val_distances_l += list((lm_coords[..., 0, 1] - coords[..., 0, 1]).cpu().detach().numpy().flatten())
        self.y_val_distances_l += list((lm_coords[..., 0, 0] - coords[..., 0, 0]).cpu().detach().numpy().flatten())

        landmarks = np.zeros_like(img.cpu().numpy()).repeat(repeats=2, axis=1)
        landmark_points = coords.cpu().numpy()
        for i in range(img.shape[-1]):
            for j, point in enumerate(landmark_points[0, i]):
                y = int(point[0])
                x = int(point[1])
                landmarks[:, j, x-7:x+7, y-7:y+7, i] = 1

        if batch_idx % 10 == 0:
            self.log_images(
                title='sample',
                num_images=1,
                num_timesteps=min(img.shape[-1], 4),
                axes_content={
                    'Image': img.squeeze(1).cpu().detach().numpy(),
                    'Heatmaps_0': heatmaps[:, 0, ...].cpu().detach().numpy(),
                    'Heatmaps_1': heatmaps[:, 1, ...].cpu().detach().numpy(),
                    'Label_0': label[:, 0, ...].cpu().detach().numpy() + landmarks[:, 0, ...],
                    'Label_1': label[:, 1, ...].cpu().detach().numpy() + landmarks[:, 1, ...],
                },
                info=[f"{batch_idx}"]
            )

        self.log(
            "val/reg_term",
            reg.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
            sync_dist=True,
        )
        self.log(
            "val/mse",
            mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
            sync_dist=True,
        )
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
            sync_dist=True,
        )
        return {"val/loss": loss}

    def on_validation_epoch_end(self):  # noqa: D102
        fig = plt.figure()
        plt.scatter(self.x_val_distances_r, self.y_val_distances_r, c='b')
        plt.scatter(self.x_val_distances_l, self.y_val_distances_l, c='r')
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        self.trainer.logger.experiment.log_figure("Val", fig, step=self.current_epoch)
        plt.close()

        # clear memory to avoid OOM when close to limit
        torch.cuda.empty_cache()

    def on_test_start(self) -> None:  # noqa: D102
        super().on_test_start()
        if self.trainer.datamodule is None:
            sw_batch_size = 2
        else:
            sw_batch_size = self.trainer.datamodule.hparams.batch_size

        self.inferer = SlidingWindowInferer(
            roi_size=self.patch_size,
            sw_batch_size=sw_batch_size,
            overlap=self.hparams.sliding_window_overlap,
            mode=self.hparams.sliding_window_importance_map,
            cache_roi_weight_map=True,
        )

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:  # noqa: D102
        img, label, properties_dict, lm_coords = batch["image"], batch["label"], \
                                                                  batch["image_meta_dict"], \
                                                                  batch["landmark_coords"]

        self.patch_size = list([img.shape[-3], img.shape[-2], self.hparams.sliding_window_len])
        self.inferer.roi_size = self.patch_size

        start_time = time.time()
        preds = self.tta_predict(img, apply_softmax=False) if self.hparams.tta else self.predict(img, apply_softmax=False)
        print(f"\nPrediction took {round(time.time() - start_time, 4)} (s).")

        coords, heatmaps = self.extract_coords(preds)
        coords = self.norm_to_coord(coords, img.shape)

        mse = self.loss(coords, lm_coords)
        reg = self.reg_term(coords)
        loss = mse + self.hparams.reg_lambda * reg.mean()

        med_ae = torch.abs(coords - lm_coords).median()
        mae = torch.abs(coords - lm_coords).mean()
        mse = ((coords - lm_coords)**2).mean()

        landmarks = np.zeros_like(img.cpu().numpy()).repeat(repeats=2, axis=1)
        landmark_points = coords[0].cpu().numpy()
        for i in range(landmark_points.shape[0]):
            for j, point in enumerate(landmark_points[i]):
                y = int(point[0])
                x = int(point[1])
                landmarks[:, j, x - 5:x + 5, y - 5:y + 5, i] = 1

        import matplotlib.animation as animation
        preds = torch.sigmoid(preds)
        fig, ax = plt.subplots()

        # est0 = ax.scatter(x=coords[0, 0, 1], y=coords[0, 0, 0], c='g')
        # est1 = ax.scatter(x=coords[0, 0, 1], y=coords[0, 0, 0], c='g')
        # lab0 = ax.scatter(x=label_points[0, 1, 1], y=label_points[0, 1, 0], c='r')
        # lab1 = ax.scatter(x=label_points[0, 1, 1], y=label_points[0, 1, 0], c='r')
        im = ax.imshow(img[0, ..., 0].cpu().detach().numpy().T, cmap='grey')
        l0 = ax.imshow(heatmaps[0, 0, ..., 0].cpu().detach().numpy().T, alpha=0.3, cmap='jet')
        l1 = ax.imshow(heatmaps[0, 1, ..., 0].cpu().detach().numpy().T, alpha=0.3, cmap='jet')
        p0 = ax.imshow(landmarks[0, 0, ..., 0].T, alpha=0.3, cmap='jet')
        p1 = ax.imshow(landmarks[0, 1, ..., 0].T, alpha=0.3, cmap='jet')
        def animate(i):
            im.set_array(img[0, ..., i].cpu().detach().numpy().T)
            p0.set_array(landmarks[0, 0, ..., i].T)
            p1.set_array(landmarks[0, 1, ..., i].T)
            l0.set_array(heatmaps[0, 0, ..., i].cpu().detach().numpy().T)
            l1.set_array(heatmaps[0, 1, ..., i].cpu().detach().numpy().T)

            # est0.set_offsets((coords[i, 0, 1], coords[i, 0, 0]))
            # est1.set_offsets((coords[i, 1, 1], coords[i, 1, 0]))
            # lab0.set_offsets((label_points[i, 0, 1], label_points[i, 0, 0]))
            # lab1.set_offsets((label_points[i, 1, 1], label_points[i, 1, 0]))
            return im, p0, p1, l0, l1

        ani = animation.FuncAnimation(fig, animate, repeat=True,
                                      frames=img.shape[-1] - 1, interval=50)

        # To save the animation using Pillow as a gif
        writer = animation.PillowWriter(fps=10,
                                        metadata=dict(artist='Me'),
                                        bitrate=1800)
        ani.save(f"./animations/{properties_dict.get('case_identifier')[0]}.gif", writer=writer)

        #plt.show()
        plt.close()


        self.log_images(
            title=f'test_{batch_idx}',
            num_images=1,
            num_timesteps=min(img.shape[-1], 4),
            axes_content={
                'Image': img.squeeze(1).cpu().detach().numpy(),
                'Pred_1': heatmaps[:, 0, ...].cpu().detach().numpy(),
                'Pred_2': heatmaps[:, 1, ...].cpu().detach().numpy(),
                'Label_1': label[:, 0, ...].cpu().detach().numpy(),
                'Label_2': label[:, 1, ...].cpu().detach().numpy(),
            }
        )

        if self.hparams.save_predictions:
            preds = preds.squeeze(0).cpu().detach().numpy()
            original_shape = properties_dict.get("original_shape").cpu().detach().numpy()[0]
            if len(preds.shape[1:]) == len(original_shape) - 1:
                preds = preds[..., None]

            save_dir = os.path.join(self.trainer.default_root_dir, "testing_raw")

            fname = properties_dict.get("case_identifier")[0]
            spacing = properties_dict.get("original_spacing").cpu().detach().numpy()[0]
            resampled_affine = properties_dict.get("resampled_affine").cpu().detach().numpy()[0]
            affine = properties_dict.get('original_affine').cpu().detach().numpy()[0]

            final_preds = np.expand_dims(preds.argmax(0), 0)
            transform = tio.Resample(spacing)
            croporpad = tio.CropOrPad(original_shape)
            final_preds = croporpad(transform(tio.LabelMap(tensor=final_preds, affine=resampled_affine))).numpy()[0]

            self.save_mask(final_preds, fname, spacing.astype(np.float64), save_dir)

        self.log(
            "test/MAE",
            mae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
            sync_dist=True,
        )
        self.log(
            "test/MedianAE",
            med_ae,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
            sync_dist=True,
        )
        self.log(
            "test/MSE",
            mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
            sync_dist=True,
        )
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
            sync_dist=True,
        )

    def on_predict_start(self) -> None:  # noqa: D102
        super().on_predict_start()
        if self.trainer.datamodule is None:
            sw_batch_size = 2
        else:
            sw_batch_size = self.trainer.datamodule.hparams.batch_size

        self.inferer = SlidingWindowInferer(
            roi_size=self.patch_size,
            sw_batch_size=sw_batch_size,
            overlap=self.hparams.sliding_window_overlap,
            mode=self.hparams.sliding_window_importance_map,
            cache_roi_weight_map=True,
        )

        print(f"\n\nPredict step parameters: \n"
              f"    - Sliding window len: {self.hparams.sliding_window_len}\n"
              f"    - Sliding window overlap: {self.hparams.sliding_window_overlap}\n"
              f"    - Sliding window importance map: {self.hparams.sliding_window_importance_map}\n")

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int):  # noqa: D102
        img, properties_dict = batch["image"], batch["image_meta_dict"]

        self.patch_size = list([img.shape[-3], img.shape[-2], self.hparams.sliding_window_len])
        self.inferer.roi_size = self.patch_size

        start_time = time.time()
        preds = self.tta_predict(img) if self.hparams.tta else self.predict(img)
        print(f"\nPrediction took {round(time.time() - start_time, 4)} (s).")

        preds = preds.squeeze(0).cpu().detach().numpy()
        original_shape = properties_dict.get("original_shape").cpu().detach().numpy()[0]
        if len(preds.shape[1:]) == len(original_shape) - 1:
            preds = preds[..., None]

        fname = properties_dict.get("case_identifier")[0]
        spacing = properties_dict.get("original_spacing").cpu().detach().numpy()[0]
        resampled_affine = properties_dict.get("resampled_affine").cpu().detach().numpy()[0]
        affine = properties_dict.get('original_affine').cpu().detach().numpy()[0]

        final_preds = np.expand_dims(preds.argmax(0), 0)
        transform = tio.Resample(spacing)
        croporpad = tio.CropOrPad(original_shape)
        final_preds = croporpad(transform(tio.LabelMap(tensor=final_preds, affine=resampled_affine))).numpy()[0]

        save_dir = os.path.join(self.trainer.default_root_dir, "inference_raw")

        if self.hparams.save_predictions:
            self.save_mask(final_preds, fname, spacing, save_dir)

        return final_preds

    def configure_optimizers(self) -> dict[str, str | Any]:
        """Configures optimizers/LR schedulers.

        Returns:
            A dict with an `optimizer` key, an optional `lr_scheduler` if a scheduler is used and
            a metric to monitor (for some schedulers).
        """
        configured_optimizer = {"optimizer": self.hparams.optimizer(params=self.parameters())}
        if type(self.hparams.scheduler) in [_LRScheduler, functools.partial]:
            configured_optimizer["lr_scheduler"] = self.hparams.scheduler(
                optimizer=configured_optimizer["optimizer"]
            )
        if self.hparams.optimizer_monitor is not None:
            configured_optimizer['monitor'] = 'val/mean_dice'
        return configured_optimizer

    def predict(
        self, image: Union[Tensor, MetaTensor], apply_softmax: bool = True
    ) -> Union[Tensor, MetaTensor]:
        """Predict 3D images with sliding window inference.

        Args:
            image: Image to predict.
            apply_softmax: Whether to apply softmax to prediction.

        Returns:
            Aggregated prediction over all sliding windows.

        Raises:
            NotImplementedError: If the patch shape is not 2D nor 3D.
            ValueError: If 3D patch is requested to predict 2D images.
        """
        if len(image.shape) == 5:
            if len(self.patch_size) == 3:
                # Pad the last dimension to avoid 3D segmentation border artifacts
                pad_len = 6 if image.shape[-1] > 6 else image.shape[-1] - 1
                image = pad(image, (pad_len, pad_len, 0, 0, 0, 0), mode="reflect")
                pred = self.predict_3D_3Dconv_tiled(image, apply_softmax)
                # Inverse the padding after prediction
                return pred[..., pad_len:-pad_len]
            else:
                raise ValueError("Check your patch size. You dummy.")
        if len(image.shape) == 4:
            raise ValueError("No 2D images here. You dummy.")

    def tta_predict(
        self, image: Union[Tensor, MetaTensor], apply_softmax: bool = True
    ) -> Union[Tensor, MetaTensor]:
        """Predict with test time augmentation.

        Args:
            image: Image to predict.
            apply_softmax: Whether to apply softmax to prediction.

        Returns:
            Aggregated prediction over number of flips.
        """
        preds = self.predict(image, apply_softmax)
        factors = [1.1, 0.9, 1.25, 0.75]
        translations = [40, 60, 80, 120]
        rotations = [5, 10, -5, -10]

        for factor in factors:
            preds += self.predict(adjust_contrast(image.permute((4, 0, 1, 2, 3)), factor).permute((1, 2, 3, 4, 0)), apply_softmax)

        def x_translate_left(img, amount=20):
            return pad(img, (0, 0, 0, 0, amount, 0), mode="constant")[:, :, :-amount, :, :]
        def x_translate_right(img, amount=20):
            return pad(img, (0, 0, 0, 0, 0, amount), mode="constant")[:, :, amount:, :, :]
        def y_translate_up(img, amount=20):
            return pad(img, (0, 0, amount, 0, 0, 0), mode="constant")[:, :, :, :-amount, :]
        def y_translate_down(img, amount=20):
            return pad(img, (0, 0, 0, amount, 0, 0), mode="constant")[:, :, :, amount:, :]

        for translation in translations:
            preds += x_translate_right(self.predict(x_translate_left(image, translation), apply_softmax), translation)
            preds += x_translate_left(self.predict(x_translate_right(image, translation), apply_softmax), translation)
            preds += y_translate_down(self.predict(y_translate_up(image, translation), apply_softmax), translation)
            preds += y_translate_up(self.predict(y_translate_down(image, translation), apply_softmax), translation)

        # TODO: optimize this for compute time
        for rotation in rotations:
            rotated = torch.zeros_like(image)
            for i in range(image.shape[-1]):
                rotated[0, :, :, :, i] = rotate(image[0, :, :, :, i], angle=rotation)
            rot_pred = self.predict(rotated, apply_softmax)
            for i in range(image.shape[-1]):
                rot_pred[0, :, :, :, i] = rotate(rot_pred[0, :, :, :, i], angle=-rotation)
            preds += rot_pred

        preds /= len(factors) + len(translations) * 4 + len(rotations) + 1
        return preds

    def predict_3D_3Dconv_tiled(
        self, image: Union[Tensor, MetaTensor], apply_softmax: bool = True
    ) -> Union[Tensor, MetaTensor]:
        """Predict 3D image with 3D model.

        Args:
            image: Image to predict.
            apply_softmax: Whether to apply softmax to prediction.

        Returns:
            Aggregated prediction over all sliding windows.

        Raises:
            ValueError: If image is not 3D.
        """
        if not len(image.shape) == 5:
            raise ValueError("image must be (b, c, w, h, d)")

        if apply_softmax:
            return softmax_helper(self.sliding_window_inference(image))
        else:
            return self.sliding_window_inference(image)

    def sliding_window_inference(
        self, image: Union[Tensor, MetaTensor]
    ) -> Union[Tensor, MetaTensor]:
        """Inference using sliding window.

        Args:
            image: Image to predict.

        Returns:
            Predicted logits.
        """
        return self.inferer(
            inputs=image,
            network=self.net,
        )

    @staticmethod
    def get_properties(image_meta_dict: dict) -> OrderedDict:
        """Convert values in image meta dictionary loaded from torch.tensor to normal list/boolean.

        Args:
            image_meta_dict: Dictionary containing image meta information.

        Returns:
            Converted properties dictionary.
        """
        properties_dict = OrderedDict()
        properties_dict["original_shape"] = image_meta_dict["original_shape"][0].tolist()
        properties_dict["resampling_flag"] = image_meta_dict["resampling_flag"].item()
        properties_dict["shape_after_cropping"] = image_meta_dict["shape_after_cropping"][0].tolist()
        if properties_dict.get("resampling_flag"):
            properties_dict["anisotropy_flag"] = image_meta_dict["anisotropy_flag"].item()
        properties_dict["crop_bbox"] = image_meta_dict["crop_bbox"][0].tolist()
        properties_dict["case_identifier"] = image_meta_dict["case_identifier"][0]
        properties_dict["original_spacing"] = image_meta_dict["original_spacing"][0].tolist()
        properties_dict["spacing_after_resampling"] = image_meta_dict["spacing_after_resampling"][0].tolist()

        return properties_dict

    def log_images(
        self, title: str, num_images: int, num_timesteps: int, axes_content: Dict[str, np.ndarray], info: Optional[List[str]] = None
    ):
        """Log images to Logger if it is a TensorBoardLogger or CometLogger.
        Args:
            title: Name of the figure.
            num_images: Number of images to log.
            num_timesteps: Number of timesteps per image.
            axes_content: Mapping of axis name and image.
            info: Additional info to be appended to title for each image.
        """
        if self.trainer.global_rank == 0:  # only log image to global rank 0 for multi-gpu training
            for i in range(num_images):
                fig, axes = plt.subplots(num_timesteps, len(axes_content.keys()), squeeze=False)
                if info is not None:
                    name = f"{title}_{info[i]}_{i}"
                else:
                    name = f"{title}_{i}"
                plt.suptitle(name)

                for j, (ax_title, imgs) in enumerate(axes_content.items()):
                    for k in range(num_timesteps):
                        if len(imgs.shape) == 4:
                            axes[k, j].imshow(imgs[i, ..., k].squeeze().T)
                        if len(imgs.shape) == 5:  # blend
                            axes[k, j].imshow(imgs[0, i, ..., k].squeeze().T)
                            axes[k, j].imshow(imgs[1, i, ..., k].squeeze().T, cmap='jet')
                        if k == 0:
                            axes[0, j].set_title(ax_title)
                        axes[k, j].tick_params(left=False,
                                               bottom=False,
                                               labelleft=False,
                                               labelbottom=False)
                plt.subplots_adjust(wspace=0, hspace=0)

                if isinstance(self.trainer.logger, TensorBoardLogger):
                    self.trainer.logger.experiment.add_figure("{}_{}".format(title, i), fig, self.current_epoch)
                if isinstance(self.trainer.logger, CometLogger):
                    self.trainer.logger.experiment.log_figure("{}_{}".format(title, i), fig, step=self.current_epoch)

                plt.close()


if __name__ == "__main__":
    from typing import List

    import hydra
    import omegaconf
    import pyrootutils
    from hydra import compose, initialize
    from lightning import Callback, LightningDataModule, LightningModule, Trainer
    from omegaconf import OmegaConf
    from patchless_nnunet.utils.instantiators import instantiate_callbacks

    root = pyrootutils.setup_root(__file__, pythonpath=True)

    with initialize(version_base="1.2", config_path="../configs/model"):
        cfg = compose(config_name="patchless_3d.yaml")
        print(OmegaConf.to_yaml(cfg))

    cfg.scheduler.max_decay_steps = 1
    cfg.save_predictions = False
    nnunet: LightningModule = hydra.utils.instantiate(cfg)

    cfg = omegaconf.OmegaConf.load(root / "patchless_nnunet" / "configs" / "datamodule" / "patchless_nnunet.yaml")
    cfg.data_dir = str(root / "data")
    cfg.dataset_name = "icardio_subset"
    cfg.batch_size = 1
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg)

    cfg = omegaconf.OmegaConf.load(root / "patchless_nnunet" / "configs" / "callbacks" / "nnunet_patchless.yaml")
    callbacks: List[Callback] = instantiate_callbacks(cfg)

    trainer = Trainer(
        max_epochs=2,
        deterministic=False,
        limit_train_batches=21,
        limit_val_batches=10,
        limit_test_batches=2,
        gradient_clip_val=12,
        precision='16-mixed',
        callbacks=callbacks,
        accelerator="gpu",
        devices=1,
    )

    trainer.fit(model=nnunet, datamodule=datamodule)
    print("Starting testing!")
    trainer.test(model=nnunet, datamodule=datamodule)
