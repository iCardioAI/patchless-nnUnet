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
from matplotlib import pyplot as plt

from monai.data import MetaTensor
from lightning.pytorch.loggers import TensorBoardLogger, CometLogger
from torch import Tensor
from torch.nn.functional import pad
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.transforms.functional import adjust_contrast, rotate, adjust_brightness

from patchless_nnunet.utils.inferers import SlidingWindowInferer
from patchless_nnunet.utils.softmax import softmax_helper
from patchless_nnunet.utils.tensor_utils import sum_tensor
from patchless_nnunet.utils.instantiators import instantiate_callbacks, instantiate_loggers
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
        sliding_window_len: int = 24,
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

        # loss function (CE - Dice), min = -1
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

    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D102
        # to initialize some class variables that depend on the model
        self.threeD = len(self.net.patch_size) == 3
        self.patch_size = list(self.net.patch_size)
        self.num_classes = self.net.num_classes

        # create a dummy input to display model summary
        self.example_input_array = torch.rand(
            1, self.net.in_channels, *self.patch_size, device=self.device
        )

    def forward(self, img: Union[Tensor, MetaTensor]) -> Union[Tensor, MetaTensor]:  # noqa: D102
        return self.net(img)

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:  # noqa: D102
        # make sure to squeeze first dimension since batch comes from only one image
        img, label = batch["image"].squeeze(0), batch["label"].squeeze(0)

        # Need to handle carefully the multi-scale outputs from deep supervision heads
        pred = self.forward(img)
        loss = self.compute_loss(pred, label)

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

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:  # noqa: D102
        # squeeze first dim since batch comes from only one image
        img, label = batch["image"].squeeze(0), batch["label"].squeeze(0)

        # Only the highest resolution output is returned during the validation
        pred = self.forward(img)
        loss = self.loss(pred, label)

        # Compute the stats that will be used to compute the final dice metric during the end of
        # epoch
        num_classes = pred.shape[1]
        pred_softmax = softmax_helper(pred)
        pred_seg = pred_softmax.argmax(1)
        label = label[:, 0]
        axes = tuple(range(1, len(label.shape)))
        tp_hard = torch.zeros((label.shape[0], num_classes - 1)).to(pred_seg.device.index)
        fp_hard = torch.zeros((label.shape[0], num_classes - 1)).to(pred_seg.device.index)
        fn_hard = torch.zeros((label.shape[0], num_classes - 1)).to(pred_seg.device.index)

        for c in range(1, num_classes):
            tp_hard[:, c - 1] = sum_tensor(
                (pred_seg == c).float() * (label == c).float(), axes=axes
            )
            fp_hard[:, c - 1] = sum_tensor(
                (pred_seg == c).float() * (label != c).float(), axes=axes
            )
            fn_hard[:, c - 1] = sum_tensor(
                (pred_seg != c).float() * (label == c).float(), axes=axes
            )

        tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
        fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
        fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

        self.online_eval_foreground_dc.append(
            list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8))
        )
        self.online_eval_tp.append(list(tp_hard))
        self.online_eval_fp.append(list(fp_hard))
        self.online_eval_fn.append(list(fn_hard))

        self.validation_step_outputs.append({"val/loss": loss})

        if batch_idx == 0:
            self.log_images(
                title='sample',
                num_images=1,
                num_timesteps=min(img.shape[-1], 4),
                axes_content={
                    'Image': img.squeeze(1).cpu().detach().numpy(),
                    'Prediction': pred_seg.cpu().detach().numpy(),
                    'Stack': np.stack((img.squeeze(1).cpu().detach().numpy(),
                                       pred_seg.cpu().detach().numpy())),
                }
            )

        return {"val/loss": loss}

    def on_validation_epoch_end(self):  # noqa: D102
        loss = self.metric_mean("val/loss", self.validation_step_outputs)
        self.validation_step_outputs.clear()  # free memory

        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        global_dc_per_class = [
            i if not np.isnan(i) else 0.0
            for i in [
                2 * i / (2 * i + j + k)
                for i, j, k in zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)
            ]
        ]

        self.all_val_eval_metrics.append(np.mean(global_dc_per_class))

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

        self.update_eval_criterion_MA()
        self.maybe_update_best_val_eval_criterion_MA()

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
        self.log(
            "val/dice_MA",
            self.val_eval_criterion_MA,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
            sync_dist=True,
        )
        for label, dice in zip(range(len(global_dc_per_class)), global_dc_per_class):
            self.log(
                f"val/dice/{label}",
                np.round(dice, 4),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=self.trainer.datamodule.hparams.batch_size,
                sync_dist=True,
            )
        self.log(
            f"val/mean_dice",
            np.mean(global_dc_per_class),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
            sync_dist=True,
        )

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
        img, label, properties_dict = batch["image"], batch["label"], batch["image_meta_dict"]

        self.patch_size = list([img.shape[-3], img.shape[-2], self.hparams.sliding_window_len])
        self.inferer.roi_size = self.patch_size

        start_time = time.time()
        preds = self.tta_predict(img) if self.hparams.tta else self.predict(img)
        print(f"\nPrediction took {round(time.time() - start_time, 4)} (s).")

        num_classes = preds.shape[1]
        pred_seg = preds.argmax(1)
        label = label[:, 0]
        axes = tuple(range(1, len(label.shape)))
        tp_hard = torch.zeros((label.shape[0], num_classes - 1)).to(pred_seg.device.index)
        fp_hard = torch.zeros((label.shape[0], num_classes - 1)).to(pred_seg.device.index)
        fn_hard = torch.zeros((label.shape[0], num_classes - 1)).to(pred_seg.device.index)
        for c in range(1, num_classes):
            tp_hard[:, c - 1] = sum_tensor(
                (pred_seg == c).float() * (label == c).float(), axes=axes
            )
            fp_hard[:, c - 1] = sum_tensor(
                (pred_seg == c).float() * (label != c).float(), axes=axes
            )
            fn_hard[:, c - 1] = sum_tensor(
                (pred_seg != c).float() * (label == c).float(), axes=axes
            )

        tp_hard = tp_hard.sum(0, keepdim=False)
        fp_hard = fp_hard.sum(0, keepdim=False)
        fn_hard = fn_hard.sum(0, keepdim=False)
        test_dice = (2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)

        test_dice = torch.mean(test_dice, 0)

        self.log(
            "test/dice",
            test_dice,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
            sync_dist=True,
        )

        self.log_images(
            title=f'test_{batch_idx}',
            num_images=1,
            num_timesteps=min(img.shape[-1], 4),
            axes_content={
                'Image': img.squeeze(1).cpu().detach().numpy(),
                'Prediction': pred_seg.cpu().detach().numpy(),
                'Stack': np.stack((img.squeeze(1).cpu().detach().numpy(),
                                   pred_seg.cpu().detach().numpy())),
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

        self.test_step_outputs.append({"test/dice": test_dice})

        return {"test/dice": test_dice}

    def on_test_epoch_end(self):  # noqa: D102
        mean_dice = self.metric_mean("test/dice", self.test_step_outputs)
        self.test_step_outputs.clear()  # free memory

        self.log(
            "test/mean_dice",
            mean_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
            sync_dist=True
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

    def configure_optimizers(self) -> dict[Literal["optimizer", "lr_scheduler", "monitor"], Any]:
        """Configures optimizers/LR schedulers.

        Returns:
            A dict with an `optimizer` key, and an optional `lr_scheduler` if a scheduler is used.
        """
        configured_optimizer = {"optimizer": self.hparams.optimizer(params=self.parameters())}
        if type(self.hparams.scheduler) is _LRScheduler:
            configured_optimizer["lr_scheduler"] = self.hparams.scheduler(
                optimizer=configured_optimizer["optimizer"]
            )
        if self.hparams.optimizer_monitor is not None:
            configured_optimizer['monitor'] = 'val/mean_dice'
        return configured_optimizer

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Save extra information in checkpoint, i.e. the evaluation metrics for all epochs.

        Args:
            checkpoint: Checkpoint dictionary.
        """
        checkpoint["all_val_eval_metrics"] = self.all_val_eval_metrics

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Load information from checkpoint to class attribute, i.e. the evaluation metrics for all
        epochs.

        Args:
            checkpoint: Checkpoint dictionary.
        """
        self.all_val_eval_metrics = checkpoint["all_val_eval_metrics"]

    def compute_loss(
        self, preds: Union[Tensor, MetaTensor], label: Union[Tensor, MetaTensor]
    ) -> float:
        """Compute the multi-scale loss if deep supervision is set to True.

        Args:
            preds: Predicted logits.
            label: Ground truth label.

        Returns:
            Train loss.
        """
        if self.net.deep_supervision:
            loss = self.loss(preds[0], label)
            for i, pred in enumerate(preds[1:]):
                downsampled_label = nn.functional.interpolate(label, pred.shape[2:])
                loss += 0.5 ** (i + 1) * self.loss(pred, downsampled_label)
            c_norm = 1 / (2 - 2 ** (-len(preds)))
            return c_norm * loss
        return self.loss(preds, label)

    def predict(
        self, image: Union[Tensor, MetaTensor], apply_softmax: bool = True
    ) -> Union[Tensor, MetaTensor]:
        """Predict 2D/3D images with sliding window inference.

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
        self, image: Union[Tensor, MetaTensor], apply_softmax: bool = True, conserve_intermediate=False
    ) -> Union[Tensor, MetaTensor]:
        """Predict with test time augmentation.

        Args:
            image: Image to predict.
            apply_softmax: Whether to apply softmax to prediction.

        Returns:
            Aggregated prediction over number of flips.
        """
        preds = self.predict(image, apply_softmax)
        pred_list = []
        factors = [1.4, 1.1, 0.9, 1.25, 0.75, 0.6]
        translations = [] # [40, 60, 80, 120] Not good for TTA unc
        rotations = [5, 10, -5, -10]

        for factor in factors:
            p = self.predict(adjust_contrast(image.permute((4, 0, 1, 2, 3)), factor).permute((1, 2, 3, 4, 0)), apply_softmax)
            pred_list += [p]
            preds += p

        for factor in factors:
            p = self.predict(adjust_brightness(image.permute((4, 0, 1, 2, 3)), factor).permute((1, 2, 3, 4, 0)), apply_softmax)
            pred_list += [p]
            preds += p
        def x_translate_left(img, amount=20):
            return pad(img, (0, 0, 0, 0, amount, 0), mode="constant", value=0)[:, :, :-amount, :, :]
        def x_translate_right(img, amount=20):
            return pad(img, (0, 0, 0, 0, 0, amount), mode="constant", value=0)[:, :, amount:, :, :]
        def y_translate_up(img, amount=20):
            return pad(img, (0, 0, amount, 0, 0, 0), mode="constant", value=0)[:, :, :, :-amount, :]
        def y_translate_down(img, amount=20):
            return pad(img, (0, 0, 0, amount, 0, 0), mode="constant", value=0)[:, :, :, amount:, :]

        for translation in translations:
            p = x_translate_right(self.predict(x_translate_left(image, translation), apply_softmax), translation)
            pred_list += [p]
            preds += p
            p = x_translate_left(self.predict(x_translate_right(image, translation), apply_softmax), translation)
            pred_list += [p]
            preds += p
            p = y_translate_down(self.predict(y_translate_up(image, translation), apply_softmax), translation)
            pred_list += [p]
            preds += p
            p = y_translate_up(self.predict(y_translate_down(image, translation), apply_softmax), translation)
            pred_list += [p]
            preds += p

        # TODO: optimize this for compute time
        for rotation in rotations:
            rotated = torch.zeros_like(image)
            for i in range(image.shape[-1]):
                rotated[0, :, :, :, i] = rotate(image[0, :, :, :, i], angle=rotation, fill=0)
            rot_pred = self.predict(rotated, apply_softmax)
            for i in range(image.shape[-1]):
                rot_pred[0, :, :, :, i] = rotate(rot_pred[0, :, :, :, i], angle=-rotation, fill=0)
            p = rot_pred
            pred_list += [p]
            preds += p

        preds /= len(factors) + len(translations) * 4 + len(rotations) + 1
        if conserve_intermediate:
            return preds, pred_list
        else:
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
    def metric_mean(name: str, outputs: list[dict[str, Tensor]]) -> Tensor:
        """Average metrics across batch dimension at epoch end.

        Args:
            name: Name of metrics to average.
            outputs: List containing outputs dictionary returned at step end.

        Returns:
            Averaged metrics tensor.
        """
        return torch.stack([out[name] for out in outputs]).mean(dim=0)

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
        properties_dict["shape_after_cropping"] = image_meta_dict["shape_after_cropping"][
            0
        ].tolist()
        if properties_dict.get("resampling_flag"):
            properties_dict["anisotropy_flag"] = image_meta_dict["anisotropy_flag"].item()
        properties_dict["crop_bbox"] = image_meta_dict["crop_bbox"][0].tolist()
        properties_dict["case_identifier"] = image_meta_dict["case_identifier"][0]
        properties_dict["original_spacing"] = image_meta_dict["original_spacing"][0].tolist()
        properties_dict["spacing_after_resampling"] = image_meta_dict["spacing_after_resampling"][
            0
        ].tolist()

        return properties_dict

    def save_mask(
        self, preds: np.ndarray, fname: str, spacing: np.ndarray, save_dir: Union[str, Path]
    ) -> None:
        """Save segmentation mask to the given save directory.

        Args:
            preds: Predicted segmentation mask.
            fname: Filename to save.
            spacing: Spacing to save the segmentation mask.
            save_dir: Directory to save the segmentation mask.
        """
        print(f"Saving segmentation for {fname}...")

        os.makedirs(save_dir, exist_ok=True)

        preds = preds.astype(np.uint8)
        itk_image = sitk.GetImageFromArray(rearrange(preds, "w h d ->  d h w"))
        itk_image.SetSpacing(spacing)
        sitk.WriteImage(itk_image, os.path.join(save_dir, fname + ".nii.gz"))

    def update_eval_criterion_MA(self):
        """Update moving average validation loss."""
        if self.val_eval_criterion_MA is None:
            self.val_eval_criterion_MA = self.all_val_eval_metrics[-1]
        else:
            self.val_eval_criterion_MA = (
                self.val_eval_criterion_alpha * self.val_eval_criterion_MA
                + (1 - self.val_eval_criterion_alpha) * self.all_val_eval_metrics[-1]
            )

    def maybe_update_best_val_eval_criterion_MA(self):
        """Update moving average validation metrics."""
        if self.best_val_eval_criterion_MA is None:
            self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
        if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
            self.best_val_eval_criterion_MA = self.val_eval_criterion_MA

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
                            axes[k, j].imshow(imgs[1, i, ..., k].squeeze().T, alpha=0.3)
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
