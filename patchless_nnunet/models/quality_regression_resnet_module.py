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
from torchvision.transforms.functional import adjust_contrast, rotate
from torchvision.models.video.resnet import VideoResNet, BasicBlock, Conv2Plus1D

from patchless_nnunet.utils.inferers import SlidingWindowInferer
from patchless_nnunet.utils.softmax import softmax_helper
from patchless_nnunet.utils.tensor_utils import sum_tensor
from patchless_nnunet.utils.instantiators import instantiate_callbacks, instantiate_loggers
import torchio as tio


class QualityRegressionLitModule(LightningModule):
    """`nnUNet` training, evaluation and test strategy converted to PyTorch Lightning.

    nnUNetLitModule includes all nnUNet key features, including the test time augmentation, sliding
    window inference etc. Currently only 2D and 3D_fullres nnUNet are supported.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        loss: torch.nn.Module,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        optimizer_monitor: str = None,
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

        class R2Plus1dStem(nn.Sequential):
            """R(2+1)D stem is different than the default one as it uses separated 3D convolution
            """

            def __init__(self):
                super(R2Plus1dStem, self).__init__(
                    nn.Conv3d(1, 45, kernel_size=(1, 7, 7),
                              stride=(1, 2, 2), padding=(0, 3, 3),
                              bias=False),
                    nn.BatchNorm3d(45),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                              stride=(1, 1, 1), padding=(1, 0, 0),
                              bias=False),
                    nn.BatchNorm3d(64),
                    nn.ReLU(inplace=True))

        self.net = VideoResNet(stem=R2Plus1dStem,
                               num_classes=1,
                               block=BasicBlock,
                               conv_makers=[Conv2Plus1D] * 4,
                               layers=[2, 2, 2, 2])

        self.loss = loss

    def forward(self, img: Union[Tensor, MetaTensor]) -> Union[Tensor, MetaTensor]:  # noqa: D102
        return self.net(img)

    def training_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:  # noqa: D102
        img, label = batch["image"], batch["label"]

        # Need to handle carefully the multi-scale outputs from deep supervision heads
        pred = self.forward(img)
        pred = torch.sigmoid(pred)
        loss = self.loss(pred, label.unsqueeze(1))

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
        img, label = batch["image"], batch["label"]

        pred = self.forward(img)
        pred = torch.sigmoid(pred)
        loss = self.loss(pred, label.unsqueeze(1))

        if batch_idx == 0:
            self.log_images(
                title='sample',
                num_images=3,
                num_timesteps=1,
                axes_content={
                    'Image': img.squeeze(1).cpu().detach().numpy(),
                },
                info=[f"{pred.cpu().detach().numpy().squeeze(1)[i]:.4f} - {label[i]}" for i in range(3)]
            )

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
            sync_dist=True,
        )
        return {"val/loss": loss}


    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int
    ) -> dict[str, Tensor]:  # noqa: D102
        img, label, properties_dict = batch["image"], batch["label"], batch["image_meta_dict"]

        pred = self.forward(img)
        pred = torch.sigmoid(pred)
        loss = self.loss(pred, label.unsqueeze(1))

        self.log_images(
            title=f'test_{batch_idx}',
            num_images=1,
            num_timesteps=1,
            axes_content={
                'Image': img.squeeze(1).cpu().detach().numpy(),
            },
            info=[f"Pred: {pred.cpu().detach().numpy().squeeze(1)[i]:.4f} -- Label: {label[i]:.4f}" for i in range(len(label))]
        )

        self.log(
            "test/test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=self.trainer.datamodule.hparams.batch_size,
            sync_dist=True
        )
        return {"test/loss": loss}

    # def on_predict_start(self) -> None:  # noqa: D102
    #     super().on_predict_start()
    #     if self.trainer.datamodule is None:
    #         sw_batch_size = 2
    #     else:
    #         sw_batch_size = self.trainer.datamodule.hparams.batch_size
    #
    #     self.inferer = SlidingWindowInferer(
    #         roi_size=self.patch_size,
    #         sw_batch_size=sw_batch_size,
    #         overlap=self.hparams.sliding_window_overlap,
    #         mode=self.hparams.sliding_window_importance_map,
    #         cache_roi_weight_map=True,
    #     )
    #
    #     print(f"\n\nPredict step parameters: \n"
    #           f"    - Sliding window len: {self.hparams.sliding_window_len}\n"
    #           f"    - Sliding window overlap: {self.hparams.sliding_window_overlap}\n"
    #           f"    - Sliding window importance map: {self.hparams.sliding_window_importance_map}\n")
    #
    # def predict_step(self, batch: dict[str, Tensor], batch_idx: int):  # noqa: D102
    #     img, properties_dict = batch["image"], batch["image_meta_dict"]
    #
    #     self.patch_size = list([img.shape[-3], img.shape[-2], self.hparams.sliding_window_len])
    #     self.inferer.roi_size = self.patch_size
    #
    #     start_time = time.time()
    #     preds = self.tta_predict(img) if self.hparams.tta else self.predict(img)
    #     print(f"\nPrediction took {round(time.time() - start_time, 4)} (s).")
    #
    #     preds = preds.squeeze(0).cpu().detach().numpy()
    #     original_shape = properties_dict.get("original_shape").cpu().detach().numpy()[0]
    #     if len(preds.shape[1:]) == len(original_shape) - 1:
    #         preds = preds[..., None]
    #
    #     fname = properties_dict.get("case_identifier")[0]
    #     spacing = properties_dict.get("original_spacing").cpu().detach().numpy()[0]
    #     resampled_affine = properties_dict.get("resampled_affine").cpu().detach().numpy()[0]
    #     affine = properties_dict.get('original_affine').cpu().detach().numpy()[0]
    #
    #     final_preds = np.expand_dims(preds.argmax(0), 0)
    #     transform = tio.Resample(spacing)
    #     croporpad = tio.CropOrPad(original_shape)
    #     final_preds = croporpad(transform(tio.LabelMap(tensor=final_preds, affine=resampled_affine))).numpy()[0]
    #
    #     save_dir = os.path.join(self.trainer.default_root_dir, "inference_raw")
    #
    #     if self.hparams.save_predictions:
    #         self.save_mask(final_preds, fname, spacing, save_dir)
    #
    #     return final_preds

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
            configured_optimizer['monitor'] = 'val/val_loss'
        return configured_optimizer


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
