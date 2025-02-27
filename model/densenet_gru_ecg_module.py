import inspect
from typing import Literal

import lightning as L
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torchmetrics
from torch import Tensor
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassAveragePrecision,
)

from .densenet_gru_ecg import DenseNetGruEcg


class DenseNetGruEcgModule(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        max_epochs: int | None = None,
        input_length: int | None = None,
        lr_scheduler_mode: (
            Literal["multi_step", "plateau", "cosine_annealing"] | None
        ) = None,
        lr: float = 1e-3,
        min_lr: float = 1e-5,
        show_valid_cm: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_epochs = max_epochs

        # settings
        self.save_hyperparameters()
        if input_length is not None:
            self.example_input_array = Tensor(1, 1, input_length)

        # model
        model_kwargs = inspect.signature(DenseNetGruEcg).parameters.keys()
        model_kwargs = {k: v for k, v in kwargs.items() if k in model_kwargs}
        self.model = DenseNetGruEcg(
            num_classes=num_classes,
            **model_kwargs,
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr_scheduler_mode = lr_scheduler_mode
        self.lr = lr
        self.min_lr = min_lr

        # metrics
        metrics = torchmetrics.MetricCollection(
            {
                "acc": MulticlassAccuracy(
                    num_classes=self.num_classes, average="micro"
                ),
                "f1": MulticlassF1Score(num_classes=self.num_classes),
                "auroc": MulticlassAUROC(num_classes=self.num_classes),
                "auprc": MulticlassAveragePrecision(num_classes=self.num_classes),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="valid_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.valid_cm = MulticlassConfusionMatrix(num_classes=self.num_classes)
        self.test_cm = MulticlassConfusionMatrix(num_classes=self.num_classes)
        self.show_valid_cm = show_valid_cm

        # log
        self.lr_log = float("inf")
        self.valid_cm_log = None
        self.test_cm_log = None

    def forward(self, x):
        return self.model(x)

    # steps
    def _common_step(self, batch) -> tuple[Tensor, Tensor, Tensor]:
        x, y = batch["signal"], batch["label"]
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        return loss, y_pred, y

    def training_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch)
        self.train_metrics.update(y_pred, y)
        self.log("lr", self.lr_log, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log_dict(self.train_metrics, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch)
        self.valid_metrics.update(y_pred, y)
        self.valid_cm.update(y_pred, y)
        self.log("valid_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log_dict(self.valid_metrics, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch)
        self.test_metrics.update(y_pred, y)
        self.test_cm.update(y_pred, y)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log_dict(self.test_metrics, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch)
        return {"loss": loss, "y_pred": y_pred, "y": y}

    # hooks
    def on_train_epoch_start(self):
        self.trainer.progress_bar_metrics.clear()  # avoid showing metrics from previous epoch
        self.lr_log = [group["lr"] for group in self.optimizers().param_groups][0]

    def on_train_epoch_end(self):
        # place show_valid_cm here to avoid repeated printing of the progress bar
        if self.show_valid_cm:
            print(f"Valid Confusion Matrix:\n{self.valid_cm_log}")

    def on_validation_epoch_end(self):
        self.valid_cm_log = self.valid_cm.compute()
        self.valid_cm.reset()

    def on_test_epoch_end(self):
        self.test_cm_log = self.test_cm.compute()
        self.test_cm.reset()

    # configurations
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-3)

        if self.lr_scheduler_mode == "plateau":
            config = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler.ReduceLROnPlateau(
                        optimizer, factor=0.5, patience=3, min_lr=self.min_lr
                    ),
                    "monitor": "valid_loss",
                },
            }
        elif self.lr_scheduler_mode == "multi_step" and self.max_epochs is not None:
            config = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler.MultiStepLR(
                        optimizer,
                        milestones=[
                            int(self.max_epochs * 0.5),
                            int(self.max_epochs * 0.75),
                        ],
                        gamma=0.5,
                    ),
                },
            }
        elif (
            self.lr_scheduler_mode == "cosine_annealing" and self.max_epochs is not None
        ):
            config = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer,
                        T_0=int(self.max_epochs * 0.2),
                        T_mult=1,
                        eta_min=self.min_lr,
                    ),
                },
            }
        else:
            config = {"optimizer": optimizer}

        return config


class DenseNetGruEcgModuleFL(DenseNetGruEcgModule):
    """FedProx loss version of DenseNetGruEcgModule"""

    def __init__(
        self,
        proximal_mu: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.proximal_mu = proximal_mu
        self.global_params = None

    def init_global_params(self):
        self.global_params = [p.detach().clone() for p in self.model.parameters()]

    # steps
    def _common_step(self, batch):
        x, y = batch["signal"], batch["label"]
        y_pred = self(x)

        if self.global_params is None:
            self.init_global_params()

        proximal_term = 0.0
        for local_weights, global_weights in zip(
            self.model.parameters(), self.global_params
        ):
            proximal_term += torch.square((local_weights - global_weights).norm(2))

        loss = self.loss_fn(y_pred, y) + (self.proximal_mu / 2) * proximal_term
        return loss, y_pred, y