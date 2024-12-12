import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import lightning as L
import torchmetrics
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)

from .densenet_1d import densenet_ecg_1d
import copy


class DenseNet1dModule(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        show_valid_cm: bool = True,
        memory_efficient: bool = True,
        lr: float = 1e-3,
        min_lr: float = 1e-5,
        drop_rate: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes

        # settings
        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(1, 1, 3600)

        # model
        self.model = densenet_ecg_1d(
            drop_rate=drop_rate,
            num_classes=self.num_classes,
            memory_efficient=memory_efficient,
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.min_lr = min_lr

        # metrics
        metrics = torchmetrics.MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=self.num_classes),
                "f1": MulticlassF1Score(num_classes=self.num_classes),
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
    def _common_step(self, batch):
        x, y = batch["signal"], batch["label"]
        y_pred = self(x)
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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Return:
            loss: torch.Tensor
            y_hat: List[int]
            y: List[int]
        """
        loss, y_pred, y = self._common_step(batch)
        y_hat = torch.argmax(y_pred, dim=1).tolist()
        y = y.tolist()
        return loss, y_hat, y

    # hooks
    def on_train_epoch_start(self):
        self.trainer.progress_bar_metrics.clear()  # FIXME: hack to clear progress bar
        self.lr_log = self.lr_schedulers().get_last_lr()[0]

    def on_train_epoch_end(self):
        # place here to avoid repeated printing of the progress bar
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=self.min_lr)  # fmt: skip
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_loss",
            },
        }


class DenseNet1dModuleFL(DenseNet1dModule):
    """FedProx loss version of DenseNet1dModule"""

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
