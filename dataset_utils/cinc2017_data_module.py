import configparser

import lightning as L
import numpy as np
import torch
from datasets import ClassLabel, Dataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms.v2 import Lambda

from .cinc2017_dataset import Cinc2017Dataset
from .transforms import *


class Cinc2017DataModule(L.LightningDataModule):
    num_classes = Cinc2017Dataset.num_classes

    def __init__(
        self,
        num_workers: int = 4,
        batch_size: int = 64,
    ) -> None:
        super().__init__()
        config = configparser.ConfigParser()
        config.read("config.ini")
        self.random_seed = config.getint("data_preprocessing", "random_seed")
        dst_freq = config.getint("data_preprocessing", "dst_freq")
        dst_time = config.getint("data_preprocessing", "dst_time")
        dst_length = dst_freq * dst_time

        # dataloader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers

        # transforms
        self.train_transforms = torch.nn.Sequential(
            RandomTimeScale(factor=0.2, p=0.3),
            RandomNoise(
                signal_freq=dst_freq,
                noise_amplitude=0.2,
                noise_freq=dst_freq // 10,
                p=0.3,
            ),
            Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            Crop(length=dst_length, mode="random"),
            MinMaxNorm(),
            Lambda(lambda x: x.unsqueeze(0)),
        )
        self.transforms = torch.nn.Sequential(
            Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
            Crop(length=dst_length, mode="head"),
            MinMaxNorm(),
            Lambda(lambda x: x.unsqueeze(0)),
        )
        self.target_transform = Lambda(lambda x: torch.tensor(x, dtype=torch.long))

    def setup(self, stage: str) -> None:
        if stage == "fit" and (
            not hasattr(self, "train_set") or not hasattr(self, "valid_set")
        ):
            self.train_set = Cinc2017Dataset(
                dataset="train",
                transform=self.train_transforms,
                target_transform=self.target_transform,
            )
            self.valid_set = Cinc2017Dataset(
                dataset="valid",
                transform=self.transforms,
                target_transform=self.target_transform,
            )
        elif stage == "validate" and not hasattr(self, "valid_set"):
            self.valid_set = Cinc2017Dataset(
                dataset="valid",
                transform=self.transforms,
                target_transform=self.target_transform,
            )
        elif stage in ["test", "predict"] and not hasattr(self, "test_set"):
            self.test_set = Cinc2017Dataset(
                dataset="test",
                transform=self.transforms,
                target_transform=self.target_transform,
            )

    def train_dataloader(self):
        # calculate class weights for weighted random sampler
        samples_weight = self._get_samples_weight(self.train_set.encoded_labels)

        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            sampler=WeightedRandomSampler(samples_weight, len(samples_weight)),
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return self.test_dataloader()

    def _get_samples_weight(self, encoded_labels):
        label_count = np.bincount(encoded_labels, minlength=self.num_classes)
        class_weight = 1.0 / label_count
        samples_weight = class_weight[encoded_labels]
        return samples_weight


class Cinc2017DataModuleFL(Cinc2017DataModule):
    """Federated Learning Data Module.
    This class wraps flower partitioners and huggingface datasets
    to provide dataloaders for federated learning.
    """

    def __init__(
        self,
        num_workers: int = 4,
        batch_size: int = 64,
        num_partitions: int = 5,
    ) -> None:
        super().__init__(num_workers=num_workers, batch_size=batch_size)

        # partitioner parameters
        self.num_partitions = num_partitions

    def setup(self, stage: str, non_iid: bool = False, alpha: float = 0.5) -> None:
        if stage == "fl_client":
            # Wrap flower partitioners and huggingface datasets for federated learning.
            self.client_set = Cinc2017Dataset(dataset="train")
            data = {"signal": self.client_set.signals, "label": self.client_set.labels}
            dataset = Dataset.from_dict(data)
            dataset = dataset.cast_column(
                "label", ClassLabel(names=self.client_set.classes)
            )

            if non_iid:
                self.client_set_partitioner = DirichletPartitioner(
                    num_partitions=self.num_partitions,
                    partition_by="label",
                    alpha=alpha,
                    seed=self.random_seed,
                    self_balancing=False,
                )
            else:
                self.client_set_partitioner = IidPartitioner(
                    num_partitions=self.num_partitions
                )

            self.client_set_partitioner.dataset = dataset
        elif stage == "train_eval":
            self.train_eval_set = Cinc2017Dataset(
                dataset="train",
                transform=self.transforms,
                target_transform=self.target_transform,
            )
        else:
            super().setup(stage)

    def train_eval_dataloader(self):
        """Provide a dataLoader for evaluating the training set without applying data augmentation."""
        return DataLoader(
            self.train_eval_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def client_dataloaders(self, partition_id: int) -> tuple[DataLoader, DataLoader]:
        """Provide dataloaders for federated learning clients."""
        partition = self.client_set_partitioner.load_partition(partition_id)
        partition_train_test = partition.train_test_split(
            test_size=0.2,
            stratify_by_column="label",
            seed=self.random_seed,
        )

        # calculate class weights for weighted random sampler
        samples_weight = self._get_samples_weight(
            partition_train_test["train"]["label"]
        )

        # apply transforms
        partition_train_test = partition_train_test.with_transform(
            self._apply_transforms
        )

        train_dataloader = DataLoader(
            partition_train_test["train"],
            batch_size=self.batch_size,
            sampler=WeightedRandomSampler(samples_weight, len(samples_weight)),
            num_workers=self.num_workers,
        )
        valid_dataloader = DataLoader(
            partition_train_test["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return train_dataloader, valid_dataloader

    def client_dataloader(self, partition_id: int) -> DataLoader:
        """Provide dataloaders for federated learning clients."""
        partition = self.client_set_partitioner.load_partition(partition_id)

        # calculate class weights for weighted random sampler
        samples_weight = self._get_samples_weight(partition["label"])

        # apply transforms
        partition = partition.with_transform(self._apply_transforms)

        train_dataloader = DataLoader(
            partition,
            batch_size=self.batch_size,
            sampler=WeightedRandomSampler(samples_weight, len(samples_weight)),
            num_workers=self.num_workers,
        )

        return train_dataloader

    def _apply_transforms(self, batch):
        batch["signal"] = [
            self.train_transforms(np.array(signal)) for signal in batch["signal"]
        ]
        return batch
