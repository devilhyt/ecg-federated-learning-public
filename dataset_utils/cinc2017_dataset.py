import configparser
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class Cinc2017Dataset(Dataset):
    datasets = ["train", "valid", "test"]
    classes = ["N", "A", "O"]  # "~" removed
    label_encoder = {label: i for i, label in enumerate(classes)}
    label_decoder = {i: label for label, i in label_encoder.items()}
    num_classes = len(classes)

    def __init__(
        self,
        dataset: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        concatenate: bool = True,
    ) -> None:
        self.concatenate = concatenate

        # select the dataset partition
        if dataset not in self.datasets:
            raise ValueError(f"dataset must be one of {Cinc2017Dataset.datasets}")
        self.dataset = dataset

        # load dataset directories from config file
        config = configparser.ConfigParser()
        config.read("config.ini")
        self.dataset_dir = Path(config["data_preprocessing"]["dataset_dir"])
        self.preprocessed_dir = Path(config["data_preprocessing"]["dst_dir"])

        # load the annotation file
        self.dataset_df = pd.read_csv(self.dataset_dir / f"{self.dataset}.csv")

        # keep only the classes of interest
        self.dataset_df = self.dataset_df[self.dataset_df["label"].isin(self.classes)]

        # load signals and labels
        self.signals = []
        self.labels = []
        if self.concatenate:
            for record_name, label in self.dataset_df.itertuples(index=False):
                signal = np.loadtxt(self.preprocessed_dir / f"{record_name}.csv", ndmin=2)
                self.signals.append(signal)
                self.labels.extend([label] * len(signal))
            self.signals = np.concatenate(self.signals)
        else:
            for record_name, label in self.dataset_df.itertuples(index=False):
                signal = np.loadtxt(self.preprocessed_dir / f"{record_name}.csv", ndmin=2)
                self.signals.append(signal)
                self.labels.append(label)
            # self.signals = np.array(self.signals)

        # encode labels
        self.encoded_labels = np.array(
            [Cinc2017Dataset.label_encoder[label] for label in self.labels]
        )

        # set transforms
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.encoded_labels)

    def __getitem__(self, idx: int):
        data = self.signals[idx]
        label = self.encoded_labels[idx]

        if self.concatenate:
            if self.transform is not None:
                data = self.transform(data)
        else:
            if self.transform is not None:
                data = [self.transform(d)[0] for d in data]
                data = np.array(data)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return {"signal": data, "label": label}


if __name__ == "__main__":
    import time

    start_time = time.time()
    dataset = Cinc2017Dataset("train")
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
