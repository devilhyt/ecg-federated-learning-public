import random
import numpy as np
import neurokit2 as nk
import torch.nn as nn
import torch
from torch import Tensor
from typing import Literal


class RandomTimeScale(nn.Module):
    # TODO: Using Tensor as input.
    def __init__(self, factor: int | float, p: int | float = 0.5) -> None:
        super().__init__()
        if not (isinstance(factor, (int, float)) and factor > 0):
            raise ValueError("factor must be a int or float greater than 0")
        if not (isinstance(p, (int, float)) and 0 <= p <= 1):
            raise ValueError("p must be a int or float between 0 and 1")
        self.factor = factor
        self.p = p

    def forward(self, signal):
        if torch.rand(1) < self.p:
            # determine the desired length of the new signal
            length = len(signal)
            desired_length = int(
                length * (1 + random.uniform(-self.factor, self.factor))
            )
            # resample the signal
            signal = nk.signal_resample(signal, desired_length)
        return signal


class RandomNoise(nn.Module):
    # TODO: Using Tensor as input.
    def __init__(
        self,
        signal_freq: int | float,
        noise_amplitude: int | float,
        noise_freq: int | float,
        p=0.5,
    ) -> None:
        super().__init__()
        if not (isinstance(signal_freq, (int, float)) and signal_freq > 0):
            raise ValueError("signal_freq must be a int or float greater than 0")
        if not (isinstance(noise_amplitude, (int, float))):
            raise ValueError("noise_amplitude must be a int or float")
        if not (isinstance(noise_freq, (int, float)) and noise_freq > 0):
            raise ValueError("noise_freq must be a int or float greater than 0")
        if not (isinstance(p, (int, float)) and 0 <= p <= 1):
            raise ValueError("p must be a int or float between 0 and 1")
        self.signal_freq = signal_freq
        self.noise_amplitude = noise_amplitude
        self.noise_freq = noise_freq
        self.p = p

    def forward(self, signal):
        if torch.rand(1) < self.p:
            signal = nk.signal.signal_distort(
                signal,
                sampling_rate=self.signal_freq,
                noise_shape="gaussian",
                noise_amplitude=self.noise_amplitude,
                noise_frequency=self.noise_freq,
            )
        return signal


class Crop(nn.Module):
    def __init__(
        self,
        length: int,
        mode: Literal["start", "random"] = "start",
        padding_if_needed: bool = True,
    ) -> None:
        super().__init__()
        if not (isinstance(length, int) and length > 0):
            raise ValueError("length must be a positive integer")
        if mode not in ["start", "random"]:
            raise ValueError("mode must be either 'start' or 'random'")
        if not isinstance(padding_if_needed, bool):
            raise ValueError("padding_if_needed must be a boolean")
        self.length = length
        self.mode = mode
        self.padding_if_needed = padding_if_needed

    def forward(self, signal: Tensor) -> Tensor:
        if not isinstance(signal, Tensor):
            raise ValueError("Input signal must be a Tensor")

        signal_length = signal.size(-1)

        if signal_length == self.length:
            return signal

        if signal_length > self.length:
            # crop
            if self.mode == "start":
                return signal[: self.length]
            else:
                start = random.randint(0, signal_length - self.length)
                return signal[start : start + self.length]

        if signal_length < self.length:
            if self.padding_if_needed:
                padding_size = self.length - signal_length
                return torch.nn.functional.pad(signal, (0, padding_size), "constant", 0)
            return signal


class RandomInvert(nn.Module):
    def __init__(self, signal_freq: int | float, p: int | float = 0.5) -> None:
        super().__init__()
        if not (isinstance(signal_freq, (int, float)) and signal_freq > 0):
            raise ValueError("signal_freq must be a int or float greater than 0")
        if not (isinstance(p, (int, float)) and 0 <= p <= 1):
            raise ValueError("p must be a int or float between 0 and 1")
        self.signal_freq = signal_freq
        self.p = p

    def forward(self, signal: Tensor) -> Tensor:
        if not isinstance(signal, Tensor):
            raise ValueError("Input signal must be a Tensor")
        if torch.rand(1) < self.p:
            signal = -signal + torch.mean(signal) * 2
        return signal


class RandomMask(nn.Module):
    def __init__(
        self, ratio_from: int | float, ratio_to: int | float, p: float | int = 0.5
    ) -> None:
        super().__init__()
        if not (isinstance(ratio_from, (int, float)) and 0 <= ratio_from <= 1):
            raise ValueError("ratio_from must be a int or float between 0 and 1")
        if not (isinstance(ratio_to, (int, float)) and 0 <= ratio_to <= 1):
            raise ValueError("ratio_to must be a int or float between 0 and 1")
        if not (isinstance(p, (int, float)) and 0 <= p <= 1):
            raise ValueError("p must be a int or float between 0 and 1")
        if ratio_from > ratio_to:
            raise ValueError("ratio_from must be less than ratio_to")

        self.ratio_from = ratio_from
        self.ratio_to = ratio_to
        self.p = p

    def forward(self, signal: Tensor) -> Tensor:
        if not isinstance(signal, Tensor):
            raise ValueError("Input signal must be a Tensor")

        if torch.rand(1) < self.p:
            signal_length = signal.size(-1)
            mask_length = int(
                signal_length * np.random.uniform(self.ratio_from, self.ratio_to)
            )
            start_idx = np.random.randint(0, signal_length - mask_length)

            signal = signal.clone()
            signal[start_idx : start_idx + mask_length] = 0
        return signal


class ZScoreClip(nn.Module):
    def __init__(self, factor: int | float = 3) -> None:
        super().__init__()
        if not (isinstance(factor, (int, float)) and factor > 0):
            raise ValueError("factor must be a int or float greater than 0")
        self.factor = factor

    def forward(self, signal: Tensor) -> Tensor:
        if not isinstance(signal, Tensor):
            raise ValueError("Input signal must be a Tensor")
        mean = torch.mean(signal)
        std = torch.std(signal)
        signal = torch.clip(
            signal,
            mean - std * self.factor,
            mean + std * self.factor,
        )
        return signal


class MinMaxNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, signal: Tensor) -> Tensor:
        if not isinstance(signal, Tensor):
            raise ValueError("Input signal must be a Tensor")
        min_val = torch.min(signal)
        max_val = torch.max(signal)

        if min_val == max_val:
            return signal

        signal = (signal - min_val) / (max_val - min_val)
        return signal


class MeanNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, signal: Tensor) -> Tensor:
        if not isinstance(signal, Tensor):
            raise ValueError("Input signal must be a Tensor")
        signal = (signal - torch.mean(signal)) / (torch.max(signal) - torch.min(signal))
        return signal


class ZScoreNorm(nn.Module):
    def __init__(self, mean: int | float, std: int | float) -> None:
        super().__init__()
        if not (isinstance(mean, (int, float))):
            raise ValueError("mean must be a int or float")
        if not (isinstance(std, (int, float))):
            raise ValueError("std must be a int or float")
        self.mean = mean
        self.std = std

    def forward(self, signal: Tensor) -> Tensor:
        if not isinstance(signal, Tensor):
            raise ValueError("Input signal must be a Tensor")
        signal = (signal - self.mean) / self.std
        return signal
