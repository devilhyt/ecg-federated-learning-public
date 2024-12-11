import random
import numpy as np
import neurokit2 as nk
import torch.nn as nn
import torch


class RandomTimeScale(nn.Module):
    def __init__(self, factor, p=0.5) -> None:
        super().__init__()
        if not (isinstance(factor, (int, float)) and factor > 0):
            raise ValueError("factor must be a int or float greater than 0")
        if not (isinstance(p, (int, float)) and 0 <= p <= 1):
            raise ValueError("p must be a int or float between 0 and 1")
        self.factor = factor
        self.p = p

    def forward(self, signal):
        if torch.rand(1).item() < self.p:
            # determine the desired length of the new signal
            length = len(signal)
            desired_length = int(
                length * (1 + random.uniform(-self.factor, self.factor))
            )
            # resample the signal
            signal = nk.signal_resample(signal, desired_length)
            # signal length standardization
            if len(signal) > length:
                # random crop
                start = random.randint(0, len(signal) - length)
                signal = signal[start : start + length]
            else:
                # do nothing or repeat the signal to match the original length
                signal = np.resize(signal, length)
        return signal


class RandomNoise(nn.Module):
    def __init__(self, signal_freq, noise_amplitude, noise_freq, p=0.5) -> None:
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
        if torch.rand(1).item() < self.p:
            signal = nk.signal.signal_distort(
                signal,
                sampling_rate=self.signal_freq,
                noise_shape="gaussian",
                noise_amplitude=self.noise_amplitude,
                noise_frequency=self.noise_freq,
            )
        return signal


class RandomInvert(nn.Module):
    def __init__(self, signal_freq, p=0.5) -> None:
        super().__init__()
        if not (isinstance(signal_freq, (int, float)) and signal_freq > 0):
            raise ValueError("signal_freq must be a int or float greater than 0")
        if not (isinstance(p, (int, float)) and 0 <= p <= 1):
            raise ValueError("p must be a int or float between 0 and 1")
        self.signal_freq = signal_freq
        self.p = p

    def forward(self, signal):
        if torch.rand(1).item() < self.p:
            signal, _ = nk.ecg_invert(
                signal, sampling_rate=self.signal_freq, force=True
            )
        return signal


class RandomMask(nn.Module):
    def __init__(self, ratio_from, ratio_to, p=0.5) -> None:
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

    def forward(self, signal):
        if torch.rand(1).item() < self.p:
            length = len(signal)
            mask_length = int(
                length * np.random.uniform(self.ratio_from, self.ratio_to)
            )
            start_idx = np.random.randint(0, length - mask_length)

            signal = np.copy(signal)
            signal[start_idx : start_idx + mask_length] = 0
        return signal


class Clip(nn.Module):
    def __init__(self, mean, std, factor=3) -> None:
        super().__init__()
        if not (isinstance(mean, (int, float))):
            raise ValueError("mean must be a int or float")
        if not (isinstance(std, (int, float))):
            raise ValueError("std must be a int or float")
        if not (isinstance(factor, (int, float)) and factor > 0):
            raise ValueError("factor must be a int or float greater than 0")
        self.mean = mean
        self.std = std
        self.factor = factor

    def forward(self, signal):
        signal = np.clip(
            signal,
            self.mean - self.std * self.factor,
            self.mean + self.std * self.factor,
        )
        return signal
    
class ClipV2(nn.Module):
    def __init__(self, factor=3) -> None:
        super().__init__()
        if not (isinstance(factor, (int, float)) and factor > 0):
            raise ValueError("factor must be a int or float greater than 0")
        self.factor = factor

    def forward(self, signal):
        mean = np.mean(signal)
        std = np.std(signal)
        signal = np.clip(
            signal,
            mean - std * self.factor,
            mean + std * self.factor,
        )
        return signal


class MinMaxNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, signal):
        signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        return signal


class MeanNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, signal):
        signal = (signal - np.mean(signal)) / (np.max(signal) - np.min(signal))
        return signal


class Standardize(nn.Module):
    def __init__(self, mean, std) -> None:
        super().__init__()
        if not (isinstance(mean, (int, float))):
            raise ValueError("mean must be a int or float")
        if not (isinstance(std, (int, float))):
            raise ValueError("std must be a int or float")
        self.mean = mean
        self.std = std

    def forward(self, signal):
        signal = (signal - self.mean) / self.std
        return signal
