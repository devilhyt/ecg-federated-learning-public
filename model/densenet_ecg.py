import math
from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch import Tensor
from torchvision.utils import _log_api_usage_once

__all__ = [
    "DenseNetEcg",
]


class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        memory_efficient: bool,
    ) -> None:
        super().__init__()

        # settings
        self.memory_efficient = memory_efficient

        # bottleneck layer
        self.norm1 = nn.BatchNorm1d(num_input_features)
        self.lrelu1 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv1d(
            num_input_features,
            bn_size * growth_rate,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        # composite function
        self.norm2 = nn.BatchNorm1d(bn_size * growth_rate)
        self.lrelu2 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # dropout layer
        self.drop = nn.Dropout(p=drop_rate)

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        out = self.norm1(concated_features)
        out = self.lrelu1(out)
        out = self.conv1(out)
        return out

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input, use_reentrant=False)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:  # noqa: F811
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        out = self.norm2(bottleneck_output)
        out = self.lrelu2(out)
        out = self.conv2(out)
        new_features = self.drop(out)

        return new_features


class _DenseBlock(nn.ModuleDict):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(num_input_features)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv = nn.Conv1d(
            num_input_features, num_output_features, kernel_size=1, stride=1, bias=False
        )
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)


class DenseNetEcg(nn.Module):
    r"""DenseNet ECG model class.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
        compression_factor (float) - factor to reduce the number of features in each transition block. (`θ` in paper)
    """

    def __init__(
        self,
        growth_rate: int = 16,
        block_config: Tuple[int, ...] = (6, 4, 12, 8, 24, 16),
        num_init_features: int = 32,
        bn_size: int = 4,
        db_drop_rate: float = 0,
        num_classes: int = 3,
        memory_efficient: bool = False,
        compression_factor: float = 0.5,
    ) -> None:

        super().__init__()
        _log_api_usage_once(self)

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv1d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm1d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),
                ]
            ) 
        )  # fmt: skip

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=db_drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            trans = _Transition(
                num_input_features=num_features,
                num_output_features=math.floor(num_features * compression_factor),
            )
            self.features.add_module("transition%d" % (i + 1), trans)
            num_features = math.floor(num_features * compression_factor)

        # Final layers
        self.features.add_module("norm_final", nn.BatchNorm1d(num_features))
        self.features.add_module("lrelu_final", nn.LeakyReLU(inplace=True))

        # fc layer
        self.fc_layer = nn.Sequential(
            OrderedDict(
                [
                    ("avgpool", nn.AdaptiveAvgPool1d(1)),
                    ("flatten", nn.Flatten()), 
                    ("out", nn.Linear(num_features, num_classes)),
                ]
            )
        )

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.features(x)
        out = self.fc_layer(out)
        return out
