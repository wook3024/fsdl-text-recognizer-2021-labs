from typing import Any, Dict, List, Optional
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


CONV_DIM = 64
FC_DIM = 128
IMAGE_SIZE = 28
LAYERS = [2, 2, 2, 2]
GROUPS = 32


class ConvBlock(nn.Module):
    """
    Simple 3x3 conv with padding size 1 (to leave the input size unchanged), followed by a ReLU.
    """
    expansion: int = 1

    def __init__(self, input_channels: int, output_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None, groups: int = 1, dilation: int = 1) -> None:
        super().__init__()

        self.downsample = downsample
        expansion_channels = input_channels * 4
        self.conv1 = nn.Conv2d(
            input_channels, expansion_channels, kernel_size=3, stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(expansion_channels)
        self.conv2 = nn.Conv2d(
            expansion_channels, output_channels, kernel_size=3, stride=1,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class CNN(nn.Module):
    """Simple CNN for recognizing characters in a square image."""

    def __init__(
        self, data_config: Dict[str, Any], args: argparse.Namespace = None
    ) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        input_dims = data_config["input_dims"]
        num_classes = len(data_config["mapping"])

        inplanes = self.args.get("conv_dim", CONV_DIM)
        # fc_dim = self.args.get("fc_dim", FC_DIM)
        layers = self.args.get("layers", LAYERS)

        self.inplanes = inplanes
        self.conv1 = ConvBlock(input_channels=input_dims[0], output_channels=self.inplanes)
        self.conv2 = ConvBlock(input_channels=self.inplanes, output_channels=self.inplanes)
        self.dropout = nn.Dropout(0.25)
        # self.max_pool = nn.MaxPool2d(2)
        self.strided_conv = nn.Conv2d(
            self.inplanes, self.inplanes, kernel_size=3, stride=2, padding=1, bias=True
        )

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(
            128, layers[1], stride=2
        )
        self.layer3 = self._make_layer(
            256, layers[2], stride=2
        )
        self.layer4 = self._make_layer(
            512, layers[3], stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ConvBlock.expansion, num_classes)
        # # Because our 3x3 convs have padding size 1, they leave the input size unchanged.
        # # The 2x2 max-pool divides the input size by 2. Flattening squares it.
        # conv_output_size = IMAGE_SIZE // 2
        # fc_input_dim = int(conv_output_size * conv_output_size * conv_dim)
        # self.fc1 = nn.Linear(fc_input_dim, fc_dim)
        # self.fc2 = nn.Linear(fc_dim, num_classes)

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * ConvBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes, kernel_size=3, stride=stride, padding=dilation,
                    groups=groups,
                    bias=False,
                    dilation=dilation,
                ),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(
            ConvBlock(
                self.inplanes,
                planes,
                stride,
                downsample,
                groups,
            )
        )
        self.inplanes = planes * ConvBlock.expansion
        for _ in range(1, blocks):
            layers.append(
                ConvBlock(
                    self.inplanes,
                    planes,
                    groups=groups,
                    dilation=dilation,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x
            (B, C, H, W) tensor, where H and W must equal IMAGE_SIZE

        Returns
        -------
        torch.Tensor
            (B, C) tensor
        """
        _B, _C, H, W = x.shape
        assert H == W == IMAGE_SIZE
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.max_pool(x)
        x = self.strided_conv(x)
        x = self.dropout(x)

        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.dropout(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        x = self.fc(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_dim", type=int, default=CONV_DIM)
        parser.add_argument("--fc_dim", type=int, default=FC_DIM)
        # https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
        parser.add_argument("--layers", nargs='+', type=int, default=LAYERS)
        parser.add_argument("--groups", type=int, default=GROUPS)
        return parser
