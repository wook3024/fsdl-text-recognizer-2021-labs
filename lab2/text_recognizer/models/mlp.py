import argparse
import numpy as np

from typing import Any, Dict
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce


EXPANSION_FACTOR = 4
DROPOUT = 0.5
CHANNELS = 1
PATCH_SIZE = 7
DEPTH = 4
IMAGE_SIZE = 28


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0.0, dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout),
    )


class MLP(nn.Module):
    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None):
        super().__init__()
        args = vars(args) if args is not None else {}

        channels = args.get("channels", CHANNELS)
        patch_size = args.get("patch_size", PATCH_SIZE)
        depth = args.get("depth", DEPTH)
        expansion_factor = args.get("expansion_factor", EXPANSION_FACTOR)
        dropout = args.get("dropout", DROPOUT)
        image_size = args.get("image_size", IMAGE_SIZE)

        dim = np.prod(data_config["input_dims"])
        num_classes = len(data_config["mapping"])
        
        assert (image_size % patch_size) == 0, "image must be divisible by patch size"
        num_patches = (image_size // patch_size) ** 2
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

        self.re_arrange = Rearrange(
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
        )
        self.linear1 = nn.Linear((patch_size ** 2) * channels, dim)
        self.pre_norm_residuals = nn.Sequential(
            *[
                nn.Sequential(
                    PreNormResidual(
                        dim,
                        FeedForward(num_patches, expansion_factor, dropout, chan_first),
                    ),
                    PreNormResidual(
                        dim, FeedForward(dim, expansion_factor, dropout, chan_last)
                    ),
                )
                for _ in range(depth)
            ]
        )
        self.layer_norm_ = nn.LayerNorm(dim)
        self.reduce_ = Reduce("b n c -> b c", "mean")
        self.linear2 = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x = self.mlp_mixer_net(x)
        x = self.re_arrange(x)
        x = self.linear1(x)
        x = self.pre_norm_residuals(x)
        x = self.layer_norm_(x)
        x = self.reduce_(x)
        x = self.linear2(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        ...
        return parser
