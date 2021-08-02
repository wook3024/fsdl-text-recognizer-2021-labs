# from typing import Any, Dict
# import argparse

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# FC1_DIM = 1024
# FC2_DIM = 128


# # class GNNBlock(nn.Module):
# #     '''https://github.com/heartcored98/Standalone-DeepLearning-Chemistry'''
# #     def __init__(self, n_layer, in_dim, hidden_dim, out_dim, device, norm_type='no', sc='sc'):
# #         super(GNNBlock, self).__init__()
# #         self.layers = nn.ModuleList()
# #         self.relu = nn.ReLU()
# #         for i in range(n_layer):
# #             self.layers.append(GNNLayer(in_dim if i==0 else hidden_dim,
# #                                         out_dim if i==n_layer-1 else hidden_dim, device,
# #                                         nn.ReLU() if i!=n_layer-1 else None,
# #                                         norm_type))

# #         if sc=='gsc':
# #             self.sc = GatedSkipConnection(in_dim, out_dim)
# #         elif sc=='sc':
# #             self.sc = SkipConnection(in_dim, out_dim)
# #         elif sc=='no':
# #             self.sc = None
# #         else:
# #             assert False, "Wrong sc type."
        
# #     def forward(self, x, adj):
# #         residual = x
# #         for i, layer in enumerate(self.layers):
# #             x, adj = layer(x, adj)
# #         if self.sc != None:
# #             x = self.sc(residual, x)
# #         out = self.relu(x)
# #         return out

# # model = MLPMixer(
# #     image_size = 256,
# #     channels = 3,
# #     patch_size = 16,
# #     dim = 512,
# #     depth = 12,
# #     num_classes = 1000
# # )


# class MLP(nn.Module):
#     """Simple MLP suitable for recognizing single characters."""

#     def __init__(
#         self,
#         data_config: Dict[str, Any],
#         args: argparse.Namespace = None,
#     ) -> None:
#         super().__init__()
#         self.args = vars(args) if args is not None else {}

#         input_dim = np.prod(data_config["input_dims"])
#         num_classes = len(data_config["mapping"])

#         fc1_dim = self.args.get("fc1", FC1_DIM)
#         fc2_dim = self.args.get("fc2", FC2_DIM)

#         self.dropout = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(input_dim, fc1_dim)
#         self.fc2 = nn.Linear(fc1_dim, fc2_dim)
#         self.fc3 = nn.Linear(fc2_dim, num_classes)

#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

#     @staticmethod
#     def add_to_argparse(parser):
#         parser.add_argument("--fc1", type=int, default=1024)
#         parser.add_argument("--fc2", type=int, default=128)
#         return parser


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
DEPTH = 2
IMAGE_SIZE = 28


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )


class MLP(nn.Module):
    def __init__ (self, data_config: Dict[str, Any], args: argparse.Namespace = None):
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
        assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
        num_patches = (image_size // patch_size) ** 2
        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

        self.re_arrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
        self.linear1 = nn.Linear((patch_size ** 2) * channels, dim)
        self.pre_norm_residuals = nn.Sequential(*[nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
            ) for _ in range(depth)])
        self.layer_norm_ = nn.LayerNorm(dim)
        self.reduce_ = Reduce('b n c -> b c', 'mean')
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
