from typing import Dict
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry

SEPARATOR_REGISTRY = Registry("SEPARATOR")
SEPARATOR_REGISTRY.__doc__ = """
Registry for separators, which separate the feature maps from the backbone to a list of feature maps
The list length is equal to the number of images in a stack

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.
"""


def build_separator(cfg, input_shape=None):
    """
    Build an separator defined by `cfg.MODEL.SEPARATOR.NAME`.
    """
    separator_name = cfg.MODEL.SEPARATOR.NAME
    return SEPARATOR_REGISTRY.get(separator_name)(cfg, input_shape)


@SEPARATOR_REGISTRY.register()
class ConvSeparator(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self._stack_size = cfg.INPUT.STACK_SIZE
        self.feature_channels = {k: v.channels for k, v in input_shape.items()}
        self.in_features = list(input_shape.keys())
        self._out_features = self.in_features

        self.sep_convs = nn.ModuleList([
            nn.Conv2d(
                self.feature_channels[f],
                self.feature_channels[f] * self._stack_size,
                kernel_size=1,
                stride=1,
                bias=False
            )
            for f in self.in_features
        ])

    def forward(self, features):
        #Separate by convolution
        z_features = [None] * len(self.in_features)
        for i, f in enumerate(self.in_features):    #inspired by adet condinst mask_branch.py
            z_features[i] = torch.split(self.sep_convs[i](features[f]), self.feature_channels[f], dim = 1)  #Will be a tuple containing stack_size tensors

        #Gather results
        z_results = [[None] * len(self.in_features) for z in range(self._stack_size)]
        for z in range(self._stack_size):
            for i in range(len(self.in_features)):
                z_results[z][i] = z_features[i][z]
            assert len(self._out_features) == len(z_results[z])

        return [dict(zip(self._out_features, z_results[z])) for z in range(self._stack_size)] #inspired by detectron2 fpn.py


@SEPARATOR_REGISTRY.register()
class SharedConvSeparator(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self._stack_size = cfg.INPUT.STACK_SIZE
        self.in_features = list(input_shape.keys())
        self.in_channels = input_shape[self.in_features[0]].channels
        self._out_features = self.in_features

        self.sep_conv = nn.Conv2d(
            self.in_channels,
            self.in_channels * self._stack_size,
            kernel_size=1,
            stride=1,
            bias=False
        )

    def forward(self, features):
        #Separate by convolution
        z_features = [None] * len(self.in_features)
        for i, f in enumerate(self.in_features):    #inspired by adet condinst mask_branch.py
            z_features[i] = torch.split(self.sep_conv(features[f]), self.in_channels, dim = 1)  #Will be a tuple containing stack_size tensors

        #Gather results
        z_results = [[None] * len(self.in_features) for z in range(self._stack_size)]
        for z in range(self._stack_size):
            for i in range(len(self.in_features)):
                z_results[z][i] = z_features[i][z]
            assert len(self._out_features) == len(z_results[z])

        return [dict(zip(self._out_features, z_results[z])) for z in range(self._stack_size)] #inspired by detectron2 fpn.py


@SEPARATOR_REGISTRY.register()
class From3dTo2d(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self._stack_size = cfg.INPUT.STACK_SIZE
        self.in_features = list(input_shape.keys())
        self._out_features = self.in_features

    def forward(self, features):
        z_results = [[None] * len(self.in_features) for z in range(self._stack_size)]
        for i, f in enumerate(self.in_features):
            for z in range(self._stack_size):
                z_results[z][i] = features[f][:, :, z, :, :]

        return [dict(zip(self._out_features, z_results[z])) for z in range(self._stack_size)] #inspired by detectron2 fpn.py
