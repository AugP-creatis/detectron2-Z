# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import fvcore.nn.weight_init as weight_init
from detectron2.modeling.weight_init import init_module
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    Conv3d,
    ConvP3d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

from .backbone import Backbone
from .build import BACKBONE_REGISTRY

from enum import Enum
from collections.abc import Iterable

__all__ = [
    "ResNetBlockBase",
    "BasicBlock",
    "VeryBasicBlock",
    "BottleneckBlock",
    "DeformBottleneckBlock",
    "BasicStem",
    "ResNet",
    "make_stage",
    "build_resnet_backbone",
]

def transform_3d_kernel_to_2d(**kwargs):
    default_args = {"kernel_size": 1, "padding": 0, "dilation": 1}

    kwargs_2d = {}
    for k, v in kwargs.items():
        if isinstance(v, Iterable):
            kwargs_2d[k] = (default_args[k],) + tuple(v[1:])
        else:
            kwargs_2d[k] = (default_args[k], v, v)

    return kwargs_2d

class Conv(Enum):
    CONV_2D = "Conv2d"
    CONV_3D = "Conv3d"
    CONV_P3D = "ConvP3d"

class BasicBlock(CNNBlockBase):
    """
    The basic residual block for ResNet-18 and ResNet-34 defined in :paper:`ResNet`,
    with two 3x3 conv layers and a projection shortcut if needed.
    """

    def __init__(self, conv_type, in_channels, out_channels, kernel_size, *, stride=1, norm="BN", inter_slice=True):
        """
        Args:
            conv_type (Conv): member of the Conv Enum
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first conv.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, stride)

        if conv_type == Conv.CONV_2D:
            ConvSize3 = Conv2d
            ConvSize1 = Conv2d
        elif conv_type == Conv.CONV_3D:
            ConvSize3 = Conv3d
            ConvSize1 = Conv3d
        elif conv_type == Conv.CONV_P3D:
            ConvSize3 = ConvP3d
            ConvSize1 = Conv3d

        if isinstance(kernel_size, Iterable):
            padding = tuple(i//2 for i in kernel_size)
        else:
            padding = kernel_size//2

        if conv_type in (Conv.CONV_3D, Conv.CONV_P3D):
            if not inter_slice:
                args_2d = transform_3d_kernel_to_2d(
                    kernel_size=kernel_size,
                    padding=padding
                )
                kernel_size = args_2d["kernel_size"]
                padding = args_2d["padding"]

        if in_channels != out_channels:
            self.shortcut = ConvSize1(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None
        
        self.conv1 = ConvSize3(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv2 = ConvSize3(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:  # shortcut can be None
                init_module(layer, weight_init.c2_msra_fill)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class VeryBasicBlock(CNNBlockBase):
    """
    The basic residual block for ResNet-18 and ResNet-34 defined in :paper:`ResNet`,
    with two 3x3 conv layers and a projection shortcut if needed.
    """

    def __init__(self, conv_type, in_channels, out_channels, kernel_size, *, stride=1, norm="BN", inter_slice=True):
        """
        Args:
            conv_type (Conv): member of the Conv Enum
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first conv.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, stride)

        if conv_type == Conv.CONV_2D:
            ConvSize3 = Conv2d
        elif conv_type == Conv.CONV_3D:
            ConvSize3 = Conv3d
        elif conv_type == Conv.CONV_P3D:
            ConvSize3 = ConvP3d

        if isinstance(kernel_size, Iterable):
            padding = tuple(i//2 for i in kernel_size)
        else:
            padding = kernel_size//2

        if conv_type in (Conv.CONV_3D, Conv.CONV_P3D):
            if not inter_slice:
                args_2d = transform_3d_kernel_to_2d(
                    kernel_size=kernel_size,
                    padding=padding
                )
                kernel_size = args_2d["kernel_size"]
                padding = args_2d["padding"]

        self.shortcut = in_channels == out_channels and stride == 1

        self.conv1 = ConvSize3(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        init_module(self.conv1, weight_init.c2_msra_fill)

    def forward(self, x):
        out = self.conv1(x)

        if self.shortcut:
            out += x

        out = F.relu_(out)
        return out


class BottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block used by ResNet-50, 101 and 152
    defined in :paper:`ResNet`.  It contains 3 conv layers with kernels
    1x1, 3x3, 1x1, and a projection shortcut if needed.
    """

    def __init__(
        self,
        conv_type,
        in_channels,
        out_channels,
        kernel_size,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
        inter_slice=True
    ):
        """
        Args:
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            num_groups (int): number of groups for the 3x3 conv layer.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            stride_in_1x1 (bool): when stride>1, whether to put stride in the
                first 1x1 convolution or the bottleneck 3x3 convolution.
            dilation (int): the dilation rate of the 3x3 conv layer.
        """
        super().__init__(in_channels, out_channels, stride)

        if conv_type == Conv.CONV_2D:
            ConvSize3 = Conv2d
            ConvSize1 = Conv2d
        elif conv_type == Conv.CONV_3D:
            ConvSize3 = Conv3d
            ConvSize1 = Conv3d
        elif conv_type == Conv.CONV_P3D:
            ConvSize3 = ConvP3d
            ConvSize1 = Conv3d

        if isinstance(kernel_size, Iterable):
            padding = tuple(i//2 * dilation for i in kernel_size)
        else:
            padding = kernel_size//2 * dilation

        if conv_type in (Conv.CONV_3D, Conv.CONV_P3D):
            if not inter_slice:
                args_2d = transform_3d_kernel_to_2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation
                )
                kernel_size = args_2d["kernel_size"]
                padding = args_2d["padding"]
                dilation = args_2d["dilation"]

        if in_channels != out_channels:
            self.shortcut = ConvSize1(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations have
        # stride in the 3x3 conv
        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = ConvSize1(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv2 = ConvSize3(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=kernel_size,
            stride=stride_3x3,
            padding=padding,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = ConvSize1(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                init_module(layer, weight_init.c2_msra_fill)

        # Zero-initialize the last normalization in each residual branch,
        # so that at the beginning, the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # See Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
        # "For BN layers, the learnable scaling coefficient γ is initialized
        # to be 1, except for each residual block's last BN
        # where γ is initialized to be 0."

        # nn.init.constant_(self.conv3.norm.weight, 0)
        # TODO this somehow hurts performance when training GN models from scratch.
        # Add it as an option when we need to use this code to train a backbone.

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        out = self.conv2(out)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class DeformBottleneckBlock(CNNBlockBase):  #Not modified yet for 3d
    """
    Similar to :class:`BottleneckBlock`, but with :paper:`deformable conv <deformconv>`
    in the 3x3 convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
        deform_modulated=False,
        deform_num_groups=1,
    ):
        super().__init__(in_channels, out_channels, stride)
        self.deform_modulated = deform_modulated

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        if deform_modulated:
            deform_conv_op = ModulatedDeformConv
            # offset channels are 2 or 3 (if with modulated) * kernel_size * kernel_size
            offset_channels = 27
        else:
            deform_conv_op = DeformConv
            offset_channels = 18

        self.conv2_offset = Conv2d(
            bottleneck_channels,
            offset_channels * deform_num_groups,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            dilation=dilation,
        )
        self.conv2 = deform_conv_op(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            deformable_groups=deform_num_groups,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        nn.init.constant_(self.conv2_offset.weight, 0)
        nn.init.constant_(self.conv2_offset.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu_(out)

        if self.deform_modulated:
            offset_mask = self.conv2_offset(out)
            offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
            offset = torch.cat((offset_x, offset_y), dim=1)
            mask = mask.sigmoid()
            out = self.conv2(out, offset, mask)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = F.relu_(out)

        out = self.conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        out += shortcut
        out = F.relu_(out)
        return out


class BasicStem(CNNBlockBase):
    """
    The standard ResNet stem (layers before the first residual block).
    """

    def __init__(self, channel_dims, in_channels=3, out_channels=64, norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, 4)
        self._channel_dims = channel_dims
        self.in_channels = in_channels

        if self._channel_dims == 2:
            self.conv1 = Conv2d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        elif self._channel_dims == 3:
            self.conv1 = Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
                bias=False,
                norm=get_norm(norm, out_channels),
            )
        weight_init.c2_msra_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        if self._channel_dims == 2:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        elif self._channel_dims == 3:
            x = F.max_pool3d(x, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        return x


class TwoConvStem(CNNBlockBase):
    """
    The standard ResNet stem (layers before the first residual block).
    """

    def __init__(self, channel_dims, in_channels=3, out_channels=64, norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, 4)
        self._channel_dims = channel_dims
        self.in_channels = in_channels

        if self._channel_dims == 2:
            self.conv1 = Conv2d(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
                norm=get_norm(norm, out_channels),
            )
            self.conv2 = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm=get_norm(norm, out_channels),
            )

        elif self._channel_dims == 3:
            self.conv1 = Conv3d(
                in_channels,
                out_channels,
                kernel_size=(1, 5, 5),
                stride=(1, 2, 2),
                padding=(0, 2, 2),
                bias=False,
                norm=get_norm(norm, out_channels),
            )
            self.conv2 = Conv3d(
                out_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=(0, 1, 1),
                bias=False,
                norm=get_norm(norm, out_channels),
            )

        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = self.conv2(x)
        x = F.relu_(x)

        if self._channel_dims == 2:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        elif self._channel_dims == 3:
            x = F.max_pool3d(x, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        return x
    

class ResNet(Backbone):
    """
    Implement :paper:`ResNet`.
    """

    def __init__(self, channel_dims, stem, stages, num_classes=None, out_features=None):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[CNNBlockBase]]): several (typically 4) stages,
                each contains multiple :class:`CNNBlockBase`.
            num_classes (None or int): if None, will not perform classification.
                Otherwise, will create a linear layer.
            out_features (list[str]): name of the layers whose outputs should
                be returned in forward. Can be anything in "stem", "linear", or "res2" ...
                If None, will return the output of the last layer.
        """
        super().__init__()
        self._channel_dims = channel_dims
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": self.stem.out_channels}

        self.stages_and_names = []
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block

            name = "res" + str(i + 2)
            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))

            for k in blocks:
                if isinstance(k.stride, int):
                    current_stride *= k.stride
                else:
                    current_stride *= k.stride[-1]

            self._out_feature_strides[name] = current_stride = int(current_stride)
            self._out_feature_channels[name] = curr_channels = blocks[-1].out_channels

        if num_classes is not None:
            if self._channel_dims == 2:
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            elif self._channel_dims == 3:
                self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.linear = nn.Linear(curr_channels, num_classes)

            # Sec 5.1 in "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour":
            # "The 1000-way fully-connected layer is initialized by
            # drawing weights from a zero-mean Gaussian with standard deviation of 0.01."
            nn.init.normal_(self.linear.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(", ".join(children))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W) or (N,C,D,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]
        """
        if self._channel_dims == 2:
            assert x.dim() == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        elif self._channel_dims == 3:
            assert x.dim() == 5, f"ResNet takes an input of shape (N, C, D, H, W). Got {x.shape} instead!"

        outputs = {}
        x = self.stem(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet. Commonly used in
        fine-tuning.

        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            freeze_at (int): number of stages to freeze.
                `1` means freezing the stem. `2` means freezing the stem and
                one residual stage, etc.

        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, (stage, _) in enumerate(self.stages_and_names, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(block_class, num_blocks, conv_type, kernel_size, first_stride, *, in_channels, out_channels, **kwargs):
        """
        Create a list of blocks of the same type that forms one ResNet stage.
        Layers that produce the same feature map spatial size are defined as one
        "stage" by :paper:`FPN`.

        Args:
            block_class (type): a subclass of CNNBlockBase that's used to create all blocks in this
                stage. A module of this type must not change spatial resolution of inputs unless its
                stride != 1.
            num_blocks (int): number of blocks in this stage
            conv_type (Conv): member of the Conv Enum
            first_stride (int): the stride of the first block. The other blocks will have stride=1.
                Therefore this is also the stride of the entire stage.
            in_channels (int): input channels of the entire stage.
            out_channels (int): output channels of **every block** in the stage.
            kwargs: other arguments passed to the constructor of `block_class`.

        Returns:
            list[nn.Module]: a list of block module.
        """
        assert "stride" not in kwargs, "Stride of blocks in make_stage cannot be changed."
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                block_class(
                    conv_type=conv_type,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=first_stride if i == 0 else 1,
                    **kwargs,
                )
            )
            in_channels = out_channels
        return blocks


ResNetBlockBase = CNNBlockBase
"""
Alias for backward compatibiltiy.
"""


def make_stage(*args, **kwargs):
    """
    Deprecated alias for backward compatibiltiy.
    """
    return ResNet.make_stage(*args, **kwargs)


@BACKBONE_REGISTRY.register()
def build_resnet_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    channel_dims = cfg.MODEL.BACKBONE.DIM
    if channel_dims == 2 :
        conv_type_per_stage = [Conv.CONV_2D] * 4
        kernel_size_per_stage = [3] * 4
        downsampling = (2, 2)
    elif channel_dims == 3 :
        conv_type_per_stage = [Conv.CONV_3D, Conv.CONV_P3D, Conv.CONV_P3D, Conv.CONV_P3D]
        kernel_size_per_stage = [(1, 3, 3), 3, 3, 3]
        downsampling = (1, 2, 2)

    # need registration of new blocks/stems?
    depth = cfg.MODEL.RESNETS.DEPTH
    norm = cfg.MODEL.RESNETS.NORM

    if depth == 11:
        stem = TwoConvStem(
            channel_dims=channel_dims,
            in_channels=input_shape.channels,
            out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
            norm=norm,
        )
    else:
        stem = BasicStem(
            channel_dims=channel_dims,
            in_channels=input_shape.channels,
            out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
            norm=norm,
        )

    # fmt: off
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    inter_slice         = cfg.MODEL.BACKBONE.INTER_SLICE
    # fmt: on
    assert res5_dilation in {1, 2}, "res5_dilation cannot be {}.".format(res5_dilation)

    num_blocks_per_stage = {
        11: [2, 2, 2, 2],
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    if depth in [18, 34]:
        assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert not any(
            deform_on_per_stage
        ), "MODEL.RESNETS.DEFORM_ON_PER_STAGE unsupported for R18/R34"
        assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "Must set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features]
    max_stage_idx = max(out_stage_idx)
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else downsampling
        stage_kargs = {
            "num_blocks": num_blocks_per_stage[idx],
            "conv_type": conv_type_per_stage[idx],
            "kernel_size": kernel_size_per_stage[idx],
            "first_stride": first_stride,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "norm": norm,
            "inter_slice": inter_slice,
        }
        if depth == 11:
            stage_kargs["block_class"] = VeryBasicBlock
        # Use BasicBlock for R18 and R34.
        elif depth in [18, 34]:
            stage_kargs["block_class"] = BasicBlock
        else:
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            if deform_on_per_stage[idx]:
                stage_kargs["block_class"] = DeformBottleneckBlock
                stage_kargs["deform_modulated"] = deform_modulated
                stage_kargs["deform_num_groups"] = deform_num_groups
            else:
                stage_kargs["block_class"] = BottleneckBlock
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        if not depth == 11:
            out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(channel_dims, stem, stages, out_features=out_features).freeze(freeze_at)
