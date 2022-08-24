# checked, should be working correctly
import sys
from math import log2
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

sys.path.append('../')


class AttentionV5d1(nn.Module):
    r"""The same as V5 but the first activation function is a sigmoid
        ABLATION STUDY"""

    def __init__(self, ft_size):
        super(AttentionV5d1, self).__init__()
        feat_height = ft_size[0][2] if ft_size[0][2] <= 56 else 56
        self.interpolate = lambda inp: F.interpolate(inp, size=(feat_height, feat_height),
                                                     mode='bilinear', align_corners=False)
        in_channels_list = [o[1] for o in ft_size]
        # noinspection PyTypeChecker
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1,
                                              padding=0, bias=True) for in_channels in in_channels_list])
        self.bn_channels = in_channels_list
        self.bns = nn.ModuleList([nn.BatchNorm2d(channels) for channels in self.bn_channels])
        self.act = nn.Sigmoid()
        # for each extra layer we need 1000 more channels to input to the fuse convolution
        fuse_channels = sum(in_channels_list)
        # noinspection PyTypeChecker
        self.fuser = nn.Conv2d(in_channels=fuse_channels, out_channels=1000, kernel_size=1, padding=0, bias=True)

    def forward(self, features):
        # Fusion Strategy
        feature_maps = features.values()
        # Now all feature map sets are of the same HxW
        # conv
        class_maps = [op(feature) for op, feature in zip(self.convs, feature_maps)]
        # batch norm
        class_maps = [op(feature) for op, feature in zip(self.bns, class_maps)]
        # add (skip connection)
        class_maps = [class_map + feature_map for class_map, feature_map in zip(class_maps, feature_maps)]
        # activation
        class_maps = [self.act(class_map) for class_map in class_maps]
        # upscale
        class_maps = [self.interpolate(feature) for feature in class_maps]
        # concat
        class_maps = torch.cat(class_maps, 1)
        # fuse into 1000 channels
        c = self.fuser(class_maps)  # batch_size x1xWxH
        # activation
        a = torch.sigmoid(c)

        return a, c


class AttentionV3d2dd1(nn.Module):
    r"""Like 3.2, but with batch norm before relu"""

    def __init__(self, ft_size):
        super(AttentionV3d2dd1, self).__init__()
        feat_height = ft_size[0][2] if ft_size[0][2] <= 56 else 56
        self.interpolate = lambda inp: F.interpolate(inp, size=(feat_height, feat_height),
                                                     mode='bilinear', align_corners=False)
        in_channels_list = [o[1] for o in ft_size]
        # noinspection PyTypeChecker
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=1000, kernel_size=1, padding=0,
                                              bias=True)
                                    for in_channels in in_channels_list])
        self.bns = nn.ModuleList([nn.BatchNorm2d(channels) for channels in [1000] * 3])
        # for each extra layer we need 1000 more channels to input to the fuse convolution
        fuse_channels = len(ft_size) * 1000
        # noinspection PyTypeChecker
        self.fuser = nn.Conv2d(in_channels=fuse_channels, out_channels=1000, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU()

    def forward(self, features):
        # Fusion Strategy
        feature_maps = features.values()
        # Now all feature map sets are of the same HxW
        class_maps = [op(feature) for op, feature in zip(self.convs, feature_maps)]
        class_maps = [op(feature) for op, feature in zip(self.bns, class_maps)]
        class_maps = [self.relu(class_map) for class_map in class_maps]
        class_maps = [self.interpolate(feature) for feature in class_maps]
        class_maps = torch.cat(class_maps, 1)
        c = self.fuser(class_maps)  # batch_size x1xWxH
        a = torch.sigmoid(c)

        return a, c


class AttentionV4d1dd4(nn.Module):
    r"""The same as V4.1.3 but inner conv doesn't have bias"""

    def __init__(self, ft_size):
        super(AttentionV4d1dd4, self).__init__()
        feat_heights = [o[2] for o in ft_size]
        self.avgpools = nn.ModuleList([nn.AvgPool2d(2 ** n) if n > 0 else nn.Identity() for n in
                                       [round(log2(height) - log2(feat_heights[-2])) for height in feat_heights]])
        feat_height = ft_size[int((len(ft_size) - 1) / 2)][2]
        self.interpolate = lambda inp: F.interpolate(inp, size=(feat_height, feat_height),
                                                     mode='bilinear', align_corners=False)
        in_channels_list = [o[1] for o in ft_size]
        # noinspection PyTypeChecker
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=1000, kernel_size=1, padding=0,
                                              bias=False)
                                    for in_channels in in_channels_list])
        self.bns = nn.ModuleList([nn.BatchNorm2d(channels) for channels in [1000] * 3])
        # for each extra layer we need 1000 more channels to input to the fuse convolution
        fuse_channels = len(ft_size) * 1000
        # noinspection PyTypeChecker
        self.fuser = nn.Conv2d(in_channels=fuse_channels, out_channels=1000, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU()

    def forward(self, features):
        # Fusion Strategy
        feature_maps = features.values()
        feature_maps = [op(feature) for op, feature in zip(self.avgpools, feature_maps)]
        # Now all feature map sets are of the same HxW
        class_maps = [op(feature) for op, feature in zip(self.convs, feature_maps)]
        class_maps = [op(feature) for op, feature in zip(self.bns, class_maps)]
        class_maps = [self.relu(class_map) for class_map in class_maps]
        class_maps = [self.interpolate(feature) for feature in class_maps]
        class_maps = torch.cat(class_maps, 1)
        c = self.fuser(class_maps)  # batch_size x1xWxH
        a = torch.sigmoid(c)

        return a, c


# a bit worse than V3.1, faster
class AttentionV3d2(nn.Module):
    r"""the same as V3.1 but the first conv layers retain the dimensionality"""

    def __init__(self, ft_size):
        super(AttentionV3d2, self).__init__()
        feat_height = ft_size[0][2] if ft_size[0][2] <= 56 else 56
        self.interpolate = lambda inp: F.interpolate(inp, size=(feat_height, feat_height),
                                                     mode='bilinear', align_corners=False)
        in_channels_list = [o[1] for o in ft_size]
        # noinspection PyTypeChecker
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, padding=0,
                                              bias=True)
                                    for in_channels in in_channels_list])
        # for each extra layer we need 1000 more channels to input to the fuse convolution
        fuse_channels = sum(in_channels_list)
        # noinspection PyTypeChecker
        self.fuser = nn.Conv2d(in_channels=fuse_channels, out_channels=1000, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU()

    def forward(self, features):
        # Fusion Strategy
        feature_maps = features.values()
        # Now all feature map sets are of the same HxW
        class_maps = [op(feature) for op, feature in zip(self.convs, feature_maps)]
        class_maps = [torch.relu(class_map) for class_map in class_maps]
        class_maps = [self.interpolate(feature) for feature in class_maps]
        class_maps = torch.cat(class_maps, 1)
        c = self.fuser(class_maps)  # batch_size x1xWxH
        a = torch.sigmoid(c)

        return a, c


# better than 4.1.3 in 15% mark
class AttentionV5(nn.Module):
    r"""The same as V3 but we use an identity block instead of a pointwise conv, and the channels don't increase"""

    def __init__(self, ft_size):
        super(AttentionV5, self).__init__()
        feat_height = ft_size[0][2] if ft_size[0][2] <= 56 else 56
        self.interpolate = lambda inp: F.interpolate(inp, size=(feat_height, feat_height),
                                                     mode='bilinear', align_corners=False)
        in_channels_list = [o[1] for o in ft_size]
        # noinspection PyTypeChecker
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1,
                                              padding=0, bias=True) for in_channels in in_channels_list])
        self.bn_channels = in_channels_list
        self.bns = nn.ModuleList([nn.BatchNorm2d(channels) for channels in self.bn_channels])
        self.relu = nn.ReLU()
        # for each extra layer we need 1000 more channels to input to the fuse convolution
        fuse_channels = sum(in_channels_list)
        # noinspection PyTypeChecker
        self.fuser = nn.Conv2d(in_channels=fuse_channels, out_channels=1000, kernel_size=1, padding=0, bias=True)

    def forward(self, features):
        # Fusion Strategy
        feature_maps = features.values()
        # Now all feature map sets are of the same HxW
        # conv
        class_maps = [op(feature) for op, feature in zip(self.convs, feature_maps)]
        # batch norm
        class_maps = [op(feature) for op, feature in zip(self.bns, class_maps)]
        # add (skip connection)
        class_maps = [class_map + feature_map for class_map, feature_map in zip(class_maps, feature_maps)]
        # activation
        class_maps = [self.relu(class_map) for class_map in class_maps]
        # upscale
        class_maps = [self.interpolate(feature) for feature in class_maps]
        # concat
        class_maps = torch.cat(class_maps, 1)
        # fuse into 1000 channels
        c = self.fuser(class_maps)  # batch_size x1xWxH
        # activation
        a = torch.sigmoid(c)

        return a, c


# better than 3.1 in 15% mark, worse overall
class AttentionV4d1dd3(nn.Module):
    r"""The same as V4.1 but before each ReLU there is a batch normalization layer"""

    def __init__(self, ft_size):
        super(AttentionV4d1dd3, self).__init__()
        feat_heights = [o[2] for o in ft_size]
        self.avgpools = nn.ModuleList([nn.AvgPool2d(2 ** n) if n > 0 else nn.Identity() for n in
                                       [round(log2(height) - log2(feat_heights[-2])) for height in feat_heights]])
        feat_height = ft_size[int((len(ft_size) - 1) / 2)][2]
        self.interpolate = lambda inp: F.interpolate(inp, size=(feat_height, feat_height),
                                                     mode='bilinear', align_corners=False)
        in_channels_list = [o[1] for o in ft_size]
        # noinspection PyTypeChecker
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=1000, kernel_size=1, padding=0,
                                              bias=True)
                                    for in_channels in in_channels_list])
        self.bn_channels = 3 * [1000]
        self.bns = nn.ModuleList([nn.BatchNorm2d(channels) for channels in self.bn_channels])
        # for each extra layer we need 1000 more channels to input to the fuse convolution
        fuse_channels = len(ft_size) * 1000
        # noinspection PyTypeChecker
        self.fuser = nn.Conv2d(in_channels=fuse_channels, out_channels=1000, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU()

    def forward(self, features):
        # Fusion Strategy
        feature_maps = features.values()
        feature_maps = [op(feature) for op, feature in zip(self.avgpools, feature_maps)]
        # Now all feature map sets are of the same HxW
        class_maps = [op(feature) for op, feature in zip(self.convs, feature_maps)]
        class_maps = [op(feature) for op, feature in zip(self.bns, class_maps)]
        class_maps = [self.relu(class_map) for class_map in class_maps]
        class_maps = [self.interpolate(feature) for feature in class_maps]
        class_maps = torch.cat(class_maps, 1)
        c = self.fuser(class_maps)  # batch_size x1xWxH
        a = torch.sigmoid(c)

        return a, c


# improvement over 4.1
class AttentionV3d1(nn.Module):
    r"""the same as V3 but the inbetween activation layer is ReLU"""

    def __init__(self, ft_size):
        super(AttentionV3d1, self).__init__()
        feat_height = ft_size[0][2] if ft_size[0][2] <= 56 else 56
        self.interpolate = lambda inp: F.interpolate(inp, size=(feat_height, feat_height),
                                                     mode='bilinear', align_corners=False)
        in_channels_list = [o[1] for o in ft_size]
        # noinspection PyTypeChecker
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=1000, kernel_size=1, padding=0,
                                              bias=True)
                                    for in_channels in in_channels_list])
        # for each extra layer we need 1000 more channels to input to the fuse convolution
        fuse_channels = len(ft_size) * 1000
        # noinspection PyTypeChecker
        self.fuser = nn.Conv2d(in_channels=fuse_channels, out_channels=1000, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU()

    def forward(self, features):
        # Fusion Strategy
        feature_maps = features.values()
        # Now all feature map sets are of the same HxW
        class_maps = [op(feature) for op, feature in zip(self.convs, feature_maps)]
        class_maps = [torch.relu(class_map) for class_map in class_maps]
        class_maps = [self.interpolate(feature) for feature in class_maps]
        class_maps = torch.cat(class_maps, 1)
        c = self.fuser(class_maps)  # batch_size x1xWxH
        a = torch.sigmoid(c)

        return a, c


# bad results
class AttentionV4d1dd2(nn.Module):
    r"""The same as V4.1 but the second activation function is also ReLU"""

    def __init__(self, ft_size):
        super(AttentionV4d1dd2, self).__init__()
        feat_heights = [o[2] for o in ft_size]
        self.avgpools = nn.ModuleList([nn.AvgPool2d(2 ** n) if n > 0 else nn.Identity() for n in
                                       [round(log2(height) - log2(feat_heights[-2])) for height in feat_heights]])
        feat_height = ft_size[int((len(ft_size) - 1) / 2)][2]
        self.interpolate = lambda inp: F.interpolate(inp, size=(feat_height, feat_height),
                                                     mode='bilinear', align_corners=False)
        in_channels_list = [o[1] for o in ft_size]
        # noinspection PyTypeChecker
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=1000, kernel_size=1, padding=0,
                                              bias=True)
                                    for in_channels in in_channels_list])
        # for each extra layer we need 1000 more channels to input to the fuse convolution
        fuse_channels = len(ft_size) * 1000
        # noinspection PyTypeChecker
        self.fuser = nn.Conv2d(in_channels=fuse_channels, out_channels=1000, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU()

    def forward(self, features):
        # Fusion Strategy
        feature_maps = features.values()
        feature_maps = [op(feature) for op, feature in zip(self.avgpools, feature_maps)]
        # Now all feature map sets are of the same HxW
        class_maps = [op(feature) for op, feature in zip(self.convs, feature_maps)]
        class_maps = [self.relu(class_map) for class_map in class_maps]
        class_maps = [self.interpolate(feature) for feature in class_maps]
        class_maps = torch.cat(class_maps, 1)
        c = self.fuser(class_maps)  # batch_size x1xWxH
        a = self.relu(c)

        return a, c


# no improvement on 4.1
class AttentionV4d3dd2(nn.Module):
    r"""The same as V4.3 but the in-between output of the relu activation function is also passed through a dropout
    layer."""

    def __init__(self, ft_size):
        super(AttentionV4d3dd2, self).__init__()
        feat_heights = [o[2] for o in ft_size]
        self.avgpools = nn.ModuleList([nn.AvgPool2d(2 ** n) if n > 0 else nn.Identity() for n in
                                       [round(log2(height) - log2(feat_heights[-2])) for height in feat_heights]])
        feat_height = ft_size[int((len(ft_size) - 1) / 2)][2]
        self.interpolate = lambda inp: F.interpolate(inp, size=(feat_height, feat_height),
                                                     mode='bilinear', align_corners=False)
        in_channels_list = [o[1] for o in ft_size]
        # noinspection PyTypeChecker
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=1000, kernel_size=1, padding=0,
                                              bias=True)
                                    for in_channels in in_channels_list])
        # for each extra layer we need 1000 more channels to input to the fuse convolution
        fuse_channels = len(ft_size) * 1000
        # noinspection PyTypeChecker
        self.fuser = nn.Conv2d(in_channels=fuse_channels, out_channels=1000, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, features):
        # Fusion Strategy
        feature_maps = features.values()
        feature_maps = [op(feature) for op, feature in zip(self.avgpools, feature_maps)]
        # Now all feature map sets are of the same HxW
        class_maps = [op(feature) for op, feature in zip(self.convs, feature_maps)]
        class_maps = [self.relu(class_map) for class_map in class_maps]
        class_maps = [self.interpolate(feature) for feature in class_maps]
        class_maps = torch.cat(class_maps, 1)
        class_maps = self.dropout(class_maps)
        c = self.fuser(class_maps)  # batch_size x1xWxH
        a = torch.sigmoid(c)
        a = self.dropout(a)

        return a, c


# no improvement on 4.1
class AttentionV4d3(nn.Module):
    r"""The same as V4.1 but the output masks are passed through a dropout layer."""

    def __init__(self, ft_size):
        super(AttentionV4d3, self).__init__()
        feat_heights = [o[2] for o in ft_size]
        self.avgpools = nn.ModuleList([nn.AvgPool2d(2 ** n) if n > 0 else nn.Identity() for n in
                                       [round(log2(height) - log2(feat_heights[-2])) for height in feat_heights]])
        feat_height = ft_size[int((len(ft_size) - 1) / 2)][2]
        self.interpolate = lambda inp: F.interpolate(inp, size=(feat_height, feat_height),
                                                     mode='bilinear', align_corners=False)
        in_channels_list = [o[1] for o in ft_size]
        # noinspection PyTypeChecker
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=1000, kernel_size=1, padding=0,
                                              bias=True)
                                    for in_channels in in_channels_list])
        # for each extra layer we need 1000 more channels to input to the fuse convolution
        fuse_channels = len(ft_size) * 1000
        # noinspection PyTypeChecker
        self.fuser = nn.Conv2d(in_channels=fuse_channels, out_channels=1000, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d()

    def forward(self, features):
        # Fusion Strategy
        feature_maps = features.values()
        feature_maps = [op(feature) for op, feature in zip(self.avgpools, feature_maps)]
        # Now all feature map sets are of the same HxW
        class_maps = [op(feature) for op, feature in zip(self.convs, feature_maps)]
        class_maps = [self.relu(class_map) for class_map in class_maps]
        class_maps = [self.interpolate(feature) for feature in class_maps]
        class_maps = torch.cat(class_maps, 1)
        c = self.fuser(class_maps)  # batch_size x1xWxH
        a = torch.sigmoid(c)
        a = self.dropout(a)

        return a, c


# bad results
class AttentionV4d2(nn.Module):
    r"""Tested only with three layers. Same as V4.1 but instead of changing dimensions with avg-pool and interpolate,
        we used strided convolution to reduce the dimension of the biggest dimensionality feature and transposed
        convolution to increase the dimension of the smallest dimensionality feature"""

    def __init__(self, ft_size):
        super(AttentionV4d2, self).__init__()
        in_channels_list = [o[1] for o in ft_size]
        # noinspection PyTypeChecker
        self.convs = [nn.Conv2d(in_channels=in_channels_list[0], out_channels=1000, kernel_size=1, padding=0,
                                stride=2,
                                bias=True)]
        self.convs.append(nn.Conv2d(in_channels=in_channels_list[1], out_channels=1000, kernel_size=1, padding=0,
                                    bias=True))
        self.convs.append(nn.ConvTranspose2d(in_channels=in_channels_list[2], out_channels=1000, kernel_size=1,
                                             padding=0,
                                             stride=2,
                                             output_padding=1,
                                             bias=True))
        self.convs = nn.ModuleList(self.convs)

        # for each extra layer we need 1000 more channels to input to the fuse convolution
        fuse_channels = len(ft_size) * 1000
        # noinspection PyTypeChecker
        self.fuser = nn.Conv2d(in_channels=fuse_channels, out_channels=1000, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU()

    def forward(self, features):
        # Fusion Strategy
        feature_maps = features.values()
        # Now all feature map sets are of the same HxW
        class_maps = [op(feature) for op, feature in zip(self.convs, feature_maps)]
        class_maps = [self.relu(class_map) for class_map in class_maps]
        class_maps = torch.cat(class_maps, 1)
        c = self.fuser(class_maps)  # batch_size x1xWxH
        a = torch.sigmoid(c)

        return a, c


# big improvement over 4, better than 3
class AttentionV4d1(nn.Module):
    r"""The same as V4 but in the V2 part, the first activation function is changed from sigmoid to relu"""

    def __init__(self, ft_size):
        super(AttentionV4d1, self).__init__()
        feat_heights = [o[2] for o in ft_size]
        self.avgpools = nn.ModuleList([nn.AvgPool2d(2 ** n) if n > 0 else nn.Identity() for n in
                                       [round(log2(height) - log2(feat_heights[-2])) for height in feat_heights]])
        feat_height = ft_size[int((len(ft_size) - 1) / 2)][2]
        self.interpolate = lambda inp: F.interpolate(inp, size=(feat_height, feat_height),
                                                     mode='bilinear', align_corners=False)
        in_channels_list = [o[1] for o in ft_size]
        # noinspection PyTypeChecker
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=1000, kernel_size=1, padding=0,
                                              bias=True)
                                    for in_channels in in_channels_list])
        # for each extra layer we need 1000 more channels to input to the fuse convolution
        fuse_channels = len(ft_size) * 1000
        # noinspection PyTypeChecker
        self.fuser = nn.Conv2d(in_channels=fuse_channels, out_channels=1000, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU()

    def forward(self, features):
        # Fusion Strategy
        feature_maps = features.values()
        feature_maps = [op(feature) for op, feature in zip(self.avgpools, feature_maps)]
        # Now all feature map sets are of the same HxW
        class_maps = [op(feature) for op, feature in zip(self.convs, feature_maps)]
        class_maps = [self.relu(class_map) for class_map in class_maps]
        class_maps = [self.interpolate(feature) for feature in class_maps]
        class_maps = torch.cat(class_maps, 1)
        c = self.fuser(class_maps)  # batch_size x1xWxH
        a = torch.sigmoid(c)

        return a, c


# not as good as V3, but faster
class AttentionV4(nn.Module):
    r"""Tested only with three layers, The biggest dimensionality feature gets avg-pool-ed down and the smallest gets
        interpolated up to the dimensions of the middle feature. The rest is the same as V2"""

    def __init__(self, ft_size):
        super(AttentionV4, self).__init__()
        feat_heights = [o[2] for o in ft_size]
        self.avgpools = nn.ModuleList([nn.AvgPool2d(2 ** n) if n > 0 else nn.Identity() for n in
                                       [round(log2(height) - log2(feat_heights[-2])) for height in feat_heights]])
        feat_height = ft_size[int((len(ft_size) - 1) / 2)][2]
        self.interpolate = lambda inp: F.interpolate(inp, size=(feat_height, feat_height),
                                                     mode='bilinear', align_corners=False)
        in_channels_list = [o[1] for o in ft_size]
        # noinspection PyTypeChecker
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=1000, kernel_size=1, padding=0,
                                              bias=True)
                                    for in_channels in in_channels_list])
        # for each extra layer we need 1000 more channels to input to the fuse convolution
        fuse_channels = len(ft_size) * 1000
        # noinspection PyTypeChecker
        self.fuser = nn.Conv2d(in_channels=fuse_channels, out_channels=1000, kernel_size=1, padding=0, bias=True)

    def forward(self, features):
        # Fusion Strategy
        feature_maps = features.values()
        feature_maps = [op(feature) for op, feature in zip(self.avgpools, feature_maps)]
        # Now all feature map sets are of the same HxW
        class_maps = [op(feature) for op, feature in zip(self.convs, feature_maps)]
        class_maps = [torch.sigmoid(class_map) for class_map in class_maps]
        class_maps = [self.interpolate(feature) for feature in class_maps]
        class_maps = torch.cat(class_maps, 1)
        c = self.fuser(class_maps)  # batch_size x1xWxH
        a = torch.sigmoid(c)

        return a, c


# big improvement over V2, worse on 15% AD-IC
class AttentionV3(nn.Module):
    r"""Features are brought up to the dimension of the biggest H feature, the rest is the same as V2"""

    def __init__(self, ft_size):
        super(AttentionV3, self).__init__()
        feat_height = ft_size[0][2] if ft_size[0][2] <= 56 else 56
        self.interpolate = lambda inp: F.interpolate(inp, size=(feat_height, feat_height),
                                                     mode='bilinear', align_corners=False)
        in_channels_list = [o[1] for o in ft_size]
        # noinspection PyTypeChecker
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=1000, kernel_size=1, padding=0,
                                              bias=True)
                                    for in_channels in in_channels_list])
        # for each extra layer we need 1000 more channels to input to the fuse convolution
        fuse_channels = len(ft_size) * 1000
        # noinspection PyTypeChecker
        self.fuser = nn.Conv2d(in_channels=fuse_channels, out_channels=1000, kernel_size=1, padding=0, bias=True)

    def forward(self, features):
        # Fusion Strategy
        feature_maps = features.values()
        # Now all feature map sets are of the same HxW
        class_maps = [op(feature) for op, feature in zip(self.convs, feature_maps)]
        class_maps = [torch.sigmoid(class_map) for class_map in class_maps]
        class_maps = [self.interpolate(feature) for feature in class_maps]
        class_maps = torch.cat(class_maps, 1)
        c = self.fuser(class_maps)  # batch_size x1xWxH
        a = torch.sigmoid(c)

        return a, c


# improvement over V1, worse on the 15% mark
class AttentionV2(nn.Module):
    r"""Features are first brought down to the dimension of the smallest H feature, pass through separate conv
         (1000 output channels), get concatenated, and are passed through the same conv layer (1000 output channels)"""

    def __init__(self, ft_size):
        super(AttentionV2, self).__init__()
        feat_heights = [o[2] for o in ft_size]
        self.avgpools = nn.ModuleList([nn.AvgPool2d(2 ** n) if n != 0 else nn.Identity() for n in
                                       [round(log2(height) - log2(feat_heights[-1])) for height in feat_heights]])
        in_channels_list = [o[1] for o in ft_size]
        # noinspection PyTypeChecker
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=1000, kernel_size=1, padding=0,
                                              bias=True)
                                    for in_channels in in_channels_list])
        # for each extra layer we need 1000 more channels to input to the fuse convolution
        fuse_channels = len(ft_size) * 1000
        # noinspection PyTypeChecker
        self.fuser = nn.Conv2d(in_channels=fuse_channels, out_channels=1000, kernel_size=1, padding=0, bias=True)

    def forward(self, features):
        # Fusion Strategy
        feature_maps = [op(feature) for op, feature in zip(self.avgpools, features.values())]
        # Now all feature map sets are of the same HxW
        class_maps = [op(feature) for op, feature in zip(self.convs, feature_maps)]
        class_maps = [torch.sigmoid(class_map) for class_map in class_maps]
        class_maps = torch.cat(class_maps, 1)
        c = self.fuser(class_maps)  # batch_size x1xWxH
        a = torch.sigmoid(c)

        return a, c


# overall improvement over using one layer
class AttentionV1(nn.Module):
    r"""Features are brought down to the same dimension as the smallest H features by using avgpools, get concatenated
     and are then passed through the same conv layer """

    def __init__(self, ft_size):
        super(AttentionV1, self).__init__()
        in_channels = sum(o[1] for o in ft_size)
        # noinspection PyTypeChecker
        self.op = nn.Conv2d(in_channels=in_channels, out_channels=1000, kernel_size=1, padding=0, bias=True)
        feat_heights = [o[2] for o in ft_size]
        self.avgpools = nn.ModuleList([nn.AvgPool2d(2 ** n) if n != 0 else nn.Identity() for n in
                                       [round(log2(height) - log2(feat_heights[-1])) for height in feat_heights]])

    def forward(self, features):
        # Fusion Strategy
        feature_maps = torch.cat([op(feature) for op, feature in zip(self.avgpools, features.values())], 1)
        # Now all feature map sets are contained on a single tensor

        c = self.op(feature_maps)  # batch_size x1xWxH
        a = torch.sigmoid(c)

        return a, c


class AttentionMech(nn.Module):
    r"""The attention mechanism component of Generic"""

    def __init__(self, version, ft_size):
        super(AttentionMech, self).__init__()
        versions = {'V1': AttentionV1,
                    'V2': AttentionV2,
                    'V3': AttentionV3,
                    'V4': AttentionV4,
                    'V4.1': AttentionV4d1,
                    'V4.2': AttentionV4d2,
                    'V4.3': AttentionV4d3,
                    'V4.3.2': AttentionV4d3dd2,
                    'V4.1.2': AttentionV4d1dd2,
                    'V3.1': AttentionV3d1,
                    'V4.1.3': AttentionV4d1dd3,
                    'V5': AttentionV5,
                    'V3.2': AttentionV3d2,
                    'V4.1.4': AttentionV4d1dd4,
                    'V3.2.1': AttentionV3d2dd1,
                    'V5.1': AttentionV5d1}
        self.attn = versions[version](ft_size)
        self.forward = self.attn.forward


class CondBatchNorm2d(nn.Module):
    """Conditional Batch Norm"""
    def __init__(self, num_features, num_conds=3):
        super(CondBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.mlp = nn.Linear(num_conds, num_features * 2)
        self.gamma = None
        self.beta = None

    def condition(self, y):
        self.gamma, self.beta = self.mlp(y).chunk(2, 0)

    def forward(self, x):
        out = self.bn(x)
        out = self.gamma.view(-1, self.num_features, 1, 1) * out + self.beta.view(-1, self.num_features, 1, 1)
        return out


class Generic(nn.Module):
    normalization = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __init__(self, mdl, feature_layers: List[str], attn_version: str, arr_version: str):
        """Args:
                mdl (nn.Module): the model which we would like to use for interpretability
                feature_layers (list): the layers, as printed by get_graph_node_names,
                    which we would like to get feature maps from
        """
        super(Generic, self).__init__()
        # get model feature extractor
        train_names, eval_names = get_graph_node_names(mdl)

        output = (train_names[-1], eval_names[-1])
        if output[0] != output[1]:
            print('WARNING! THIS MODEL HAS DIFFERENT OUTPUTS FOR TRAIN AND EVAL MODE')

        self.output = output[0]

        self.body = create_feature_extractor(
            mdl, return_nodes=(feature_layers + [self.output]))

        # Dry run to get number of channels for the attention mechanism
        inp = torch.randn(2, 3, 224, 224)
        self.body.eval()
        with torch.no_grad():
            out = self.body(inp)
        out.pop(self.output)

        # Required for attention mechanism initialization
        ft_size = [o.shape for o in out.values()]

        self.arr = arr_version
        self.attn_mech = AttentionMech(attn_version, ft_size)
        # Build Attention mechanism
        if self.arr == 'hyper':
            assert hasattr(self.attn_mech.attn, 'bns'),\
                f'Version {attn_version} does not have a BN layer required by hyper mode'
            self.attn_mech.attn.bns = \
                nn.ModuleList([CondBatchNorm2d(channels) for channels in self.attn_mech.attn.bn_channels])

        # Get loss and forward training method
        arrangement = Arrangement(arr_version, self.body, self.output)
        self.train_policy, self.get_loss = (arrangement.train_policy, arrangement.loss)

        self.a = None
        self.c = None

    def forward(self, x, label=None, **kwargs):
        x_norm = Generic.normalization(x)

        features = self.body(x_norm)
        x_norm = features.pop(self.output)

        # features now only has the feature maps since we popped the output in case we are in eval mode

        # Attention mechanism
        if self.arr == 'hyper':
            [cond_bn.condition(kwargs['coeffs']) for cond_bn in self.attn_mech.attn.bns]

        a, c = self.attn_mech(features)
        self.a = a
        self.c = c
        # if in training mode we need to do another forward pass with our masked input as input

        if self.training:
            return self.train_policy(a, label, x)
        else:
            return x_norm

    def get_c(self, label):
        map1 = self.c
        map1 = map1[:, label, :, :]
        return map1

    def get_a(self, label):
        map1 = self.a
        map1 = map1[:, label, :, :]
        return map1


class Arrangement(nn.Module):
    r"""The train_policy and get_loss components of Generic"""

    def __init__(self, version, body, output_name):
        super(Arrangement, self).__init__()
        arrangements = {'1-1': (self.train_policy1, self.loss1),
                        '2-2': (self.train_policy2, self.loss2),
                        '2-3': (self.train_policy3, self.loss3),
                        '4-4': (self.train_policy4, self.loss4),
                        '4-1': (self.train_policy4, self.loss1),
                        '1-5': (self.train_policy1, self.loss5),
                        '1-6': (self.train_policy1, self.loss6),
                        '1-7': (self.train_policy1, self.loss7),
                        '1-8': (self.train_policy1, self.loss8),
                        '1-9': (self.train_policy1, self.loss9),
                        '1-10': (self.train_policy1, self.loss10),
                        'hyper': (self.train_policy1, self.hyper_loss)}

        self.loss_cross_entropy = nn.CrossEntropyLoss()
        self.body = body
        self.output_name = output_name

        self.ce_coeff = 1.5  # lambda3
        self.area_loss_coeff = 2  # lambda2
        self.smoothness_loss_coeff = 0.01  # lambda1
        self.area_loss_power = 0.3  # lambda4

        self.extra_masks = None
        if 'coeffs:' in version:
            # e.g. : version = 'coeffs: X, Y, Z' -> coeffs = [X, Y, Z]
            coeffs = [float(coeff) for coeff in version.split(':')[1].split(',')]
            self.ce_coeff = coeffs[0]
            self.area_loss_coeff = coeffs[1]
            self.smoothness_loss_coeff = coeffs[2]
            self.train_policy, self.loss = arrangements['1-1']
        else:
            self.train_policy, self.loss = arrangements[version]

    def area_loss(self, masks):
        if self.area_loss_power != 1:
            # add e to prevent nan (derivative of sqrt at 0 is inf)
            masks = (masks + 0.0005) ** self.area_loss_power
        return torch.mean(masks)

    @staticmethod
    def smoothness_loss(masks, power=2, border_penalty=0.3):
        B, C, H, W = masks.size()
        x_loss = torch.sum((torch.abs(masks[:, :, 1:, :] - masks[:, :, :-1, :])) ** power)
        y_loss = torch.sum((torch.abs(masks[:, :, :, 1:] - masks[:, :, :, :-1])) ** power)
        if border_penalty > 0:
            border = float(border_penalty) * \
                     torch.sum(masks[:, :, -1, :] ** power +
                               masks[:, :, 0, :] ** power +
                               masks[:, :, :, -1] ** power +
                               masks[:, :, :, 0] ** power)
        else:
            border = 0.
        return (x_loss + y_loss + border) / float(power * B)  # watch out, normalised by the batch size!

    @staticmethod
    def tv_loss(masks):
        x_loss = torch.mean(torch.abs(masks[:, :, 1:, :] - masks[:, :, :-1, :]))
        y_loss = torch.mean(torch.abs(masks[:, :, :, 1:] - masks[:, :, :, :-1]))
        return (x_loss + y_loss) * 0.5

    @staticmethod
    def area_loss2(masks):
        masks = torch.square(masks)
        return torch.mean(masks)

    def loss1(self, logits, labels, masks):
        labels = labels.long()
        variation_loss = self.smoothness_loss_coeff * Arrangement.smoothness_loss(masks)
        area_loss = self.area_loss_coeff * self.area_loss(masks)
        cross_entropy = self.ce_coeff * self.loss_cross_entropy(logits, labels)

        loss = cross_entropy + area_loss + variation_loss

        return [loss, cross_entropy, area_loss, variation_loss]

    def train_policy1(self, masks, labels, inp):
        B, C, H, W = masks.size()
        indexes = labels.expand(H, W, 1, B).permute(*torch.arange(masks.ndim - 1, -1, -1))
        masks = torch.gather(masks, 1, indexes)  # select masks
        masks = F.interpolate(masks, size=(224, 224), mode='bilinear', align_corners=False)
        x_masked = masks * inp
        x_norm = Generic.normalization(x_masked)
        return self.body(x_norm)[self.output_name]

    def loss2(self, logits, labels, masks):
        labels = labels.long()
        variation_loss = self.smoothness_loss_coeff * Arrangement.smoothness_loss(masks)
        area_loss = self.area_loss_coeff * self.area_loss(masks)
        cross_entropies = [self.ce_coeff * self.loss_cross_entropy(logit, labels) for logit in logits]

        cross_entropy = 0.75 / 3 * cross_entropies[0] + 0.75 / 3 * cross_entropies[1] + 1.5 / 3 * cross_entropies[2]
        loss = cross_entropy + area_loss + variation_loss
        return [loss, cross_entropy, area_loss, variation_loss]

    def train_policy2(self, masks, labels, inp):
        B, C, H, W = masks.size()
        indexes = labels.expand(H, W, 1, B).permute(*torch.arange(masks.ndim - 1, -1, -1))
        masks = torch.gather(masks, 1, indexes)  # select masks
        masks = F.interpolate(masks, size=(224, 224), mode='bilinear', align_corners=False)
        B, C, H, W = masks.size()
        percent = lambda pc: masks.flatten(start_dim=1, end_dim=3) \
            .quantile(pc, dim=1) \
            .expand(H, W, 1, B) \
            .permute(*torch.arange(masks.ndim - 1, -1, -1))
        masks2 = masks.masked_fill(masks < percent(0.5), 0)
        masks3 = masks.masked_fill(masks < percent(0.85), 0)
        mask_ls = [masks, masks2, masks3]
        x_masked_ls = [mask * inp for mask in mask_ls]
        x_norm_ls = [Generic.normalization(x_masked) for x_masked in x_masked_ls]
        return [self.body(x_norm)[self.output_name] for x_norm in x_norm_ls]

    def loss3(self, logits, labels, masks):
        masks = [masks] + self.extra_masks
        labels = labels.long()

        variation_loss = sum([self.smoothness_loss_coeff * Arrangement.smoothness_loss(mask) for mask in masks]) / 3
        area_loss = sum([self.area_loss_coeff * self.area_loss(mask) for mask in masks]) / 3
        cross_entropy = sum([self.ce_coeff * self.loss_cross_entropy(logit, labels) for logit in logits]) / 3

        loss = sum(cross_entropy) / 3 + sum(area_loss) / 3 + sum(variation_loss) / 3
        return [loss, cross_entropy, area_loss, variation_loss]

    def train_policy3(self, masks, labels, inp):
        B, C, H, W = masks.size()
        percent = lambda pc: masks.flatten(start_dim=2, end_dim=3) \
            .quantile(pc, dim=2) \
            .expand(H, W, B, C) \
            .permute(2, 3, 0, 1)
        self.extra_masks = [masks.masked_fill(masks < percent(pc), 0) for pc in [0.5, 0.85]]
        return self.train_policy2(masks, labels, inp)

    def loss4(self, logits, labels, masks):
        masks = self.extra_masks
        return self.loss1(logits, labels, masks)

    def train_policy4(self, masks, labels, inp):
        B, C, H, W = masks.size()
        percent = lambda pc: masks.flatten(start_dim=2, end_dim=3) \
            .quantile(pc, dim=2) \
            .expand(H, W, B, C) \
            .permute(2, 3, 0, 1)
        self.extra_masks = masks.masked_fill(masks < percent(0.5), 0)
        return self.train_policy1(self.extra_masks, labels, inp)

    def loss5(self, logits, labels, masks):
        if not hasattr(self, 'L1'):
            self.L1 = 1
            self.L2 = 1
            self.L3 = 1

        labels = labels.long()
        variation_loss = Arrangement.tv_loss(masks)
        area_loss = Arrangement.area_loss2(masks)
        cross_entropy = self.loss_cross_entropy(logits, labels)

        beta = 0.1
        coeffs = nn.functional.softmax(
            torch.tensor([beta * (variation_loss - self.L1),
                          beta * (area_loss - self.L2),
                          beta * (cross_entropy - self.L3)], device='cuda'),
            dim=0)
        self.L1 = variation_loss
        self.L2 = area_loss
        self.L3 = cross_entropy
        loss = coeffs[0] * variation_loss + coeffs[1] * area_loss + coeffs[2] * cross_entropy

        # print('loss',loss)
        # print('ce',cross_entropy)
        # print('area_loss',area_loss)
        # print('var_loss',variation_loss)
        # print(f"c0: {coeffs[0]}, c1: {coeffs[1]}, c2: {coeffs[2]}")

        return [loss, coeffs[2] * cross_entropy, coeffs[1] * area_loss, coeffs[0] * variation_loss]

    def loss6(self, logits, labels, masks):

        labels = labels.long()
        variation_loss = Arrangement.tv_loss(masks)
        area_loss = Arrangement.area_loss2(masks)
        cross_entropy = self.loss_cross_entropy(logits, labels)

        loss = variation_loss + area_loss + cross_entropy

        return [loss, cross_entropy, area_loss, variation_loss]

    def loss7(self, logits, labels, masks):
        if not hasattr(self, 'L1'):
            self.L1 = 1
            self.L2 = 1
            self.L3 = 1

        labels = labels.long()
        variation_loss = Arrangement.smoothness_loss(masks)
        area_loss = self.area_loss(masks)
        cross_entropy = self.loss_cross_entropy(logits, labels)

        beta = 0.1
        with torch.no_grad():
            coeffs = nn.functional.softmax(
                torch.tensor([torch.log(variation_loss) + beta * (variation_loss - self.L1),
                              torch.log(area_loss) + beta * (area_loss - self.L2),
                              torch.log(cross_entropy) + beta * (cross_entropy - self.L3)], device='cuda'),
                dim=0)

        self.L1 = variation_loss
        self.L2 = area_loss
        self.L3 = cross_entropy
        loss = coeffs[0] * variation_loss + coeffs[1] * area_loss + coeffs[2] * cross_entropy

        # print('loss',loss)
        # print('ce',cross_entropy)
        # print('area_loss',area_loss)
        # print('var_loss',variation_loss)
        # print(f"c0: {coeffs[0]}, c1: {coeffs[1]}, c2: {coeffs[2]}")

        return [loss, coeffs[2] * cross_entropy, coeffs[1] * area_loss, coeffs[0] * variation_loss]

    def loss8(self, logits, labels, masks):
        if not hasattr(self, 'L1'):
            self.L1 = 1
            self.L2 = 1
            self.L3 = 1

        labels = labels.long()
        variation_loss = Arrangement.tv_loss(masks)
        area_loss = Arrangement.area_loss2(masks)
        cross_entropy = self.loss_cross_entropy(logits, labels)

        beta = 0.1

        coeffs = nn.functional.softmax(
            torch.tensor([torch.log(variation_loss) + beta * (variation_loss - self.L1),
                          torch.log(area_loss) + beta * (area_loss - self.L2),
                          torch.log(cross_entropy) + beta * (cross_entropy - self.L3)], device='cuda'),
            dim=0)
        self.L1 = variation_loss
        self.L2 = area_loss
        self.L3 = cross_entropy
        loss = coeffs[0] * variation_loss + coeffs[1] * area_loss + coeffs[2] * cross_entropy

        # print('loss',loss)
        # print('ce',cross_entropy)
        # print('area_loss',area_loss)
        # print('var_loss',variation_loss)
        # print(f"c0: {coeffs[0]}, c1: {coeffs[1]}, c2: {coeffs[2]}")

        return [loss, coeffs[2] * cross_entropy, coeffs[1] * area_loss, coeffs[0] * variation_loss]

    def loss9(self, logits, labels, masks):
        if not hasattr(self, 'L1'):
            self.L1 = 1
            self.L2 = 1
            self.L3 = 1

        labels = labels.long()
        variation_loss = Arrangement.smoothness_loss(masks)
        area_loss = self.area_loss(masks)
        cross_entropy = self.loss_cross_entropy(logits, labels)

        beta = 0.1
        coeffs = nn.functional.softmax(
            torch.tensor([beta * (variation_loss - self.L1),
                          beta * (area_loss - self.L2),
                          beta * (cross_entropy - self.L3)], device='cuda'),
            dim=0)
        self.L1 = variation_loss
        self.L2 = area_loss
        self.L3 = cross_entropy
        loss = coeffs[0] * variation_loss + coeffs[1] * area_loss + coeffs[2] * cross_entropy

        # print('loss',loss)
        # print('ce',cross_entropy)
        # print('area_loss',area_loss)
        # print('var_loss',variation_loss)
        # print(f"c0: {coeffs[0]}, c1: {coeffs[1]}, c2: {coeffs[2]}")

        return [loss, coeffs[2] * cross_entropy, coeffs[1] * area_loss, coeffs[0] * variation_loss]

    def loss10(self, logits, labels, masks):

        beta = 1
        # Initialize memory for this loss
        # Two datapoints memory
        if not hasattr(self, 'L1'):
            self.L1 = 1
            self.L2 = 1
            self.L3 = 1
            self.coeffs = torch.tensor([1., 1., 1.], device='cuda')
            self.count = 1

        labels = labels.long()
        variation_loss = Arrangement.smoothness_loss(masks)
        area_loss = self.area_loss(masks)
        cross_entropy = self.loss_cross_entropy(logits, labels)

        if self.count == 0:
            diffs = torch.tensor([variation_loss - self.L1,
                                  area_loss - self.L2,
                                  cross_entropy - self.L3], device='cuda')
            diffs = torch.nn.functional.normalize(diffs, p=1, dim=0)
            self.coeffs = nn.functional.softmax(beta * diffs, dim=0)
            self.count = 1

        elif self.count == 1:
            self.L1 = variation_loss
            self.L2 = area_loss
            self.L3 = cross_entropy
            self.count -= 1

        loss = self.coeffs[0] * variation_loss + self.coeffs[1] * area_loss + self.coeffs[2] * cross_entropy

        # print('loss',loss)
        # print('ce',cross_entropy)
        # print('area_loss',area_loss)
        # print('var_loss',variation_loss)
        # print(f"c0: {coeffs[0]}, c1: {coeffs[1]}, c2: {coeffs[2]}")

        return [loss, self.coeffs[2] * cross_entropy, self.coeffs[1] * area_loss, self.coeffs[0] * variation_loss]

    def hyper_loss(self, logits, labels, masks, coeffs):
        labels = labels.long()
        cross_entropy = coeffs[0] * self.loss_cross_entropy(logits, labels)
        area_loss = coeffs[1] * self.area_loss(masks)
        variation_loss = coeffs[2] * Arrangement.smoothness_loss(masks)

        loss = cross_entropy + area_loss + variation_loss

        return [loss, cross_entropy, area_loss, variation_loss]


