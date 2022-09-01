# checked, should be working correctly
import sys
from math import log2
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


class AttentionTAME(nn.Module):
    def __init__(self, ft_size):
        super(AttentionTAME, self).__init__()
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


class AttentionMech(nn.Module):
    r"""The attention mechanism component of Generic"""

    def __init__(self, version, ft_size):
        super(AttentionMech, self).__init__()
        versions = {'TAME': AttentionTAME,
                    'Noskipconnection': AttentionV3d2dd1,
                    'NoskipNobatchnorm': AttentionV3d2,
                    'Sigmoidinfeaturebranch': AttentionV5d1
                    }
        self.attn = versions[version](ft_size)
        self.forward = self.attn.forward


class Generic(nn.Module):
    normalization = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __init__(self, mdl, feature_layers: List[str], attn_version: str):
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

        self.arr = '1-1'
        # Build Attention mechanism
        self.attn_mech = AttentionMech(attn_version, ft_size)

        # Get loss and forward training method
        arrangement = Arrangement('1-1', self.body, self.output)
        self.train_policy, self.get_loss = (arrangement.train_policy, arrangement.loss)

        self.a = None
        self.c = None

    def forward(self, x, label=None):
        x_norm = Generic.normalization(x)

        features = self.body(x_norm)
        x_norm = features.pop(self.output)

        # features now only has the feature maps since we popped the output in case we are in eval mode

        # Attention mechanism

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
        arrangements = {'1-1': (self.train_policy1, self.loss1)}

        self.loss_cross_entropy = nn.CrossEntropyLoss()
        self.body = body
        self.output_name = output_name

        self.ce_coeff = 1.5  # lambda3
        self.area_loss_coeff = 2  # lambda2
        self.smoothness_loss_coeff = 0.01  # lambda1
        self.area_loss_power = 0.3  # lambda4

        self.extra_masks = None
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
