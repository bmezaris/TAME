import json

import torch
from torchvision import models
from torchvision.models.feature_extraction import get_graph_node_names

from utilities.composite_models import Generic

mdl = models.vgg16(pretrained=True)


def model(feature_layers):
    vgg_img = Generic(mdl, feature_layers)
    return vgg_img


if __name__ == '__main__':
    train_names, eval_names = get_graph_node_names(mdl)
    print(train_names)
    with open("../snapshots/data/resnet_layers.json", mode='w') as f:
        json.dump(train_names, f, indent=4)
    # inp = torch.randn(3, 3, 224, 224).cuda()
    # lab = torch.randint(0, 1000, (3,)).cuda()
    # mdl2 = Generic(mdl, ['layer2', 'layer3', 'layer4'], 'V4.1').cuda()
    #
    # out = mdl2.body(inp)
    # out.pop(mdl2.output)
    # # # hout = [op(feature) for op, feature in zip(mdl2.avgpools, out.values())]
    # feat_heights = [o.shape[2] for o in out.values()]
    # print(feat_heights)
    # # # new_model = model(['features.29'])
    # # # print(new_model.in_channels)
