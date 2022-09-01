import json

from torchvision import models
from torchvision.models.feature_extraction import get_graph_node_names

from utilities.composite_models import Generic

mdl = models.vgg16(pretrained=True)


def model(feature_layers):
    vgg_img = Generic(mdl, feature_layers.split(), 'TAME')
    return vgg_img


if __name__ == '__main__':
    train_names, eval_names = get_graph_node_names(mdl)
    print(train_names)
    # layers = "features.16 features.23 features.30"
    # model(layers)
