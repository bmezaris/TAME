import torchvision.models as models


def model_prep(model_name):
    models_dict = {'resnet50': models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
                   'vgg16': models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)}
    return models_dict[model_name]
