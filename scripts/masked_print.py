# checked, should be working correctly

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms

from utilities import metrics
from utilities.composite_models import Generic
from utilities.restore import restore

# Paths
os.chdir('../')
ROOT_DIR = os.getcwd()
print('Project Root Dir:', ROOT_DIR)
IMG_DIR = r'/m2/ILSVRC2012_img_val'

# Static paths
train_list = os.path.join(ROOT_DIR, 'datalist', 'ILSVRC', 'VGG16_train.txt')
test_list = os.path.join(ROOT_DIR, 'datalist', 'ILSVRC', 'Evaluation_2000.txt')
Snapshot_dir = os.path.join(ROOT_DIR, 'snapshots', 'VGG16_L_CAM_Img')

# imgs = [('flamingo.png', 129), ('flamingo.png', 130), ('soccer_ball.JPEG', 805),
#         ('soccer_ball.JPEG', 167), ('head_cabbage.JPEG', 936), ('head_cabbage.JPEG', 942)]
# imgs = [('padlock.JPEG', 695), ('aclock.JPEG', 409)]
imgs = [('273VS269.JPEG', 273), ('273VS269.JPEG', 269), ('437VS835.JPEG', 437), ('437VS835.JPEG', 835),
        ('162VS166.JPEG', 162), ('162VS166.JPEG', 166)]


def get_arguments():
    parser = argparse.ArgumentParser(description='ResNet50_aux_ResNet18_init')
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR)
    parser.add_argument("--img_dir", type=str, default=IMG_DIR)
    parser.add_argument("--snapshot_dir", type=str, default=Snapshot_dir)
    parser.add_argument("--restore_from", type=str, default=r'/home/xiaolin/.torch/models/vgg16-397923af.pth')
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--arch", type=str, default='VGG16_L_CAM_Img')
    parser.add_argument("--model", type=str, default='vgg16')
    parser.add_argument("--layers", type=str, default='features.29')
    parser.add_argument("--version", type=str, default='V1')
    parser.add_argument("--arrangement", type=str, default='1-1')
    parser.add_argument("--global_counter", type=int, default=0)
    return parser.parse_args()


def get_model(args):
    models_dict = {'resnet50': 0,
                   'vgg16': 1}
    mdl_num = models_dict[args.model]
    mdl = None
    if mdl_num == 0:
        mdl = models.resnet50(pretrained=True)
    elif mdl_num == 1:
        mdl = models.vgg16(pretrained=True)

    mdl = Generic(mdl, args.layers.split(), args.version, args.arrangement)
    mdl.cuda()
    restore(args, mdl, istrain=False)
    return mdl


def main():
    args = get_arguments()

    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    args.snapshot_dir = os.path.join(args.snapshot_dir, f'{args.model}_{args.version}{args.arch}', '')

    model = get_model(args)
    model.eval()

    tsfm_val = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])

    tsfm = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(crop_size),
    ])
    for percentage in [0]:
        heatmap_dir = os.path.join(args.img_dir, "heatmaps",
                                   f'{args.model}_{args.version}_{args.arch}_{int(100 * (1 - percentage))}', '')
        # ensures that we can't overfill folders
        os.makedirs(heatmap_dir, exist_ok=True)
        for img_dir, img_name, label in [(os.path.join(args.img_dir, img_name), img_name, label)
                                         for img_name, label in imgs]:
            img = Image.open(img_dir)
            im = tsfm_val(img).unsqueeze(0).cuda()
            img = tsfm(img)
            img.save(os.path.join(heatmap_dir, f"{label}_{img_name}"))
            # to take care of png imgs
            im = im[:, 0:3, :, :]

            with torch.inference_mode():
                # forward pass
                logits = model(im)
                logits = F.softmax(logits, dim=1)
                print(f"Img: {img_name}, Max label: {torch.max(logits, dim=1)[1].item()}, Chosen label: {label}")
                cam_map = model.get_c(label)
                cam_map = metrics.normalizeMinMax(cam_map)
                cam_map = F.interpolate(cam_map.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
                cam_map = metrics.drop_Npercent(cam_map, percentage)
                opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                mask = np.array(cam_map.squeeze().cpu().numpy() * 255, dtype=np.uint8)
                mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                mask_image = cv2.addWeighted(mask, 0.5, opencvImage, 0.5, 0)
                mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
                mask_image = Image.fromarray(mask_image)
                mask_image.save(os.path.join(heatmap_dir, f"m_{label}_{img_name}"))


if __name__ == '__main__':
    main()


