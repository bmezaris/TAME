# checked, should be working correctly

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import utilities.metrics as metrics
from utilities.composite_models import Generic
from utilities.restore import restore
from utilities.model_prep import model_prep

# Paths
os.chdir('../')
ROOT_DIR = os.getcwd()
print('Project Root Dir:', ROOT_DIR)

# Static paths
snapshot_dir = os.path.join(ROOT_DIR, 'snapshots')
img_dir = os.path.join(ROOT_DIR, 'images')

def get_arguments():
    parser = argparse.ArgumentParser(description='Generate mask for a given image-label pair with a given model')
    parser.add_argument("--snapshot_dir", type=str, default=snapshot_dir)
    parser.add_argument("--restore-dir", type=str, default='')
    parser.add_argument("--model", type=str, default='vgg16')
    parser.add_argument("--version", type=str, default='TAME')
    parser.add_argument("--layers", type=str, default='features.16 features.23 features.30')
    parser.add_argument("--name", type=str, default='162_166.JPEG')
    parser.add_argument("--label", type=int, default=162)
    return parser.parse_args()


def get_model(args):
    args.snapshot_dir = os.path.join(args.snapshot_dir, f'{args.model}_{args.version}', '')
    mdl = model_prep(args.model)
    mdl = Generic(mdl, args.layers.split(), args.version)
    restore(args, mdl, istrain=False)
    mdl.cuda()
    return mdl


def main():
    args = get_arguments()

    input_size = 256
    crop_size = 224

    model = get_model(args)
    model.eval()

    tsfm = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(crop_size),
    ])

    tsfm_val = transforms.Compose([
        tsfm,
        transforms.ToTensor(),
    ])

    heatmap_dir = os.path.join(img_dir, "heatmaps",
                               f'{args.model}_{args.version}', '')
    os.makedirs(heatmap_dir, exist_ok=True)
    img_name = args.name
    label = args.label
    img = Image.open(os.path.join(img_dir, img_name))
    if img.size[0] == img.size[1] == crop_size:
        tsfm = torch.nn.Identity()
    im = tsfm_val(img).unsqueeze(0).cuda()
    img = tsfm(img)
    img_name = os.path.splitext(img_name)
    img.save(os.path.join(heatmap_dir, f"{img_name[0]}_{label}{img_name[1]}"))
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
        cam_map = metrics.drop_Npercent(cam_map, 0)
        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        mask = np.array(cam_map.squeeze().cpu().numpy() * 255, dtype=np.uint8)
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        mask_image = cv2.addWeighted(mask, 0.5, opencvImage, 0.5, 0)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
        mask_image = Image.fromarray(mask_image)
        mask_image.save(os.path.join(heatmap_dir, f"{img_name[0]}_{label}_mask{img_name[1]}"))


if __name__ == '__main__':
    main()


