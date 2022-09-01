# checked, should be working correctly

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utilities import metrics
from utilities.avg_meter import AverageMeter
from utilities.composite_models import Generic
from utilities.load_data import data_loader
from utilities.model_prep import model_prep
from utilities.restore import restore

# Paths
os.chdir('../')
ROOT_DIR = os.getcwd()
print('Project Root Dir:', ROOT_DIR)

# Static paths
snapshot_dir = os.path.join(ROOT_DIR, 'snapshots')


def get_arguments():
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument("--val-dir", type=str)
    parser.add_argument("--snapshot-dir", type=str, default=snapshot_dir)
    parser.add_argument("--restore-from", type=str, default='')
    parser.add_argument("--test-list", type=str)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model", type=str, default='vgg16')
    parser.add_argument("--version", type=str, default='TAME',
                        choices=['TAME', 'Noskipconnection', 'NoskipNobatchnorm', 'Sigmoidinfeaturebranch'])
    parser.add_argument("--layers", type=str, default='features.16 features.23 features.30')
    parser.add_argument("--global-counter", type=int, default=0)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--end-epoch", type=int, default=8)
    return parser.parse_args()


def get_model(args):
    mdl = model_prep(args.model)
    mdl = Generic(mdl, args.layers.split(), args.version)
    restore(args, mdl, istrain=False)
    mdl.cuda()
    return mdl


def main():
    args = get_arguments()
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4))

    args.test_list = os.path.join(ROOT_DIR, 'datalist', 'ILSVRC', args.test_list)

    val_loader = data_loader(args, train=False)
    data_dir = os.path.join(args.snapshot_dir, 'data', 'results')
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir,
                           f"{args.model}_{args.version}.csv"), 'x') as file:
        file.write("epoch,AD(100%),IC(100%),AD(50%),IC(50%),AD(15%),IC(15%)\n")
        # for each epoch
        args.snapshot_dir = os.path.join(args.snapshot_dir,
                                         f'{args.model}_{args.version}', '')
        for model_path, epoch in [(f, epoch) for f, epoch in
                                  [(os.path.join(args.snapshot_dir, f'epoch_{epoch}.pt'), epoch)
                                   for epoch in range(args.start_epoch, args.end_epoch + 1)] if os.path.isfile(f)]:
            file.write(f"{epoch}")
            args.restore_from = os.path.join(args.snapshot_dir, model_path)
            model = get_model(args)
            model.eval()
            for percent in [0, 0.5, 0.85]:

                top1 = AverageMeter()
                top5 = AverageMeter()
                top1.reset()
                top5.reset()

                top1_ = AverageMeter()
                top5_ = AverageMeter()
                top1_.reset()
                top5_.reset()

                global_counter = 0

                y_mask_image = []
                y_image = []

                # Read Images
                # tqdm is the commandline progress bar
                # for each batch
                with torch.inference_mode():
                    for idx, dat in enumerate(tqdm(val_loader, desc=f'Eval for {int(100 * (1 - percent))}%')):
                        imgs, labels = dat
                        imgs, labels = imgs.cuda(), labels.cuda()

                        global_counter += 1
                        # forward pass

                        logits = model(imgs, labels)
                        logits = F.softmax(logits, dim=1)
                        prec1_1, prec5_1 = metrics.accuracy(logits, labels.long(), topk=(1, 5))
                        class_1 = logits.max(1)[-1]
                        index_gt_y = class_1.long().cpu().numpy()
                        Y_i_c = logits.max(1)[0].item()
                        y_image.append(Y_i_c)

                        top1.update(prec1_1[0], imgs.size()[0])
                        top5.update(prec5_1[0], imgs.size()[0])

                        cam_map = model.get_c(index_gt_y)
                        cam_map = metrics.normalizeMinMax(cam_map)
                        cam_map = F.interpolate(cam_map, size=(224, 224), mode='bilinear', align_corners=False)
                        cam_map = metrics.drop_Npercent(cam_map, percent)
                        image = imgs
                        mask_image = image * cam_map

                        # forward pass
                        logits = model(mask_image, labels)

                        logits = F.softmax(logits, dim=1)
                        prec1_1, prec5_1 = metrics.accuracy(logits, labels.long(), topk=(1, 5))
                        Y_i_c_ = logits[:, index_gt_y][0].item()
                        y_mask_image.append(Y_i_c_)
                        top1_.update(prec1_1[0], imgs.size()[0])
                        top5_.update(prec5_1[0], imgs.size()[0])

                y_image = np.array(y_image)
                y_mask_image = np.array(y_mask_image)

                file.write(f",{metrics.AD(y_image, y_mask_image)},{metrics.IC(y_image, y_mask_image)}")
            file.write("\n")
            print(f"epoch {epoch} evaluation finished")


if __name__ == '__main__':
    main()
