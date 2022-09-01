import argparse
import json
import os

import numpy as np
import torch
import torch.optim as optim
import torchvision.models as models
from tqdm.auto import tqdm

from utilities.composite_models import Generic
from utilities.load_data import data_loader

# Paths
from utilities.model_prep import model_prep

os.chdir('../')
ROOT_DIR = os.getcwd()
print('Project Root Dir:', ROOT_DIR)

# Static paths
data_dir = os.path.join(ROOT_DIR, 'snapshots', 'data', 'LR')

# Default parameters
num_workers = 4


def get_arguments():
    parser = argparse.ArgumentParser(description='LR finder')
    parser.add_argument("--img-dir", type=str, help='Directory of training images')
    parser.add_argument('--data-dir', type=str, default=data_dir)
    parser.add_argument("--train-list", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=num_workers)
    parser.add_argument("--model", type=str, default='vgg16')
    parser.add_argument("--layers", type=str, default='features.16 features.23 features.30')
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--version", type=str, default='TAME',
                        choices=['TAME', 'Noskipconnection', 'NoskipNobatchnorm', 'Sigmoidinfeaturebranch'])
    return parser.parse_args()


def get_model(args):
    mdl = model_prep(args.model)
    mdl = Generic(mdl, args.layers.split(), args.version)
    mdl.cuda()
    return mdl


def find_lr(args, init_value=1e-8, final_value=10., beta=0.999):
    net = get_model(args)
    net.requires_grad_(requires_grad=False)
    net.attn_mech.requires_grad_()

    net.train()

    trn_loader = data_loader(args)
    num = len(trn_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value

    weights = [weight for name, weight in net.attn_mech.named_parameters() if 'weight' in name]
    biases = [bias for name, bias in net.attn_mech.named_parameters() if 'bias' in name]
    torch.autograd.set_detect_anomaly(True)

    optimizer = optim.SGD([{'params': weights, 'lr': lr, 'weight_decay': args.wd},
                           {'params': biases, 'lr': lr * 2}],
                          momentum=0.9, nesterov=True)

    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = 2 * lr

    avg_loss = 0.
    best_loss = 0.

    batch_num = 0
    losses = []
    lrs = []

    tq_loader = tqdm(trn_loader,
                     unit='batches',
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [ETA: {remaining}, {rate_fmt}{postfix}]')
    for idx, data in enumerate(tq_loader):
        batch_num += 1
        # As before, get the loss for this mini-batch of inputs/outputs
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()

        logits = net(inputs, labels)
        masks = net.get_a(labels.long())
        loss, a, b, c = net.get_loss(logits, labels, masks)

        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        losses.append(smoothed_loss)
        postfix = {'Current Loss': smoothed_loss, 'Min Loss': best_loss,
                   'Current LR': lr}

        tq_loader.set_postfix(postfix)

        lrs.append(lr)
        # Do the SGD step
        loss.backward()
        optimizer.step()
        # Update the lr for the next step
        lr *= mult

        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = 2 * lr

    return lrs, losses


def main():
    args = get_arguments()
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4))
    args.train_list = os.path.join(ROOT_DIR, 'datalist', 'ILSVRC', args.train_list)
    os.makedirs(data_dir, exist_ok=True)
    lrs, losses = find_lr(args)
    json_dir = os.path.join(args.data_dir, f'{args.model}_{args.version}_{args.batch_size}.json')

    with open(json_dir, mode='x') as loss_samples_file:
        json.dump((lrs, losses), loss_samples_file)


if __name__ == '__main__':
    main()
