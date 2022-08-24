import argparse
import json
import os
import traceback

import numpy as np
import torch
import torch.optim as optim
import torchvision.models as models
from tqdm.auto import tqdm

from utilities.composite_models import Generic
from utilities.load_data import data_loader

# Paths
os.chdir('../')
ROOT_DIR = os.getcwd()
print('Project Root Dir:', ROOT_DIR)
IMG_DIR = r'/m2/ILSVRC2012_img_train'

# Static paths
train_list = os.path.join(ROOT_DIR, 'datalist', 'ILSVRC', 'VGG16_train.txt')
test_list = os.path.join(ROOT_DIR, 'datalist', 'ILSVRC', 'Evaluation_2000.txt')
data_dir = os.path.join(ROOT_DIR, 'snapshots', 'data', 'LR')

# Default parameters
Batch_size = 64
num_workers = 1
models_dict = {'resnet50': 0,
               'vgg16': 1}


def get_arguments():
    parser = argparse.ArgumentParser(description='Att_CS_sigmoidSaliency')
    parser.add_argument("--root-dir", type=str, default=ROOT_DIR, help='Root dir for the project')
    parser.add_argument("--img-dir", type=str, default=IMG_DIR, help='Directory of training images')
    parser.add_argument('--data-dir', type=str, default=data_dir)
    parser.add_argument("--train-list", type=str, default=train_list)
    parser.add_argument("--batch-size", type=int, default=Batch_size)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=num_workers)
    parser.add_argument("--model", type=str, default='vgg16')
    parser.add_argument("--freeze-bn", type=str, default='False', choices=['True', 'False'])
    parser.add_argument("--layers", type=str, default='features.29')
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--version", type=str, default='V1')
    parser.add_argument("--arch", type=str, default='')
    parser.add_argument("--arrangement", type=str, default='1-1')
    parser.add_argument("--optim", type=str, default="SGD", choices=['SGD', 'AdamW'])
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--mode", type=str, default="tot", choices=['tot', 'all'])
    parser.add_argument("--early-stop", type=str, default='True', choices=['True', 'False'])
    return parser.parse_args()


def get_model(args):
    mdl_num = models_dict[args.model]
    model = None
    if mdl_num == 0:
        model = models.resnet50(pretrained=True)
    elif mdl_num == 1:
        model = models.vgg16(pretrained=True)

    model = Generic(model, args.layers.split(), args.version, args.arrangement)
    model.cuda()
    return model


def find_lr(args, init_value=1e-8, final_value=10., beta=0.999):
    net = get_model(args)
    net.requires_grad_(requires_grad=False)
    net.attn_mech.requires_grad_()
    if args.freeze_bn == 'True':
        net.train()
        net.body.eval()
    else:
        net.train()

    trn_loader = data_loader(args)
    num = len(trn_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value

    weights = [weight for name, weight in net.attn_mech.named_parameters() if 'weight' in name]
    biases = [bias for name, bias in net.attn_mech.named_parameters() if 'bias' in name]
    torch.autograd.set_detect_anomaly(True)

    if args.optim == "SGD":
        optimizer = optim.SGD([{'params': weights, 'lr': lr, 'weight_decay': args.wd},
                               {'params': biases, 'lr': lr * 2}],
                              momentum=0.9, nesterov=True)

        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = 2 * lr
    else:
        optimizer = optim.AdamW(net.attn_mech.parameters(), lr=lr, betas=(0.9, args.b2), weight_decay=args.wd)
        optimizer.param_groups[0]['lr'] = lr
    if args.mode == 'tot':
        avg_loss = 0.
        best_loss = 0.
    else:
        avg_loss = np.array([0., 0., 0., 0.])
        best_loss = np.array([0., 0., 0., 0.])
    batch_num = 0
    losses = []
    lrs = []
    try:
        tq_loader = tqdm(trn_loader,
                         unit='batches',
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [ETA: {remaining}, {rate_fmt}{postfix}]')
        for idx, data in enumerate(tq_loader):
            batch_num += 1
            # As before, get the loss for this mini-batch of inputs/outputs
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            if args.arrangement == 'hyper':
                var = 2 * torch.rand(1, device='cuda') - 1  # we vary between -1 and 1
                ce_coeff = 1.5 + 0.5 * var  # lambda3
                area_coeff = 1.75 + 1.25 * var  # lambda2
                var_coeff = 0.0125 + 0.0075 * var  # lambda1
                coeffs = torch.tensor([ce_coeff, area_coeff, var_coeff], device='cuda')
                logits = net(inputs, labels, coeffs=coeffs)
                masks = net.get_a(labels.long())
                loss, a, b, c = \
                    net.get_loss(logits, labels, masks, coeffs)

            else:
                logits = net(inputs, labels)
                masks = net.get_a(labels.long())
                loss, a, b, c = net.get_loss(logits, labels, masks)

            if args.mode == 'tot':
                # Compute the smoothed loss
                avg_loss = beta * avg_loss + (1-beta) * loss.item()
                smoothed_loss = avg_loss / (1 - beta**batch_num)
                # Stop if the loss is exploding
                if args.early_stop == 'True':
                    if batch_num > 1 and smoothed_loss > 4 * best_loss:
                        return lrs, losses
                    # Record the best loss
                    if smoothed_loss < best_loss or batch_num == 1:
                        best_loss = smoothed_loss
                losses.append(smoothed_loss)
                postfix = {'Current Loss': smoothed_loss, 'Min Loss': best_loss,
                           'Current LR': lr}

            else:
                loss_comps = np.array([loss.item(), a.item(), b.item(), c.item()])
                # Compute the smoothed loss

                avg_loss = beta * avg_loss + (1 - beta) * loss_comps
                smoothed_loss = avg_loss / (1 - beta ** batch_num)
                # Stop if the loss is exploding
                if args.early_stop == 'True':
                    if batch_num > 1 and smoothed_loss[0] > 4 * best_loss[0]:
                        return lrs, losses
                    # Record the best loss
                    if smoothed_loss[0] < best_loss[0] or batch_num == 1:
                        best_loss = smoothed_loss
                # Store the values
                losses.append(list(smoothed_loss))
                postfix = {'Current Loss': smoothed_loss[0], 'Min Loss': best_loss[0],
                           'Current LR': lr}

            tq_loader.set_postfix(postfix)

            lrs.append(lr)
            # Do the SGD step
            loss.backward()
            optimizer.step()
            # Update the lr for the next step
            lr *= mult

            if args.optim == "SGD":
                optimizer.param_groups[0]['lr'] = lr
                optimizer.param_groups[1]['lr'] = 2 * lr
            else:
                optimizer.param_groups[0]['lr'] = lr
    except Exception as ex:
        raise ValueError("oopsie") from ex
    finally:
        return lrs, losses


def main():
    args = get_arguments()
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4))
    lrs, losses = find_lr(args)

    json_dir = os.path.join(args.data_dir, f'{args.model}_{args.version}{args.arch}_{args.batch_size}.json')
    if args.mode == 'all':
        json_dir = os.path.join(args.data_dir, f'{args.model}_{args.version}{args.arch}_all_{args.batch_size}.json')
        tot = [loss[0] for loss in losses]
        a = [loss[1] for loss in losses]
        b = [loss[2] for loss in losses]
        c = [loss[3] for loss in losses]
        losses = [[loss[i] for loss in losses] for i in range(4)]

    with open(json_dir, mode='x') as loss_samples_file:
        json.dump((lrs, losses), loss_samples_file)


if __name__ == '__main__':
    main()
