import argparse
import datetime
import json
import os
import time

import torch
import torch.optim as optim
import torchvision.models as models

from utilities.composite_models import Generic
from utilities.avg_meter import AverageMeter
from utilities import metrics
from utilities.load_data import data_loader
from utilities.restore import restore
from utilities.schedule import schedule

# Paths
os.chdir('../')
ROOT_DIR = os.getcwd()
print('Project Root Dir:', ROOT_DIR)
IMG_DIR = r'/m2/ILSVRC2012_img_train'

# Static paths
train_list = os.path.join(ROOT_DIR, 'datalist', 'ILSVRC', 'VGG16_train.txt')
test_list = os.path.join(ROOT_DIR, 'datalist', 'ILSVRC', 'Evaluation_2000.txt')
snapshot_dir = os.path.join(ROOT_DIR, 'snapshots')

# Default parameters
EPOCH = 8
Batch_size = 64
num_workers = 1

models_dict = {'resnet50': 0,
               'vgg16': 1}

def get_arguments():
    parser = argparse.ArgumentParser(description='Att_CS_sigmoidSaliency')
    parser.add_argument("--root-dir", type=str, default=ROOT_DIR, help='Root dir for the project')
    parser.add_argument("--img-dir", type=str, default=IMG_DIR, help='Directory of training images')
    parser.add_argument("--snapshot-dir", type=str, default=snapshot_dir)
    parser.add_argument("--restore-from", type=str, default='')
    parser.add_argument("--train-list", type=str, default=train_list)
    parser.add_argument("--batch-size", type=int, default=Batch_size)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=num_workers)
    parser.add_argument("--resume", type=str, default='True')
    parser.add_argument("--arch", type=str, default='VGG16_L_CAM_Img')
    parser.add_argument("--model", type=str, default='vgg16')
    parser.add_argument("--layers", type=str, default='features.29')
    parser.add_argument("--version", type=str, default='V1')
    parser.add_argument("--arrangement", type=str, default='1-1')
    parser.add_argument("--base-lr", type=float, default=3e-7)
    parser.add_argument("--max-lr", type=float, default=5e-5)
    parser.add_argument("--gamma", type=float, default=0.75)
    parser.add_argument("--epoch", type=int, default=EPOCH)
    parser.add_argument("--current-epoch", type=int, default=0)
    parser.add_argument("--global-counter", type=int, default=0)
    parser.add_argument("--schedule", type=str, default='step')
    return parser.parse_args()


def save_checkpoint(args, state, filename):
    save_path = os.path.join(args.snapshot_dir, filename)
    torch.save(state, save_path)


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


def train(args):
    if args.current_epoch == args.epoch:
        print('Training Finished')
        return
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses_meanMask = AverageMeter()  # Mask energy loss
    losses_variationMask = AverageMeter()  # Mask variation loss
    losses_ce = AverageMeter()

    model = get_model(args)

    # define optimizer
    # We initially decay the learning rate by one step before the first epoch
    weights = [weight for name, weight in model.attn_mech.named_parameters() if 'weight' in name]
    biases = [bias for name, bias in model.attn_mech.named_parameters() if 'bias' in name]

    # noinspection PyArgumentList
    torch.autograd.set_detect_anomaly(True)
    optimizer = optim.SGD([{'params': weights, 'lr': args.base_lr, 'weight_decay': 0.0005},
                           {'params': biases, 'lr': args.base_lr * 2}],
                          momentum=0.9, nesterov=True)

    args.snapshot_dir = os.path.join(args.snapshot_dir, f'{args.arch}_{args.version}_({args.arrangement})', '')
    os.makedirs(args.snapshot_dir, exist_ok=True)
    if args.resume == 'True':
        restore(args, model, optimizer)

    model.requires_grad_(requires_grad=False)
    model.attn_mech.requires_grad_()

    model.train()

    train_loader = data_loader(args)
    steps_per_epoch = len(train_loader)

    with open(os.path.join(args.snapshot_dir, 'train_record.txt'), 'a') as fw:
        config = json.dumps(vars(args), indent=4, separators=(',', ':'))
        fw.write(config)
        fw.write('\nepoch,loss,losses_meanMask,losses_variationMask,losses_ce,pred@1,pred@5, LR1, LR2\n')

    total_epoch = args.epoch
    global_counter = args.global_counter
    current_epoch = args.current_epoch

    # last epoch sets the correct LR when restarting model
    # First, create loss curve to find correct base_lr and max_lr
    scheduler = schedule(args, optimizer, steps_per_epoch)

    end = time.perf_counter()
    max_iter = total_epoch * len(train_loader)
    print('Max iter:', max_iter)

    # Epoch loop
    while current_epoch < total_epoch:
        losses.reset()
        losses_meanMask.reset()
        losses_variationMask.reset()  # Mask variation loss
        losses_ce.reset()  # (1-mask)*img cross entropy loss
        top1.reset()
        top5.reset()
        batch_time.reset()
        disp_time = time.perf_counter()

        it = 0

        sample_freq = 50
        samples_interval = int(steps_per_epoch / sample_freq)

        # Batch loop
        for idx, dat in enumerate(train_loader):
            it = it + 1
            imgs, labels = dat
            global_counter += 1
            imgs, labels = imgs.cuda(), labels.cuda()

            # forward pass
            logits = model(imgs, labels)
            masks = model.get_a(labels.long())

            # loss_val,loss_ce_val,loss_ce_trans_val,loss_meanMask_val,loss_variationMask_val = model.get_loss(
            # logits, label_var, masks[0])
            loss_val, loss_ce_val, loss_meanMask_val, loss_variationMask_val = model.get_loss(logits, labels, masks)

            # gradients that aren't computed are set to None
            optimizer.zero_grad(set_to_none=True)

            # backwards pass
            loss_val.backward()

            # optimizer step
            optimizer.step()

            if args.schedule != 'step':
                # learning rate reduction step
                scheduler.step()

            logits1 = torch.squeeze(logits)
            prec1_1, prec5_1 = metrics.accuracy(logits1, labels.long(), topk=(1, 5))
            top1.update(prec1_1[0], imgs.size()[0])
            top5.update(prec5_1[0], imgs.size()[0])

            # imgs.size()[0] is simply the batch size
            losses.update(loss_val.item(), imgs.size()[0])
            losses_meanMask.update(loss_meanMask_val.item(), imgs.size()[0])
            losses_variationMask.update(loss_variationMask_val.item(), imgs.size()[0])
            # losses_ce_trans.update(loss_ce_trans_val.data, img.size()[0])
            losses_ce.update(loss_ce_val.item(), imgs.size()[0])
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

            # every disp_interval batches we print this
            if global_counter % samples_interval == 0:
                disp_time = time.perf_counter() - disp_time
                eta_seconds = ((total_epoch - current_epoch - 1) * steps_per_epoch + (
                        steps_per_epoch - idx)) * batch_time.avg
                eta_str = (datetime.timedelta(seconds=int(eta_seconds)))
                eta_seconds_epoch = steps_per_epoch * batch_time.avg
                eta_str_epoch = (datetime.timedelta(seconds=int(eta_seconds_epoch)))
                print(f'Epoch: [{current_epoch}][{global_counter % len(train_loader)}/{len(train_loader)}]\t'
                      f'Batch Time {batch_time.avg:.3f}s\t'
                      f'Disp Time {disp_time:.0f}s\t'
                      f'ETA {eta_str} ({eta_str_epoch})\t'
                      f'Total Loss: {losses.avg:.3f}\t'
                      f'Mean Loss: {losses_meanMask.avg:.3f}\t'
                      f'Var Loss: {losses_variationMask.avg:.4f}\t'
                      f'CE Loss: {losses_ce.avg:.4f}\t'
                      f'Prec@1 {top1.avg:.0f}%\t'
                      f'Prec@5 {top5.avg:.0f}%\t')
                disp_time = time.perf_counter()
                losses.reset()
                losses_meanMask.reset()
                losses_variationMask.reset()
                losses_ce.reset()
                top1.reset()
                top5.reset()

        save_checkpoint(args,
                        {
                            'epoch': current_epoch,
                            'arch': args.arch,
                            'global_counter': global_counter,
                            'state_dict': model.attn_mech.state_dict(),
                            'optimizer': optimizer.state_dict()
                        },
                        filename=f'epoch_{current_epoch}.pt'
                        )

        with open(os.path.join(args.snapshot_dir, 'train_record.txt'), 'a') as fw:
            fw.write(f'{current_epoch}, {losses.avg:.4f}, {losses_meanMask.avg:.4f}, {losses_variationMask.avg:.4f},'
                     f'{losses_ce.avg:.4f}, {top1.avg:.3f}, {top5.avg:.3f},'
                     f'{scheduler.get_last_lr()[0]}, {scheduler.get_last_lr()[1]}\n')

        if args.schedule == 'step':
            scheduler.step()

        current_epoch += 1


def main():
    cmd_args = get_arguments()
    print('Running parameters:\n')
    print(json.dumps(vars(cmd_args), indent=4))
    if not os.path.exists(cmd_args.snapshot_dir):
        os.mkdir(cmd_args.snapshot_dir)
    train(cmd_args)


if __name__ == '__main__':
    main()
