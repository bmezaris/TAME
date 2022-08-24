import argparse
import datetime
import json
import os
import time

import torch

import torch.optim as optim
from utilities import AverageMeter
from utilities import metrics
from utilities.load_data import data_loader
from utilities.restore import restore
import torchvision.models as models
from utilities.composite_models import Generic

# Paths
os.chdir('../')
ROOT_DIR = os.getcwd()
print('Project Root Dir:', ROOT_DIR)
IMG_DIR = r'/m2/ILSVRC2012_img_train'

# Static paths
train_list = os.path.join(ROOT_DIR, 'datalist', 'ILSVRC', 'VGG16_train.txt')
test_list = os.path.join(ROOT_DIR, 'datalist', 'ILSVRC', 'Evaluation_2000.txt')
Snapshot_dir = os.path.join(ROOT_DIR, 'snapshots', 'VGG16_L_CAM_Fm')

# Default parameters
EPOCH = 8
Batch_size = 64
disp_interval = 200
num_workers = 1
num_classes = 1000
dataset = 'imagenet'
LR = 0.0001
models_dict = {'resnet50': 0,
               'vgg16': 1}

scheduler_dict = {'resnet50': list(range(1, 13)),
                  'vgg16': list(range(1, 23))}


def get_arguments():
    parser = argparse.ArgumentParser(description='Att_CS_sigmoidSaliency')
    parser.add_argument("--root-dir", type=str, default=ROOT_DIR, help='Root dir for the project')
    parser.add_argument("--img-dir", type=str, default=IMG_DIR, help='Directory of training images')
    parser.add_argument("--snapshot-dir", type=str, default=Snapshot_dir)
    parser.add_argument("--restore-from", type=str, default='')
    parser.add_argument("--train-list", type=str, default=train_list)
    parser.add_argument("--test-list", type=str, default=test_list)
    parser.add_argument("--batch-size", type=int, default=Batch_size)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--dataset", type=str, default=dataset)
    parser.add_argument("--num-classes", type=int, default=num_classes)
    parser.add_argument("--num-workers", type=int, default=num_workers)
    parser.add_argument("--resume", type=str, default='True')
    parser.add_argument("--arch", type=str, default='VGG16_L_CAM_Img')
    parser.add_argument("--model", type=str, default='vgg16')
    parser.add_argument("--layers", type=str, default='features.29')
    parser.add_argument("--version", type=str, default='V1')
    parser.add_argument("--arrangement", type=str, default='1-1')
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--gamma", type=float, default=0.75)
    parser.add_argument("--epoch", type=int, default=EPOCH)
    parser.add_argument("--current-epoch", type=int, default=0)
    parser.add_argument("--global-counter", type=int, default=0)
    parser.add_argument("--disp-interval", type=int, default=disp_interval)
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
    lr = args.lr
    weights = [weight for name, weight in model.attn_mech.named_parameters() if 'weight' in name]
    biases = [bias for name, bias in model.attn_mech.named_parameters() if 'bias' in name]

    torch.autograd.set_detect_anomaly(True)
    # noinspection PyArgumentList
    optimizer = optim.SGD([{'params': weights, 'lr': lr, 'weight_decay': 0.0005},
                           {'params': biases, 'lr': lr * 2}],
                          momentum=0.9, nesterov=True)

    args.snapshot_dir = os.path.join(args.snapshot_dir, f'{args.arch}_{args.version}_({args.arrangement})', '')
    os.makedirs(args.snapshot_dir, exist_ok=True)
    if args.resume == 'True':
        restore(args, model, optimizer)

    # last epoch sets the correct LR when restarting model
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, scheduler_dict[args.model], gamma=args.gamma,
                                               last_epoch=args.current_epoch - 1)

    model.requires_grad_(requires_grad=False)
    model.attn_mech.requires_grad_()

    model.train()

    train_loader = data_loader(args)

    with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
        config = json.dumps(vars(args), indent=4, separators=(',', ':'))
        fw.write(config)
        fw.write('#epoch,loss,losses_meanMask,losses_variationMask,losses_ce,pred@1,pred@5, LR\n')

    total_epoch = args.epoch
    global_counter = args.global_counter
    current_epoch = args.current_epoch

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

        it = 0
        steps_per_epoch = len(train_loader)

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

            # every 1000 batches
            if global_counter % 2000 == 0:
                losses.reset()
                losses_meanMask.reset()
                losses_variationMask.reset()
                #    losses_ce_trans.reset()
                losses_ce.reset()
                top1.reset()
                top5.reset()

            logits1 = torch.squeeze(logits)
            prec1_1, prec5_1 = metrics.accuracy(logits1, labels.long(), topk=(1, 5))
            top1.update(prec1_1[0], imgs.size()[0])
            top5.update(prec5_1[0], imgs.size()[0])

            # imgs.size()[0] is simply the batch size
            losses.update(loss_val.data, imgs.size()[0])
            losses_meanMask.update(loss_meanMask_val.data, imgs.size()[0])
            losses_variationMask.update(loss_variationMask_val.data, imgs.size()[0])
            # losses_ce_trans.update(loss_ce_trans_val.data, img.size()[0])
            losses_ce.update(loss_ce_val.data, imgs.size()[0])

            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

            # every disp_interval batches we print this
            if global_counter % args.disp_interval == 0:
                eta_seconds = ((total_epoch - current_epoch - 1) * steps_per_epoch + (
                        steps_per_epoch - idx)) * batch_time.avg
                eta_str = (datetime.timedelta(seconds=int(eta_seconds)))
                eta_seconds_epoch = steps_per_epoch * batch_time.avg
                eta_str_epoch = (datetime.timedelta(seconds=int(eta_seconds_epoch)))
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'ETA {eta_str}({eta_str_epoch})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Loss_meanMask {loss_meanMask.val:.4f} ({loss_meanMask.avg:.4f})\t'
                      'Loss_variationMask {loss_variationMask.val:.4f} ({loss_variationMask.avg:.4f})\t'
                      'Loss_ce {loss_ce.val:.4f} ({loss_ce.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    current_epoch, global_counter % len(train_loader), len(train_loader), batch_time=batch_time,
                    eta_str=eta_str, eta_str_epoch=eta_str_epoch, loss=losses, loss_meanMask=losses_meanMask,
                    loss_variationMask=losses_variationMask, loss_ce=losses_ce, top1=top1, top5=top5))

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

        with open(os.path.join(args.snapshot_dir, 'train_record.csv'), 'a') as fw:
            # fw.write('%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f\n'%(current_epoch, losses.avg,losses_meanMask.avg,
            # losses_variationMask.avg,losses_ce.avg,losses_ce_trans.avg, top1.avg, top5.avg))
            fw.write('%d, %.4f, %.4f, %.4f, %.4f, %.3f, %.3f, %f, %f\n' % (
                current_epoch, losses.avg, losses_meanMask.avg, losses_variationMask.avg, losses_ce.avg, top1.avg,
                top5.avg, scheduler.get_last_lr()[0], scheduler.get_last_lr()[1]))

        # learning rate reduction step
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
