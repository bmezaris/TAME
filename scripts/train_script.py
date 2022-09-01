import argparse
import datetime
import json
import os
import time

import torch
import torch.optim as optim
from tqdm.auto import tqdm

from utilities.model_prep import model_prep
from utilities.composite_models import Generic
from utilities.avg_meter import AverageMeter
import utilities.metrics as metrics
from utilities.load_data import data_loader
from utilities.restore import restore
from utilities.schedule import schedule

# Paths
os.chdir('../')
ROOT_DIR = os.getcwd()
print('Project Root Dir:', ROOT_DIR)


# Static paths
snapshot_dir = os.path.join(ROOT_DIR, 'snapshots')

# Default parameters
EPOCH = 8
num_workers = 4


def get_arguments():
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument("--img-dir", type=str, help='Directory of training images')
    parser.add_argument("--snapshot-dir", type=str, default=snapshot_dir)
    parser.add_argument("--restore-from", type=str, default='')
    parser.add_argument("--train-list", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=num_workers)
    parser.add_argument("--resume", type=str, default='True')
    parser.add_argument("--model", type=str, default='vgg16')
    parser.add_argument("--version", type=str, default='TAME',
                        choices=['TAME', 'Noskipconnection', 'NoskipNobatchnorm', 'Sigmoidinfeaturebranch'])
    parser.add_argument("--layers", type=str, default='features.16 features.23 features.30')
    parser.add_argument("--max-lr", type=float, default=5e-5)
    parser.add_argument("--epoch", type=int, default=EPOCH)
    parser.add_argument("--current-epoch", type=int, default=0)
    parser.add_argument("--global-counter", type=int, default=0)
    parser.add_argument("--wd", type=float, default=5e-4)
    return parser.parse_args()


def save_checkpoint(args, state, filename):
    save_path = os.path.join(args.snapshot_dir, filename)
    torch.save(state, save_path)


def get_model(args):
    mdl = model_prep(args.model)
    mdl = Generic(mdl, args.layers.split(), args.version)
    mdl.cuda()
    return mdl


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
    weights = [weight for name, weight in model.attn_mech.named_parameters() if 'weight' in name]
    biases = [bias for name, bias in model.attn_mech.named_parameters() if 'bias' in name]

    torch.autograd.set_detect_anomaly(True)

    # noinspection PyArgumentList
    optimizer = optim.SGD([{'params': weights, 'lr': 1e-7, 'weight_decay': args.wd},
                           {'params': biases, 'lr': 1e-7 * 2}],
                          momentum=0.9, nesterov=True)
    if args.restore_from != '':
        args.snapshot_dir = args.restore_from
    else:
        args.snapshot_dir = os.path.join(snapshot_dir,
                                         f'{args.model}_{args.version}', '')
    os.makedirs(args.snapshot_dir, exist_ok=True)
    if args.resume == 'True':
        restore(args, model, optimizer)
        if args.current_epoch > args.epoch:
            print('Training Finished')
            return

    # freeze classifier
    model.requires_grad_(requires_grad=False)
    model.attn_mech.requires_grad_()

    model.train()

    train_loader = data_loader(args)
    steps_per_epoch = len(train_loader)

    with open(os.path.join(args.snapshot_dir, 'train_record.json'), 'a') as fw:
        config = json.dumps(vars(args), indent=4, separators=(',', ':'))
        fw.write(config)
        fw.write('\n\n')

    total_epoch = args.epoch
    global_counter = args.global_counter
    current_epoch = args.current_epoch

    # last epoch sets the correct LR when restarting model
    # First, create loss curve to find correct base_lr and max_lr
    scheduler = schedule(args, optimizer, steps_per_epoch)

    end = time.perf_counter()
    max_iter = total_epoch * steps_per_epoch
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
        sample_freq = 100
        samples_interval = int(steps_per_epoch / sample_freq)

        # Batch loop
        tq_loader = tqdm(train_loader,
                         desc=f'Epoch {current_epoch}',
                         unit='batches',
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [Batch ETA: {remaining}, {rate_fmt}{postfix}]',
                         miniters=sample_freq)

        for idx, dat in enumerate(tq_loader):
            imgs, labels = dat
            imgs, labels = imgs.cuda(), labels.cuda()

            # forward pass
            logits = model(imgs, labels)
            masks = model.get_a(labels.long())
            loss_val, loss_ce_val, loss_meanMask_val, loss_variationMask_val = model.get_loss(logits, labels, masks)

            # gradients that aren't computed are set to None
            optimizer.zero_grad(set_to_none=True)

            # backwards pass
            loss_val.backward()

            # optimizer step
            optimizer.step()

            # lr reduction step
            scheduler.step()

            logits1 = torch.squeeze(logits)
            prec1_1, prec5_1 = metrics.accuracy(logits1, labels.long(), topk=(1, 5))
            top1.update(prec1_1[0].item(), imgs.size()[0])
            top5.update(prec5_1[0].item(), imgs.size()[0])

            # imgs.size()[0] is simply the batch size
            losses.update(loss_val.item(), imgs.size()[0])
            losses_meanMask.update(loss_meanMask_val.item(), imgs.size()[0])
            losses_variationMask.update(loss_variationMask_val.item(), imgs.size()[0])
            losses_ce.update(loss_ce_val.item(), imgs.size()[0])
            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

            # every samples_interval batches we update the postfix
            if global_counter % samples_interval == 0:
                eta_seconds = ((total_epoch - current_epoch) * steps_per_epoch + (
                            steps_per_epoch - idx)) * batch_time.avg
                eta_str = (datetime.timedelta(seconds=int(eta_seconds)))
                postfix = {'ETA': eta_str, 'Total Loss': losses.avg, 'CE Loss': losses_ce.avg,
                           'Mean Loss': losses_meanMask.avg, 'Var Loss': losses_variationMask.avg,
                           'Top1 Acc': top1.avg, 'Top5 Acc': top5.avg}
                tq_loader.set_postfix(postfix)

                losses.reset()
                losses_meanMask.reset()
                losses_variationMask.reset()
                losses_ce.reset()
                top1.reset()
                top5.reset()

            global_counter += 1

        current_epoch += 1
        # first epoch: 1, during training it is current_epoch == 0, saved as epoch_1 ...
        # last epoch: 8, during training it is current_epoch ==7, saved as epoch_8
        save_checkpoint(args,
                        {
                            'epoch': current_epoch,
                            'global_counter': global_counter,
                            'state_dict': model.attn_mech.state_dict(),
                            'optimizer': optimizer.state_dict()
                        },
                        filename=f'epoch_{current_epoch}.pt')

def main():
    cmd_args = get_arguments()
    cmd_args.train_list = os.path.join(ROOT_DIR, 'datalist', 'ILSVRC', cmd_args.train_list)
    print('Running parameters:\n')
    print(json.dumps(vars(cmd_args), indent=4))
    if not os.path.exists(cmd_args.snapshot_dir):
        os.mkdir(cmd_args.snapshot_dir)
    train(cmd_args)


if __name__ == '__main__':
    main()
