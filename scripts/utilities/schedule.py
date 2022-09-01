import torch.optim
from torch.optim import lr_scheduler as lr


# noinspection PyTypeChecker
def schedule(args, optimizer, steps_per_epoch):
    return lr.OneCycleLR(optimizer, [args.max_lr, 2 * args.max_lr], epochs=args.epoch,
                         steps_per_epoch=steps_per_epoch,
                         # this denotes the last iteration, if we are just starting out it should be its default
                         # value, -1
                         last_epoch=(args.current_epoch * steps_per_epoch)
                         if args.current_epoch != 0 else -1)
