import torch.optim
from torch.optim import lr_scheduler as lr


# noinspection PyTypeChecker
def schedule(args, optimizer, steps_per_epoch):
    if args.schedule == 'step':
        return lr.StepLR(optimizer, 1, gamma=args.gamma, last_epoch=args.current_epoch - 1)

    elif args.schedule == 'cyclic':
        if args.optim == 'SGD':
            return lr.CyclicLR(optimizer, base_lr=[args.base_lr, 2 * args.base_lr],
                               max_lr=[args.max_lr, 2 * args.max_lr],
                               step_size_up=steps_per_epoch * 2,
                               last_epoch=(args.current_epoch * steps_per_epoch)
                               if args.current_epoch != 0 else -1)
        elif args.optim == 'AdamW':
            return lr.CyclicLR(optimizer, base_lr=args.base_lr,
                               max_lr=args.max_lr,
                               step_size_up=steps_per_epoch * 2,
                               last_epoch=(args.current_epoch * steps_per_epoch)
                               if args.current_epoch != 0 else -1)

    elif args.schedule == 'onecycle':
        if args.optim == 'SGD':
            return lr.OneCycleLR(optimizer, [args.max_lr, 2 * args.max_lr], epochs=args.epoch,
                                 steps_per_epoch=steps_per_epoch,
                                 # this denotes the last iteration, if we are just starting out it should be its default
                                 # value, -1
                                 last_epoch=(args.current_epoch * steps_per_epoch)
                                 if args.current_epoch != 0 else -1)
        elif args.optim == 'AdamW':
            return lr.OneCycleLR(optimizer, args.max_lr, epochs=args.epoch,
                                 steps_per_epoch=steps_per_epoch,
                                 last_epoch=(args.current_epoch * steps_per_epoch)
                                 if args.current_epoch != 0 else -1)

    else:
        raise ValueError(f"Schedule {args.schedule} is not a valid schedule")
    raise ValueError(f"Scheduler {args.schedule} has not been implemented for optimizer {args.optim}")
