import os

import torch


def restore(args, model, optimizer=None, istrain=True):
    if os.path.isfile(args.restore_from) and ('.pt' in args.restore_from):
        snapshot = args.restore_from
    else:
        if args.restore_from != '':
            restore_dir = args.restore_from
        else:
            restore_dir = args.snapshot_dir

            filelist = os.listdir(restore_dir)
            filelist = [x for x in filelist
                        if os.path.isfile(os.path.join(restore_dir, x)) and x.endswith('.pt')]
            if len(filelist) > 0:
                # The newer the file, the bigger the modification time (time since epoch) so we do reverse=True
                filelist.sort(key=lambda fn: os.path.getmtime(os.path.join(restore_dir, fn)), reverse=True)
                snapshot = os.path.join(restore_dir, filelist[0])
            else:
                snapshot = ''

    if os.path.isfile(snapshot):
        print("=> loading checkpoint '{}'".format(snapshot))
        checkpoint = torch.load(snapshot)
        try:
            if istrain:
                args.current_epoch = checkpoint['epoch'] + 1
                args.global_counter = checkpoint['global_counter'] + 1
                optimizer.load_state_dict(checkpoint['optimizer'])
            model.attn_mech.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(snapshot, checkpoint['epoch']))
        except KeyError:
            print("KeyError: Corrupt checkpoint file")
    else:
        print("=> no checkpoint found at '{}'".format(snapshot))
