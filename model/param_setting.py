import sys

import torch
from collections import defaultdict


def get_default_params(model):
    return list(filter(lambda p: p.requires_grad, model.parameters()))

def filter_requires_grad(param_list):
    return list(filter(lambda p: p.requires_grad, param_list))

def get_LigPoseStruct_dict(model):
    dic_param = defaultdict(list)
    for name, param in model.named_parameters():
        dic_param['all_param'].append(param)

    dic_param = {k: filter_requires_grad(v) for k, v in dic_param.items()}
    return dic_param


def get_LigPose_params(model, loss_object, args):
    param = model.module if args.use_multi_gpu else model
    dic_param = get_LigPoseStruct_dict(param)

    param_1 = [
        {'params': dic_param['all_param'], 'lr': args.lr, 'betas': (0.9, 0.999)},
    ]

    # optimizer & scheduler
    optimizer_1 = torch.optim.Adam(param_1, lr=args.lr, weight_decay=args.weight_decay)
    lr_verbose = True if args.rank == 0 or not args.use_multi_gpu else False
    scheduler_1 = torch.optim.lr_scheduler.ExponentialLR(optimizer_1, args.lr_decay, last_epoch=-1, verbose=lr_verbose)
    return {'optimizer': [optimizer_1],
            'scheduler': [scheduler_1],
            }