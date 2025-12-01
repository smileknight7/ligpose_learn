import sys

import torch
from collections import defaultdict


def get_default_params(model):
    return list(filter(lambda p: p.requires_grad, model.parameters()))

def filter_requires_grad(param_list):
    return list(filter(lambda p: p.requires_grad, param_list))

def get_LigPoseStruct_dict(model):                  #从模型中将可训练的参数进行收集，并按照字典的形式进行组织
    dic_param = defaultdict(list)
    for name, param in model.named_parameters():    #这个的来源是神经网络中层的nn.Parameter---->nn.Linear为例，他们都是作为属性挂载在nn.module上的
        dic_param['all_param'].append(param)

    dic_param = {k: filter_requires_grad(v) for k, v in dic_param.items()}  #这里是过滤掉了不需要进行训练的参数，构建成了{all_param： [param1, param2, ...]}的形式的字典
    return dic_param


def get_LigPose_params(model, loss_object, args):  #这个可能是留的用于训练loss_object中的接口
    param = model.module if args.use_multi_gpu else model
    dic_param = get_LigPoseStruct_dict(param)

    param_1 = [
        {'params': dic_param['all_param'], 'lr': args.lr, 'betas': (0.9, 0.999)},
    ]

    # optimizer & scheduler
    optimizer_1 = torch.optim.Adam(param_1, lr=args.lr, weight_decay=args.weight_decay)
    lr_verbose = True if args.local_rank == 0 or not args.use_multi_gpu else False       #这里是学习率调度器，只让主进程打印学习率
    scheduler_1 = torch.optim.lr_scheduler.ExponentialLR(optimizer_1, args.lr_decay, last_epoch=-1, verbose=lr_verbose)#ExponentialLR使用指数衰减来调整学习率
    return {'optimizer': [optimizer_1],
            'scheduler': [scheduler_1],
            }