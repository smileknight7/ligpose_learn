import os
import sys
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))

import argparse
from keras_progbar import Progbar
import warnings
warnings.filterwarnings('ignore')

import torch
# https://github.com/pytorch/pytorch/issues/11201 # Too many open files error
torch.multiprocessing.set_sharing_strategy('file_system')


from model.layers import LigPoseStruct
from model.loss import StructLoss
from utils.pdbbind_utils import split_pdbbind, ComplexStructDataset, collate_struct
from model.param_setting import get_LigPose_params
from utils.training_utils import *
from utils.common import *



def train(rank, world_size, port, args):
    ####################################################################################################################
    # Set GPU device
    ####################################################################################################################
    set_gpu_device(rank, world_size, port, args)
    torch.cuda.set_device(rank)
    args.rank = rank


    ####################################################################################################################
    # Training objects construction
    ####################################################################################################################
    # model
    if rank == 0 or not args.use_multi_gpu:
        print('Initializing model...')
    my_model = LigPoseStruct(args).to(rank)
    if rank == 0 or not args.use_multi_gpu:
        summarize_model(my_model)
    if args.use_multi_gpu:
        my_model = torch.nn.parallel.DistributedDataParallel(my_model, device_ids=[rank], find_unused_parameters=True)



###########################################
    #加载数据集划分部分
###########################################


    # dataset
    if rank == 0 or not args.use_multi_gpu:
        print('Loading dataset...')
    if (not check_data_split(path=args.data_list_path)) or args.regenerate_data_list:
        if rank == 0 or not args.use_multi_gpu:
            print('Generating data list...')


        train_list, val_list, test_list = split_pdbbind(args.pdbbind_path, args.data_split_rate, core_list_path=args.core_list_path)


        save_data_split(train_list, val_list, test_list, path=args.data_list_path)
    train_list, val_list, test_list = load_data_split(path=args.data_list_path, blind_training=args.blind_training)
    if rank == 0 or not args.use_multi_gpu:


        print(f'train_list: {len(train_list)}, val_list: {len(val_list)}, test_list: {len(test_list)}')
    train_dataset = ComplexStructDataset('train', args, train_list, cache_path=args.cache_path)
    val_dataset = ComplexStructDataset('val', args, val_list, cache_path=args.cache_path)
    train_loader, train_sampler, val_loader, val_sampler = get_dataloader(args, train_dataset, val_dataset,
                                                                          world_size, collate_fn=collate_struct)
    train_loader.dataset.training = True
    val_loader.dataset.training = False

    # # other training objects
    loss_object = StructLoss(args).to(rank)
    if args.use_multi_gpu_for_loss_object:
        loss_object = torch.nn.parallel.DistributedDataParallel(loss_object, device_ids=[rank], find_unused_parameters=False)
    dic_opt = get_LigPose_params(my_model, loss_object, args)
    opt_object = CustomOptimization(args, dic_opt)

    # initialize parameters
    my_model, loss_object, opt_object, dic_traj = init_params(args, my_model, loss_object, opt_object)

    # traj visualization
    if rank == 0 or not args.use_multi_gpu:
        writer = init_tensorboard(args.log_dir, args.log_port, args.restart, dic_traj,
                                  start_new_tensorboard=args.start_new_tensorboard,
                                  drop_head_epoch=args.log_drop_head_epoch,
                                  sele_env=args.log_env)


    ####################################################################################################################
    # Training
    ####################################################################################################################
    if rank == 0 or not args.use_multi_gpu:
        print('Begin training...')
    for epoch in range(args.restart, args.n_epoch):
        # =================== Train ===================
        my_model.train(True)
        loss_object.train(True)
        if args.use_multi_gpu:
            train_sampler.set_epoch(epoch)
        if rank == 0 or not args.use_multi_gpu:
            print(f"Epoch [{epoch + 1}/{args.n_epoch}], lr: {opt_object.optimizers[0].param_groups[0]['lr']:.2e}")
            progBar = Progbar(len(train_loader))
            dic_train = defaultdict(list)
        for i, dic_data in enumerate(train_loader):
            dic_data = dic_data.to(rank)
            if args.use_amp_mix:
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    tup_pred = my_model(dic_data)
                    grad_loss, eval_loss = loss_object(tup_pred, dic_data, epoch=epoch)
            else:
                tup_pred = my_model(dic_data)
                grad_loss, eval_loss = loss_object(tup_pred, dic_data, epoch=epoch)
            opt_object(grad_loss, update_together=args.update_together, epoch=epoch)

            if rank == 0 or not args.use_multi_gpu:
                progBar.update(i + 1, [] if args.mute_progbar
                else [*[(k, np.around(v, 5)) for k, v in eval_loss.items()]])
                for k in eval_loss.keys():
                    dic_train[k].append(eval_loss[k])
        if rank == 0 or not args.use_multi_gpu:
            for k in eval_loss.keys():
                dic_traj['train'][k].append(np.mean(dic_train[k]))

        if args.use_multi_gpu:
            torch.distributed.barrier()

        # =================== Val ===================
        my_model.train(False)
        loss_object.train(False)
        if args.use_multi_gpu:
            val_sampler.set_epoch(epoch)
        with torch.no_grad():
            if rank == 0 or not args.use_multi_gpu:
                progBar = Progbar(len(val_loader))
            dic_val = defaultdict(list)
            for i, dic_data in enumerate(val_loader):
                dic_data = dic_data.to(rank)
                tup_pred = my_model(dic_data)
                grad_loss, eval_loss = loss_object(tup_pred, dic_data, epoch=epoch)
                for k in eval_loss.keys():
                    dic_val[k].append(eval_loss[k])
                if rank == 0 or not args.use_multi_gpu:
                    progBar.update(i + 1, [] if args.mute_progbar
                    else [*[(k, np.around(v, 5)) for k, v in eval_loss.items()]])
        save_val(dic_val, rank)
        if args.use_multi_gpu:
            torch.distributed.barrier()

        opt_object.update_schedulers(epoch=epoch)

        # =================== save and vis ===================
        if rank == 0 or not args.use_multi_gpu:
            dic_val = load_val()
            for k in eval_loss.keys():
                dic_traj['val'][k].append(np.mean(dic_val[k]))

            dic_traj['train']['lr_curve'].append(opt_object.optimizers[0].param_groups[0]['lr'])
            dic_traj['val']['lr_curve'].append(opt_object.optimizers[0].param_groups[0]['lr'])

            if (epoch + 1) % args.weight_save_freq == 0:
                save_params(args, my_model, loss_object, opt_object, dic_traj,
                            f'{args.weight_path}/state_{epoch + 1}.chk')

            update_tensorboard(writer, dic_traj, epoch, drop_head_epoch=args.log_drop_head_epoch)

        torch.cuda.empty_cache()
        if args.use_multi_gpu:
            torch.distributed.barrier()
    if args.use_multi_gpu:
        torch.distributed.destroy_process_group()



if __name__ == '__main__':
    ####################################################################################################################
    # main args
    ####################################################################################################################
    parser = argparse.ArgumentParser()

    # data source
    parser.add_argument('--pdbbind_path', type=str,
                        default='/home/dtj/work_site/test/tmp',
                        help='path to prepared data')
    parser.add_argument('--l_npz_path', type=str,
                        default='./',
                        help='dataset path')
    parser.add_argument('--p_npz_path', type=str,
                        default='./',
                        help='dataset path')
    parser.add_argument('--c_npz_path', type=str,
                        default='./',
                        help='dataset path')
    parser.add_argument('--choose_start_weight', type=str,
                        default=None,
                        help='init weight (restart=0), None for random')
    parser.add_argument('--regenerate_data_list', type=str,
                        default=False, help='regenerate data split. (if no file exists)')
    parser.add_argument('--data_list_path', type=str,
                        default='../eval/pdbbind/core_test',
                        help='path to data list, core_test/core_test_reduce_train/som_pretrain/som_pretrain_reduce_train/zinc250k')
    parser.add_argument('--data_split_rate', type=str,
                        default='0.75-0.05-0.2',
                        help='rate for training, validation and testing. 0.75-0.05-0.2/0.95-0.025-0.025')





#以下是修改内容
    parser.add_argument('--labeled_set_path', type=str,
                        default='../eval/pdbbind/labeled_list.txt', help='labeled_core set of PDBbind')
    parser.add_argument('--unlabeled_set_path', type=str,
                        default='../eval/pdbbind/unlabeled_list.txt', help='unlabeled_general set of PDBbind')






    parser.add_argument('--core_list_path', type=str,
                        default='../eval/pdbbind/core_list.txt', help='core set of PDBbind')






    # parser.add_argument('--blind_training', type=str,
    #                     default=False, help='Blind training')
    # parser.add_argument('--original_path', type=str,
    #                     default='/home/dtj/work_site/prepare_data/complex/v2020-PL', help='PDBbind path')


    # training settings
    parser.add_argument('--restart', type=int, default=0, help='restart step, set to 0 for new run')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.99, help='learning rate decay, omitted in lambda LambdaLR')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay (L2 penalty)')
    parser.add_argument('--n_epoch', type=int, default=10000, help='number of epoch')
    parser.add_argument('--update_together', type=str, default=False, help='if update grad together')
    parser.add_argument('--lr_sche_batch_wise', type=str, default=None,
                    help="lr schedulers update per batch, split by ','  e.g. '0,1'. Warning: cause wrong lr if restart>0")
    parser.add_argument('--persistent_workers', type=str, default=True,
                    help='dataloader persistent_workers, True may cause wrong values for func involved in dataset.epoch')
    parser.add_argument('--coor_scale', type=int, default=10, help='coordinate scaler')
    parser.add_argument('--aff_scale', type=int, default=10, help='affinity scaler')


    # model hyperparameters
    parser.add_argument('--max_len_before_sampling', type=int, default=800, help='max number of atoms in complex')
    parser.add_argument('--max_len_after_sampling', type=int, default=700, help='max number of atoms in complex')
    parser.add_argument('--max_len_after_sampling_for_eval', type=int, default=700, help='max number of atoms in complex')
    parser.add_argument('--max_len_ligand', type=int, default=150, help='max number of ligand atoms')
    parser.add_argument('--n_cycle', type=int, default=4, help='number of cycle')
    parser.add_argument('--protein_input_channel', type=int, default=79, help='input dimension of protein node')
    parser.add_argument('--ligand_input_channel', type=int, default=45, help='input dimension of ligand node')
    parser.add_argument('--edge_input_channel', type=int, default=6, help='input dimension of edge')
    parser.add_argument('--node_hidden', type=int, default=768, help='hidden size of node')
    parser.add_argument('--edge_hidden', type=int, default=384, help='hidden size of edge')
    parser.add_argument('--n_block', type=int, default=6, help='number of block')
    parser.add_argument('--n_head', type=int, default=8, help='number of attention head')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--max_ligand_atom_init_distance', type=float, default=10,
                        help='(A) +- max distance to randomly place ligand')

    # additional training settings
    parser.add_argument('--semi_rate', type=float, default=0.0, help='rate to sample a unlabeled data')
    parser.add_argument('--mask_rate_l', type=float, default=.15, help='BERT-like mask rate')
    parser.add_argument('--mask_rate_p', type=float, default=.15, help='BERT-like mask rate')
    parser.add_argument('--noise_distance', type=float, default=2.,
                        help='(A) moise distance for protein atom E.g. C-C=1.5A')
    parser.add_argument('--sample_pocket_flag', type=str, default=True,
                        help='whether sample different pocket')
    parser.add_argument('--select_center_type', type=str, default='any_atom',
                        help='how to select pocket center, any_atom, geo_center')
    parser.add_argument('--select_pocket_type', type=str, default='CA',
                        help='how to select residue, any_atom, CA')
    parser.add_argument('--max_ligand_atom_pretrain_distance', type=float, default=30,
                        help='(A) +- max distance to randomly place ligand')


    # loss hyperparameters
    parser.add_argument('--gamma_1', type=float, default=1., help='loss weight')
    parser.add_argument('--gamma_2', type=float, default=1, help='loss weight')
    parser.add_argument('--gamma_noise', type=float, default=0.1, help='coor noise loss weight')
    parser.add_argument('--gamma_p_x_1', type=float, default=0.1, help='BERT-like mask loss weight')
    parser.add_argument('--gamma_p_x_2', type=float, default=0.1, help='BERT-like mask loss weight')
    parser.add_argument('--gamma_l_x', type=float, default=0.1, help='BERT-like mask loss weight')
    parser.add_argument('--gamma_edge', type=float, default=0.1, help='BERT-like mask loss weight')


    # tmp settings
    parser.add_argument('--cache_path', type=str,
                        default='./cache', help='path to tmp data')


    # device settings
    parser.add_argument('--use_seed', type=str, default=True, help='use random seed')
    parser.add_argument('--seed', type=int, default=random.randint(0, 100), help='random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers in Dataloader')
    parser.add_argument('--use_multi_gpu', type=str, default=False, help='if use GPUs')
    parser.add_argument('--use_multi_gpu_for_loss_object', type=str, default=False, help='if loss object need use GPUs')
    parser.add_argument('--gpu_list', type=str, default='0', help='available GPU list')
    parser.add_argument('--world_size', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--port', type=str, default=str(random.randint(10000, 20000)), help='parallel port')
    parser.add_argument('--use_amp_mix', type=str, default=False, help='mixed precision')


    # saving settings
    parser.add_argument('--weight_save_freq', type=int, default=5, help='save model every x epoch')
    parser.add_argument('--weight_path', type=str, default='./weights', help='path to save model')
    parser.add_argument('--vis', type=str, default=False, help='save visualization')
    parser.add_argument('--vis_path', type=str, default='./vis', help='visualization path')
    parser.add_argument('--start_new_tensorboard', type=str, default=False, help='start a new tensorboard (kill old)')
    parser.add_argument('--log_dir', type=str, default='./log', help='Tensorboard path (./log or /root/tf-logs/)')
    parser.add_argument('--log_port', type=int, default=1212, help='Tensorboard port (6007)')
    parser.add_argument('--log_drop_head_epoch', type=int, default=0, help='omit first epochs in Tensorboard')
    parser.add_argument('--log_env', type=str, default='py36', help='Tensorboard env name, None for jumping env act')
    parser.add_argument('--mute_progbar', type=str, default=False, help='if mute Progbar')


    args = parser.parse_args()


    ####################################################################################################################
    # init args
    ####################################################################################################################
    args = fix_bool(args)

    args.data_split_rate = split_rate(args.data_split_rate)
    print_args(args)
    if args.use_seed:
        set_all_seed(args.seed)
    delmkdir(args.cache_path)


    ####################################################################################################################
    # train
    ####################################################################################################################
    train_with_args(args, train)
    print('='*20 + ' DONE ' + '='*20)












