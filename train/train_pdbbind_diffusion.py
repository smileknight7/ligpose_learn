"""
coarse diffusion by motif-based model for ligand pose prediction
"""
import logging
import os
import sys


import argparse
import torch
import warnings

from utils.common import *
from keras_progbar import Progbar
from utils.training_utils import *
from utils.pdbbind_utils import *
from diffusionpart.diffusion import DiffusionLoss
from diffusionpart.egnn import EGNN_dynamics
from model.param_setting import get_LigPose_params
from diffusionpart.data_utils import Fragment, collate_struct




def train_diffusion(rank, world_size, port, args):

# setting gpu device
    set_gpu_device(rank, world_size, port, args)
    torch.cuda.set_device(rank)
    rank = args.local_rank 

    
    if rank == 0 or not args.use_multi_gpu:
        print('Initializing Diffusion model...')
    
# # setting mode for struct or screening

    # if args.model_type == 'struct':
    #     base_model = LigPoseStruct(args).to(rank)
    # elif args.model_type == 'scr':
    #     base_model = LigPoseScr(args).to(rank)                # 上面这里可以写成diffusion接口的形式
    # else:
    #     raise ValueError(f"Unknown model type: {args.model_type}")
    
# instantiation model                                           # 对于lightning架构这里要设置 model，trainer，datamodule

    my_model = EGNN_dynamics(args).to(rank)
    

    print(f"{sum(p.numel() for p in my_model.parameters()):,} parameters in total.")
    print(f"{sum(p.numel() for p in my_model.parameters() if p.requires_grad):,} trainable parameters.")
    print(f"[Rank{rank} Using GPU: {torch.cuda.get_device_name(rank)}({torch.cuda.current_device()})]")
    if rank == 0 or not args.use_multi_gpu:
        summarize_model(my_model)
    if args.use_multi_gpu:
        my_model = torch.nn.parallel.DistributedDataParallel(my_model, device_ids=[rank], find_unused_parameters=True)

# loading data
    if rank == 0 or not args.use_multi_gpu:
        print('Loading dataset...')

# split data
    need_split =  (not check_data_split(path=args.data_list_path)) or args.regenerate_data_list
    if need_split:
        if rank == 0 or not args.use_multi_gpu:             
            print('Generating data list...')
    train_list, val_list, test_list = split_pdbbind(args.pdbbind_path, args.data_split_rate, core_list_path=args.core_list_path)
    save_data_split(train_list, val_list, test_list, path=args.data_list_path)


    train_list, val_list, test_list = load_data_split(path=args.data_list_path, blind_training=args.blind_training)
    if rank == 0 or not args.use_multi_gpu:
        print(f'train_list: {len(train_list)}, val_list: {len(val_list)}, test_list: {len(test_list)}')
    
# get dataset and dataloder

        args.feature_size = 8

    
    # with open(args.vocab_path, 'rb') as f:                                # 但是推理的时候是要用这部分内容啊
    #     vocab = [x.strip() for x in f.readlines()]                
    # vocab = Vocab(vocab, fp_df=pd.read_csv(vocab_fp_path, index_col=0))   
    train_dataset = Fragment('train', train_list, args.dataname)
    val_dataset = Fragment('val', val_list, args.dataname)
    train_loader, train_sampler, val_loader, val_sampler = get_dataloader(args, train_dataset, val_dataset,
                                                                        args.world_size, collate_fn=collate_struct)
    train_loader.dataset.training = True
    val_loader.dataset.training = False

# loss
    loss_object = loss_object = DiffusionLoss(args).to(rank)
    if args.use_multi_gpu_for_loss_object:
        loss_object = torch.nn.parallel.DistributedDataParallel(loss_object, device_ids=[rank], find_unused_parameters=False)
    dic_opt = get_LigPose_params(my_model, loss_object, args)
    opt_object = CustomOptimization(args, dic_opt)

# init_weight and restart
    my_model, loss_object, opt_object, dic_traj = init_params(args, my_model, loss_object, opt_object)

# Tensorboard
    if rank == 0 or not args.use_multi_gpu:
        writer = init_tensorboard(args.log_dir, args.log_port, args.restart, dic_traj,
                                start_new_tensorboard=args.start_new_tensorboard,
                                drop_head_epoch=args.log_drop_head_epoch,
                                sele_env=args.log_env)

# train loop
    if rank == 0 or not args.use_multi_gpu:
        print('Begin training Diffusion model...')
    for epoch in range(args.restart, args.n_epoch):
        
        my_model.train(True)                                        # 这个是不是不用设置这个train模式呢？没有使用到 self.training
        loss_object.train(True) 
        
        if args.use_multi_gpu:
            train_sampler.set_epoch(epoch)
        if rank == 0 or not args.use_multi_gpu:                     # 这个epoch是不是不需要手动写呢？
            print(f"Epoch [{epoch + 1}/{args.n_epoch}], lr: {opt_object.optimizers[0].param_groups[0]['lr']:.2e}")
            progBar = Progbar(len(train_loader))
            dic_train = defaultdict(list)
        
        for i, batch in enumerate(train_loader):
            batch = batch.to(rank)
        
        # clear gradients
            for opt in opt_object.optimizers:
                opt.zero_grad()
        
        # forward loss
            if args.use_amp_mix:                                   
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    out  = loss_object(my_model, batch) 
                    loss = out["loss"]                         
            else:
                out  = loss_object(my_model, batch)
                loss = out["loss"]

        # backward
            if args.use_amp_mix and hasattr(opt_object, "scaler"):
                opt_object.scaler.scale(loss).backward()
                for opt in opt_object.optimizers:
                    opt_object.scaler.step(opt)
                opt_object.scaler.update()
            else:
                loss.backward()
                for opt in opt_object.optimizers:
                    opt.step()

        #     if rank == 0 or not args.use_multi_gpu:                    
        #         progBar.update(i + 1, [] if args.mute_progbar
        #         else [*[(k, np.around(v, 5)) for k, v in eval_loss.items()]])
        #         for k in loss.keys():
        #             dic_train[k].append(loss[k])
        
        # if rank == 0 or not args.use_multi_gpu:
        #     for k in loss.keys():
        #         dic_traj['train'][k].append(np.mean(dic_train[k]))

        # if args.use_multi_gpu:
        #     torch.distributed.barrier()
            if rank == 0 or not args.use_multi_gpu:
                progBar.update(i + 1, [] if args.mute_progbar
                                else [("loss", float(loss.detach().cpu().item()))])
                dic_train["loss"].append(float(loss.detach().cpu().item()))
            if rank == 0 or not args.use_multi_gpu:
                dic_traj['train']['loss'].append(np.mean(dic_train["loss"]))

            if args.use_multi_gpu:
                torch.distributed.barrier()


# validation loop
        my_model.train(False)
        loss_object.train(False)
        if args.use_multi_gpu:
            val_sampler.set_epoch(epoch)
        
        
        dic_val = defaultdict(list)
        with torch.no_grad():
            if rank == 0 or not args.use_multi_gpu:
                progBar = Progbar(len(val_loader))
            
            for i, batch in enumerate(val_loader):
                batch = batch.to(rank)

                out  = loss_object(my_model, batch)
                loss = out["loss"]
                
                dic_val["loss"].append(float(loss.detach().cpu().item()))
                if rank == 0 or not args.use_multi_gpu:
                    progBar.update(i + 1, [] if args.mute_progbar
                                else [("val_loss", float(loss.detach().cpu().item()))])                         
                
        
        if rank == 0 or not args.use_multi_gpu:
            mean_val_loss = float(np.mean(dic_val["loss"]))
            dic_traj['val']['loss'].append(mean_val_loss)
            save_val(dic_val, rank)

        if args.use_multi_gpu:
            torch.distributed.barrier()

# ---- update scheduler ----
        opt_object.update_schedulers(epoch=epoch)
        
# ----save checkpoint and update tensorboard----
        if rank == 0 or not args.use_multi_gpu:
            # lr 曲线
            cur_lr = opt_object.optimizers[0].param_groups[0]['lr']
            dic_traj['train']['lr_curve'].append(cur_lr)
            dic_traj['val']['lr_curve'].append(cur_lr)

            if (epoch + 1) % args.weight_save_freq == 0:
                save_params(args, my_model, loss_object, opt_object, dic_traj,
                            f'{args.weight_path}/diffusion_state_{epoch + 1}.chk')

            update_tensorboard(writer, dic_traj, epoch, drop_head_epoch=args.log_drop_head_epoch)

        torch.cuda.empty_cache()
        if args.use_multi_gpu:
            torch.distributed.barrier()
    
    if args.use_multi_gpu:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()


    # 数据路径和列表
    
    # data source
    parser.add_argument('--pdbbind_path', type=str,
                        default='/data/lpw/ligpose/data/work_file/tmp',
                        help='path to prepared data')
    parser.add_argument('--l_npz_path', type=str,
                        default='./',
                        help='dataset path')
    parser.add_argument('--p_npz_path', type=str,
                        default='./',
                        help='dataset path')
    parser.add_argument('--c_npz_path', type=str,
                        default='./',
                        help='dataset path semi')
    parser.add_argument('--choose_start_weight', type=str,
                        default=None,
                        help='init weight (restart=0), None for random')
    parser.add_argument('--regenerate_data_list', type=str,
                        default=False, help='regenerate data split. (if no file exists)')
    parser.add_argument('--data_list_path', type=str,
                        default='/data/lpw/ligpose/data/work_file/tmp',
                        help='path to data list, core_test/core_test_reduce_train/som_pretrain/som_pretrain_reduce_train/zinc250k')
    parser.add_argument('--data_split_rate', type=str,
                        default='0.75-0.05-0.2',
                        help='rate for training, validation and testing. 0.75-0.05-0.2/0.95-0.025-0.025')

    # 现有参数（从train_pdbbind_struct.py复制）
    
    
    parser.add_argument('--core_list_path', type=str, default='/home/smileknight/learn/LigPose_demo_linux/eval/pdbbind/core_list.txt', help='core set of PDBbind')
    parser.add_argument('--blind_training', type=str, default=False, help='Blind training')
    

    # 训练设置
    parser.add_argument('--restart', type=int, default=0, help='restart step, set to 0 for new run')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.99, help='learning rate decay')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay (L2 penalty)')
    parser.add_argument('--n_epoch', type=int, default=10000, help='number of epoch')
    parser.add_argument('--update_together', type=str, default=False, help='if update grad together')
    parser.add_argument('--lr_sche_batch_wise', type=str, default=None,
                    help="lr schedulers update per batch, split by ','  e.g. '0,1'. Warning: cause wrong lr if restart>0")
    parser.add_argument('--persistent_workers', type=str, default=True,
                    help='dataloader persistent_workers, True may cause wrong values for func involved in dataset.epoch')
    parser.add_argument('--coor_scale', type=int, default=10, help='coordinate scaler')
    parser.add_argument('--aff_scale', type=int, default=10, help='affinity scaler')

    
    # 模型类型
    parser.add_argument('--model_type', type=str, default='struct', choices=['struct', 'scr'], help='Model type: struct or scr')
    
    # 模型超参数
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
    # 时间嵌入参数
    parser.add_argument('--time_emb_dim', type=int, default=256, help='time embedding dimension')
    parser.add_argument('--time_emb_mode', type=str, default='sinusoidal', choices=['simple', 'sinusoidal'], help='time embedding mode')
    
    # Diffusion特有参数
    parser.add_argument('--diffusion_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--diffusion_beta_start', type=float, default=1e-4, help='Initial beta value for noise schedule')
    parser.add_argument('--diffusion_beta_end', type=float, default=0.02, help='Final beta value for noise schedule')
    parser.add_argument('--schedule_type', type=str, default='linear', choices=['linear', 'cosine'], help='Type of noise schedule')
    parser.add_argument('--noise_loss_weight', type=float, default=1.0, help='Weight for the diffusion noise prediction loss')
    
    # 损失权重
    parser.add_argument('--gamma_1', type=float, default=1., help='loss weight for coordinate')
    parser.add_argument('--gamma_2', type=float, default=1, help='loss weight for affinity')
    parser.add_argument('--gamma_noise', type=float, default=0.1, help='coor noise loss weight')
    parser.add_argument('--gamma_p_x_1', type=float, default=0.1, help='BERT-like mask loss weight')
    parser.add_argument('--gamma_p_x_2', type=float, default=0.1, help='BERT-like mask loss weight')
    parser.add_argument('--gamma_l_x', type=float, default=0.1, help='BERT-like mask loss weight')
    parser.add_argument('--gamma_edge', type=float, default=0.1, help='BERT-like mask loss weight')
    
    # 设备设置
    parser.add_argument('--local_rank', type=int, default=1,
                        help='由 PyTorch 分布式启动器自动设置的当前进程的本地 GPU 排名。最好不要手动设置此参数。')
    parser.add_argument('--use_seed', type=str, default=True, help='use random seed')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers in Dataloader')
    parser.add_argument('--use_multi_gpu', type=str, default=False, help='if use GPUs')
    parser.add_argument('--use_multi_gpu_for_loss_object', type=str, default=False, help='if loss object need use GPUs')
    parser.add_argument('--gpu_list', type=str, default='0', help='available GPU list')
    parser.add_argument('--world_size', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--port', type=str, default=str(random.randint(10000, 20000)), help='parallel port')
    parser.add_argument('--use_amp_mix', type=str, default=False, help='mixed precision')
    
    # 其他设置
    parser.add_argument('--cache_path', type=str, default='./cache', help='path to tmp data')
    parser.add_argument('--weight_save_freq', type=int, default=5, help='save model every x epoch')
    parser.add_argument('--weight_path', type=str, default='/home/lpw/ligpose_learn/weights/diffsuion', help='path to save model')
    parser.add_argument('--log_dir', type=str, default='./diffusion_log', help='Tensorboard path')
    parser.add_argument('--log_port', type=int, default=1212, help='Tensorboard port')
    parser.add_argument('--vis', type=str, default=False, help='save visualization')
    parser.add_argument('--vis_path', type=str, default='./vis', help='visualization path')
    parser.add_argument('--start_new_tensorboard', type=str, default=False, help='start a new tensorboard')
    parser.add_argument('--log_drop_head_epoch', type=int, default=0, help='omit first epochs in Tensorboard')
    parser.add_argument('--log_env', type=str, default='py36', help='Tensorboard env name')
    parser.add_argument('--mute_progbar', type=str, default=False, help='if mute Progbar')
    
    
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

    args = parser.parse_args()
    
    args.data_split_rate = split_rate(args.data_split_rate)
    print(args)
    if args.use_seed:
        set_all_seed(args.seed)
        
    train_with_args(args, train_diffusion)
    print('='*20 + ' DIFFUSION TRAINING DONE ' + '='*20)
