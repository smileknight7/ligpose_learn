import os
import shutil
import sys
import math
import numpy as np
import random
import pickle
import copy
from tqdm import tqdm
import scipy
import scipy.spatial
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path


import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import torch_geometric


from utils.common import load_idx_list
from utils.data_utils import pad_zeros, batch_index_select_for_edge, batch_index_select
from utils.pdbbind_preprocess import gen_pdbbind_screening_list
from utils.training_utils import load_data_split, save_data_split

#from training_utils_revise import load_data_split, save_data_split

# # for test
# from common import load_idx_list
# from data_utils import pad_zeros, batch_index_select_for_edge, batch_index_select
# from pdbbind_preprocess import gen_pdbbind_screening_list
# from training_utils import load_data_split, save_data_split



#拆分数据集，为训练集和测试集          所以最开始要对所有数据进行处理
def split_pdbbind(pdbbind_path, data_split_rate, core_list_path=None):
    if isinstance(core_list_path, type(None)):
        pdb_list = os.listdir(pdbbind_path)
        random.shuffle(pdb_list)

        l = len(pdb_list)
        cut_1 = int(data_split_rate[0] * l)
        cut_2 = cut_1 + int(data_split_rate[1] * l)
        train_list = pdb_list[:cut_1]
        val_list = pdb_list[cut_1:cut_2]
        test_list = pdb_list[cut_2:]
    else:
        pdb_list = os.listdir(pdbbind_path)
        test_list = [f'{i}.npz' for i in load_idx_list(core_list_path) if f'{i}.npz' in pdb_list]
        rest_list = [i for i in pdb_list if i not in test_list]
        random.shuffle(rest_list)

        l = len(rest_list)
        cut_2 = int(data_split_rate[1] * l)
        val_list = rest_list[:cut_2]
        train_list = rest_list[cut_2:]

    return train_list, val_list, test_list

def get_semi_list(protein_dir, ligand_dir, output_txt, seed):
    protein_names = [f.stem for f in Path(protein_dir).glob("*.npz")]
    ligand_names = [f.stem for f in Path(ligand_dir).glob("*.npz")]
    protein_names.sort()
    ligand_names.sort()
    print(ligand_names)
    random.seed(seed)
    random.shuffle(ligand_names)
    random.shuffle(protein_names)
    n = min(len(protein_names), len(ligand_names))
    protein_sel = protein_names[:n]
    ligand_sel = ligand_names[:n]
    with open(output_txt, "w") as f:
        for p_name, l_name in zip(protein_sel, ligand_sel):
            f.write(f"{p_name}-{l_name}semi\n")
def load_semi_list(semi_list_path):
    semi_list = []
    with open(semi_list_path, "r") as f:
        for line in f:
            semi_list.append(line.strip())
    return semi_list
# 这里要测试一下写成什么样子



#有core_list_path时是要将core_list_path中的数据作为测试集，其余数据再进行划分训练集和验证集
#data_split_rate default'0.75-0.05-0.2',



#collate_struct 的索引空间分为三类(这里不适用使用稀疏矩阵进行处理)

# N = max_len_complex_before_sampling       （protein, ligand, pad，pad）这部分张量按照全图节点 800
# N = len_complex_after_sampling                                                              700
# batch-level / 辅助信息 / 非节点对齐量        上面的全图空间映射到子图空间




#构建batch数据------>构建PYG格式的数据   （每个epoch会进行一次补一样的n_cycle采样）
def collate_struct(batch_list):                                                                                   # 这里要注意这个 max_len 是进行了改变的，这里的max_len是样本内的最大值                                                                # batchlist就是构建的dataset集合
    batch_list = [g for g in batch_list if g is not None]
    if len(batch_list) == 0:
        return None
    # get max len
    max_len_complex_before_sampling = 0
    max_len_protein_before_sampling = 0
    max_len_ligand = 0
    max_len_complex_after_sampling = 0
                                                                                                # 增加fragment节点长度 max_len_fragment = 0
                             # 下面主要是对同一个batch内的不同graph数据进行对齐
    for g in batch_list:     #batch_list = [dataset[0], dataset[1], dataset[2], dataset[3]]，batchlist是这种形式的，而dataset是利用pdb_list按照顺序构建的
        max_len_complex_before_sampling = max(max_len_complex_before_sampling, g['len_complex_before_sampling'])
        max_len_protein_before_sampling = max(max_len_protein_before_sampling, g['len_protein_before_sampling'])
        max_len_ligand = max(max_len_ligand, g['len_ligand'])                                   # 如果是在这里添加fragment节点的话要这样 max_len_fragment = 0
        max_len_complex_after_sampling = max(max_len_complex_after_sampling, g['len_complex_after_sampling'])
        g['node_mask_after_sampling'] = torch.ones(g['len_complex_after_sampling'])                                 # 这两个是新构建的
        g['edge_mask_after_sampling'] = torch.ones(g['len_complex_after_sampling'], g['len_complex_after_sampling'])

                                                                                                                         # 每个元素g是一个样本包含分子图的结构信息字典，通过g['']来获取对应信息
                                                                                                                         # 首先是进行初始化，随后进行计算蛋白，小分子，复合物三种样本的最大长度，方便后面进行pad对齐
                                                                                                                         # 建立node_mask作为键值对存入g中，为了方便区分padding部分和非padding部分
# 对样本数据进行padding填充，通过update进行加和

    # feat & coor
    dic_data = {}
    dic_data.update(pad_zeros(batch_list,                                  # 按照batch进行pad --->  （N, F）---> (B, N, F) 
                    [
                        'protein_node_feature_init',
                    ],
                    max_len_protein_before_sampling,
                    collect_dim=-3, data_type='1d', output_dtype=torch.float))        # 这里是对蛋白质的节点特征进行padding--主要是原子数量哪一类
    dic_data.update(pad_zeros(batch_list,
                    [
                        'ligand_node_feature_init',
                    ],
                    max_len_ligand,
                    collect_dim=-3, data_type='1d', output_dtype=torch.float))
    dic_data.update(pad_zeros(batch_list,
                    [
                        'edge_feature_init',
                    ],
                    max_len_complex_before_sampling,
                    collect_dim=-4, data_type='2d', output_dtype=torch.float))
    dic_data.update(pad_zeros(batch_list,
                    [
                        'coor_init', 'coor_true'
                    ],
                    max_len_complex_before_sampling,
                    collect_dim=-3, data_type='1d', output_dtype=torch.float))


# fragment部分


            

    # index padding for cycling
    # node_sampling_loc: [n_cycle, n_loc]
    # Warning: no paddings inside n_loc (i.e. between protein and ligand),
    # nodes have to be selected with idx_remove_middle_pad before

# 这里可以看出pad_zeros和F.pad的不同，前一个是pad nodes节点，后一个是pad feature部分
# ....因为这个pad_zeros是自定义的


#node_sampling_loc（内部的数据是原本大图的节点索引protein_before_sample的node索引---->这里被padding到了protein_after_sample的长度）
    dic_data['node_sampling_loc'] = torch.stack([                                 # complex_graph中的node_sampling_loc----->(n_cycle, len_after_sampling)
        F.pad(g['node_sampling_loc'],
              (0, max_len_complex_after_sampling - g['node_sampling_loc'].shape[1]),  # node_sampling_loc的形状是这样的(n_cycle, len_after_sampling)
              'constant',                                                            # len_after_sampling内部元素就是每次采样出的ptotein_core+other+ligand
              max_len_complex_before_sampling - 1)                                  # max_len_complex_before_sampling-1 used for padding sampling
        for g in batch_list], dim=1).long()  # to (n_cycle, n_batch, n_loc)----->         # 注意这里是dim=1将n_batch堆叠到了中间的维度中了
                                                                                          # max_len_complex_before_sampling - 1以这个数字进行pad，可能是作为一个标签表示这个图不合法？？
                                                                                          # 这个是哨兵索引，指向在整图padding时，额外留的最后一列，避免越界，也让被pad的位置在后续进行gather的时候拿到的的是无效的零行列

#这里最后pad成了(n_cycle, n_batch, n_loc)这个形状，最后一维的长度是max_len_complex_after_sampling这个，内容不是0



# 调整数据的位置，这样其他位置就是进行batch内pad的内容了
    # nodes: [protein, pad, ligand, pad] -> [protein, ligand, pad, pad]
    dic_data['idx_remove_middle_pad'] = torch.stack([                             # 这里显示在一个batch中对索引进行了拼接，随后又按照batch_list进行了堆叠
        torch.cat([
            torch.arange(0, g['len_protein_before_sampling']),                                                  # 有意义区块索引
            torch.arange(max_len_protein_before_sampling, max_len_protein_before_sampling + g['len_ligand']), 
            torch.arange(g['len_protein_before_sampling'], max_len_protein_before_sampling),                    # padding区块索引
            torch.arange(max_len_protein_before_sampling + g['len_ligand'],
                         max_len_protein_before_sampling + max_len_ligand)
        ], dim=0) for g in batch_list], dim=0).long()


# 进行样本对齐（这里的mask主要是针对，一些核心的点protein和ligand）
    dic_data.update(pad_zeros(batch_list,
                    [
                        'node_cycling_mask', 'ligand_mask_after_sampling',
                        'node_mask_after_sampling',                                         # 'len_complex_after_sampling'mask的形状是长度为这个东西的，1维度张量内部填充的是1
                    ],                                                                      # 这里只是补齐到了max_len_complex_after_sampling这个长度
                    max_len_complex_after_sampling,
                    collect_dim=-2, data_type='1d', output_dtype=torch.float))
    dic_data.update(pad_zeros(batch_list,
                    [
                        'edge_cycling_mask', 'edge_mask_after_sampling',                    # (len_complex_after_sampling,len_complex_after_sampling)内部填充之为1的张量
                    ],                                                                      # data_type='2d'这个就是说从倒数第3维开始对两个维度进行padding
                    max_len_complex_after_sampling,
                    collect_dim=-3, data_type='2d', output_dtype=torch.float))
    dic_data.update(pad_zeros(batch_list,                                        # 这里是对采样前后的ligand原子索引进行padding
                    [
                        'ligand_node_loc_before_sampling', 'ligand_node_loc_after_sampling',
                    ],
                    max_len_ligand,
                    collect_dim=-2, data_type='1d', output_dtype=torch.long))
    dic_data.update(pad_zeros(batch_list,                                       # 这里是对采样前的protein原子索引进行padding 
                    [
                        'protein_node_loc_before_sampling',
                    ],
                    max_len_protein_before_sampling,
                    collect_dim=-2, data_type='1d', output_dtype=torch.long))



#这里是在对与ligand相关的索引---->扁平化为跨batch的全局索引，并构建一些scatter中用到分组id----->方便进行聚合/还原？？？？？



    # For ligand matching                                           这里主要是讲ligand索引构建batch的索引偏移   
    len_tmp_after_sampling = 0                                      # 分为三个空间 len_tmp_after_sampling(complex), 
    len_tmp_ligand = 0                                              #             len_tmp_ligand(ligand),
    len_tmp_batch_match = 0                                         #           len_tmp_batch_match(ligand_match),
    ligand_node_loc_after_sampling = []
    ligand_match = []
    ligand_nomatch = []
    scatter_ligand_1 = []
    scatter_ligand_2 = []
    for i, g in enumerate(batch_list):                                                                          # i就是第几个complex_graph，g--->dataset[i](complex_graph)
        ligand_node_loc_after_sampling.append(g['ligand_node_loc_after_sampling'] + len_tmp_after_sampling)              # ligand_match是将不同的匹配拼接到一起了内含元素就是length_ligand*matchs
        ligand_match.append(g['ligand_match'] + len_tmp_ligand)             # 注意：这里只是构建成ligand的全局偏移
        ligand_nomatch.append(g['ligand_nomatch'] + len_tmp_ligand)
        
        # scatter_ligand_1   ligand匹配组映射到原子编号上(给的偏移量是匹配组数)
        # scatter_ligand_2   ligand匹配组归属的样本编号(给的偏移量是样本数)     样本数就是batch内位置
        
        # ligand原子属于哪个匹配组
        scatter_ligand_1.append(repeat(torch.arange(0, len(g['ligand_match']) // g['len_ligand']),      # len(g['ligand_match']) // g['len_ligand'] -----> num_match
                                       'i -> (i m)', m=g['len_ligand']) + len_tmp_batch_match)                   # 这里是产生与ligand_match对齐的索引向量（这里是将来ligand索引加上batch内的位置实现batch偏移）
        # 匹配组属于哪个样本
        scatter_ligand_2.append(torch.zeros(len(g['ligand_match']) // g['len_ligand']) + i)                         # 这个构建的是ligand归属的batch内（样本）列表
        
        len_tmp_after_sampling += max_len_complex_after_sampling                 # 样本偏移量累积
        len_tmp_ligand += g['len_ligand']
        len_tmp_batch_match += len(g['ligand_match']) // g['len_ligand']
    
    # 这里是在构建batch的全局索引拼接 ----->不带batch维度 （这个ligand_match是做对称平均最小化的）                              也就是将多个complex_graph中的ligand原子索引进行拼接构成一个大图
    dic_data['ligand_node_loc_after_sampling_flat'] = torch.cat(ligand_node_loc_after_sampling, dim=0).long()
    dic_data['ligand_match'] = torch.cat(ligand_match, dim=0).long()
    dic_data['ligand_nomatch'] = torch.cat(ligand_nomatch, dim=0).long()
    dic_data['scatter_ligand_1'] = torch.cat(scatter_ligand_1, dim=0).long()
    dic_data['scatter_ligand_2'] = torch.cat(scatter_ligand_2, dim=0).long()


    # for suppl info
    dic_data['aff_true'] = torch.cat([g['aff_true'] for g in batch_list], dim=0).float()   # 是不是一个ligand有多个值？（这个有空要再看看）
    dic_data['aff_mask'] = torch.Tensor([g['aff_mask'] for g in batch_list]).float()
    dic_data['coor_mask'] = torch.Tensor([g['coor_mask'] for g in batch_list]).float()
    dic_data['len_ligand'] = torch.Tensor([g['len_ligand'] for g in batch_list]).float()
    dic_data['idx'] = [g['idx'] for g in batch_list]


    # pre sample edge
    # unlike nodes, no padding between protein and ligand: [protein, ligand, pad]
    # edge_feature: [protein edges, ligand edges, pad edges]
    # node_feature: [protein nodes, pad, ligand nodes, pad]

    dic_data['edge_feature_init_cycle'] = torch.stack([
        batch_index_select_for_edge(dic_data['edge_feature_init'], loc) for loc in dic_data['node_sampling_loc']
    ], dim=0).float()


    # for masking （这里的mask是针对哪些点的特征要归零）
    dic_data.update(pad_zeros(batch_list,                                # (n,) ----> （b，n）
                    [
                        'p_x_mask_bool', 'l_x_mask_bool',                           # 这个是对protein和ligand部分的node设置的掩码进行padding
                    ],
                    max_len_complex_before_sampling,
                    collect_dim=-2, data_type='1d', value=False, output_dtype=torch.bool))
    dic_data.update(pad_zeros(batch_list,
                    [
                        'edge_mask_bool',                                           # 这个是对复合物部分的edge设置的掩码进行padding
                    ],
                    max_len_complex_before_sampling,
                    collect_dim=-3, data_type='2d', value=False, output_dtype=torch.bool))
    dic_data.update(pad_zeros(batch_list,
                    [
                        'p_x_mask_label_1', 'p_x_mask_label_2', 'l_x_mask_label',   # p_x_mask_label_1  protein原子，protein_resi，ligand类别padding（由one_hot转的）
                    ],
                    max_len_complex_before_sampling,
                    collect_dim=-2, data_type='1d', output_dtype=torch.long))
    dic_data.update(pad_zeros(batch_list,
                    [
                        'edge_mask_label',                                           # edge_mask_label  edge类别padding（由one_hot转的）   
                    ],
                    max_len_complex_before_sampling,
                    collect_dim=-3, data_type='2d', output_dtype=torch.long))


# emmm这部分是用于进行loss计算的

# 这里是从complex_graph中采样出子图索引                                               # node_sampling_loc---->(n_cycle, n_batch, n_loc)  他这个n_cycle在外边可以用于循环
    # for masking pre-sampling                                                      # 这里是堆叠n_cycle后的p_x_mask_bool，下面也是一样的
    dic_data['p_x_mask_bool_cycle'] = torch.stack([                         # loc 形状: (n_batch, n_loc)     每次取出一个 cycle → (n_batch, n_loc)  最后是在cycle的维度上进行堆叠dim=0-----最后的形状应该是(n_cycle, n_batch, n_loc)
        torch.gather(dic_data['p_x_mask_bool'], dim=-1, index=loc) for loc in dic_data['node_sampling_loc']   
    ], dim=0).bool()
    dic_data['l_x_mask_bool_cycle'] = torch.stack([
        torch.gather(dic_data['l_x_mask_bool'], dim=-1, index=loc) for loc in dic_data['node_sampling_loc']
    ], dim=0).bool()
    dic_data['edge_mask_bool_cycle'] = torch.stack([
        batch_index_select_for_edge(dic_data['edge_mask_bool'], loc, mask=True) for loc in dic_data['node_sampling_loc']
    ], dim=0).bool()
    dic_data['p_x_mask_label_1_cycle'] = torch.stack([
        torch.gather(dic_data['p_x_mask_label_1'], dim=-1, index=loc) for loc in dic_data['node_sampling_loc']
    ], dim=0).long()
    dic_data['p_x_mask_label_2_cycle'] = torch.stack([
        torch.gather(dic_data['p_x_mask_label_2'], dim=-1, index=loc) for loc in dic_data['node_sampling_loc']
    ], dim=0).long()
    dic_data['l_x_mask_label_cycle'] = torch.stack([
        torch.gather(dic_data['l_x_mask_label'], dim=-1, index=loc) for loc in dic_data['node_sampling_loc']
    ], dim=0).long()
    dic_data['edge_mask_label_cycle'] = torch.stack([
        batch_index_select_for_edge(dic_data['edge_mask_label'], loc, mask=True) for loc in dic_data['node_sampling_loc']
    ], dim=0).long()

    # dic_data['node_sampling_loc'] = (n_cycle, n_batch, n_loc padding to max_len_complex_after_sampling)
    
    # addidional batch info     按照给定的模式，把一个张量在新的维度上复制/扩展------>这里的作用是将batch的id拓展到每个循环和节点上，最后一维的数值就是batch内归属
    dic_data['x_batch_info'] = repeat(torch.arange(len(batch_list)), 'b -> c b n',         # (n_cycle，batch_list, complex_before) node中的信息是batch归属
                                      c=dic_data['node_sampling_loc'].size(0),                           
                                      n=max_len_complex_before_sampling).long()
    dic_data['edge_batch_info'] = repeat(torch.arange(len(batch_list)), 'b -> c b i j',
                                         c=dic_data['node_sampling_loc'].size(0),                        # (n_cycle，batch_list, complex_before， complex_before）
                                         i=max_len_complex_before_sampling,
                                         j=max_len_complex_before_sampling).long()
    
    dic_data['x_batch_info_cycle'] = torch.gather(dic_data['x_batch_info'], dim=-1, index=dic_data['node_sampling_loc']).long()   # 在最后一个维度上截取出n_loc的长度
                                                                                                              #  (n_cycle，batch_list, n_loc）
    


    edge_batch_info_cycle = torch.gather(dic_data['edge_batch_info'], dim=-1,                             
                                         index=repeat(dic_data['node_sampling_loc'], 'c b j -> c b i j',
                                                      i=max_len_complex_before_sampling))
    dic_data['edge_batch_info_cycle'] = torch.gather(edge_batch_info_cycle, dim=-2,
                                         index=repeat(dic_data['node_sampling_loc'], 'c b i -> c b i j',
                                                      j=edge_batch_info_cycle.size(-1))).long()          #  (n_cycle，batch_list, n_loc， n_loc）


    # for protein coor noise
    dic_data.update(pad_zeros(batch_list,
                    [
                        'coor_noise_bool',
                    ],
                    max_len_complex_before_sampling,
                    collect_dim=-2, data_type='1d', value=False, output_dtype=torch.bool))
    dic_data.update(pad_zeros(batch_list,
                    [
                        'coor_noise_true',
                    ],
                    max_len_complex_before_sampling,
                    collect_dim=-3, data_type='1d', output_dtype=torch.float))


    # coor noise pre-sampling
    dic_data['coor_noise_bool_cycle'] = torch.stack([
        torch.gather(dic_data['coor_noise_bool'], dim=-1, index=loc) for loc in dic_data['node_sampling_loc']
    ], dim=0).bool()
    dic_data['coor_noise_true_cycle'] = torch.stack([
        batch_index_select(dic_data['coor_noise_true'], loc) for loc in dic_data['node_sampling_loc']
    ], dim=0).float()




    # dic_data['flex_coor_mask_after_sampling'] = dic_data['ligand_mask_after_sampling']
    dic_data.update(pad_zeros(batch_list,
                    [
                        'flex_coor_mask',     # 标记cycle前complex中的ligand位置                                        
                    ],
                    max_len_complex_before_sampling,
                    collect_dim=-2, data_type='1d', output_dtype=torch.float))
    
    #flex_coor_mask_cycle------>    根据loc的采样顺序重排flex_coor_mask_cycle(protein/ligand)的掩码矩阵
    dic_data['flex_coor_mask_cycle'] = torch.stack([                                                            # 按照每轮的cycle在进行gather和堆叠----->#(batch_list，n_cycle,atom）
        torch.gather(dic_data['flex_coor_mask'], dim=-1, index=loc) for loc in dic_data['node_sampling_loc']      # 1个dic_data['node_sampling_loc']---->形状(protein_core_atom_loc , rest_atom_loc[cycle_i] , ligand_block)
    ], dim=0).float()                                                                                                   # loc应该是一个[cycle]的列表
                                                                                                                        # 堆叠之后 (n_cycle, batch_list, node_sampling_loc）   
    batch_data = torch_geometric.data.Data(**dic_data)
    return batch_data

# dic_data['node_sampling_loc'] 通过这个index对齐   ----->  最后长度都是max_len_complex_after_sampling




# 构建dataset数据
class ComplexStructDataset(torch.utils.data.Dataset):
    def __init__(self, mode, args, data_list, semi_list=None, cache_path='./cache'):
        self.mode = mode
        self.pdbbind_list = data_list
        self.fragment = args.fragment
        # pdbbind data source
        self.pdbbind_path = args.pdbbind_path
        self.original_path = args.original_path

        # unlabeled data source
        self.semi_list = semi_list
        self.l_npz_path = args.l_npz_path                               # 这里就是半监督训练有问题的原因了(好像没有构建半监督的数据)
        self.p_npz_path = args.p_npz_path
        self.c_npz_path = args.c_npz_path
        self.f_pkl_path = args.f_pkl_path                       


        # model hyperparameters
        self.n_cycle = args.n_cycle                                     # 循环次数
        self.coor_scale = args.coor_scale                               # 坐标缩放
        self.aff_scale = args.aff_scale                                 # 结合亲和力缩放
        self.max_len_before_sampling = args.max_len_before_sampling
        self.max_len_after_sampling = args.max_len_after_sampling if mode == 'train' else args.max_len_after_sampling_for_eval
        self.max_len_ligand = args.max_len_ligand
        self.max_ligand_atom_init_distance = args.max_ligand_atom_init_distance
        self.max_ligand_atom_pretrain_distance = args.max_ligand_atom_pretrain_distance

        # Sample pocket
        self.sample_pocket_flag = args.sample_pocket_flag               # 判断是否采样不同口袋
        self.select_pocket_type = args.select_pocket_type
        self.select_center_type = args.select_center_type

        # for masking                                                     
        self.semi_rate = args.semi_rate                                 # 半监督数据使用率
        self.mask_rate_l = args.mask_rate_l                             # ligand掩码率
        self.mask_rate_p = args.mask_rate_p                             # protein掩码率

        # for protein pretraining                                       # protein预训练 (这里为什么要叫预训练呢)
        self.init_protein_pretraining_label()
        self.noise_distance = args.noise_distance                       # 噪声距离(最开始的偏移距离)

        # for pocket dropout
        self.training = False
        self.dropout = args.dropout
        self.epoch = 0


    def __getitem__(self, i):
        if self.mode == 'train':
            if torch.rand(1) > self.semi_rate:
                f_name = self.pdbbind_list[i]
            else:
                f_name = random.choice(self.semi_list)  # 直接在 semi pool 中抽

        else:
            f_name = self.pdbbind_list[i]
        
        
        try:
            complex_graph = self.get_complex(f_name)                     # 特点主要在这里
        
        
        
        except Exception as e:
            print(f"[WARNING] Failed loading {f_name}, retrying with another complex...")
            
            for _ in range(5):
                
                f_retry = random.choice(self.pdbbind_list)
                try:
                    return self.get_complex(f_retry)
                    
                except Exception as e2:

                    print(f"[WARNING] Failed loading {f_retry}: {e2}")
            return None
                    #raise RuntimeError('Error in dataloader')                  # 原本的写法

        return complex_graph


#构建复合物图
    def get_complex(self, f_name):
        ################################################################################################################
        # load prepared data
        ################################################################################################################
        if '-' in f_name:
            data_type = 'semi'
        else:
            data_type = 'pdbbind'

        if data_type == 'semi':
            complex_idx = f_name[:-4]                                          # 这个是去掉文件后缀
            protein_idx = complex_idx.split('-')[0]         
            ligand_idx = complex_idx.split('-')[1]                             # 按道理这里是应该是分开的才对，分开之后应该是更灵活

            dic_pocket = np.load(f'{self.p_npz_path}/{protein_idx}.npz', allow_pickle=False)
            protein_node_feature_init = dic_pocket['protein_node_features']    # (protein_atoms, 78)
            protein_edge_feature_init = dic_pocket['protein_edge_features']    # (protein_atoms, protein_atoms, 6)
            protein_position_true = dic_pocket['protein_true_posi']            # (protein_atoms, 3)
            protein_pdb_info = dic_pocket['protein_pdb_info']                  # [A_45]这种指示蛋白链和氨基酸位置的表

            dic_ligand = np.load(f'{self.l_npz_path}/{ligand_idx}.npz', allow_pickle=True)
            ligand_node_feature_init = dic_ligand['ligand_node_features']       # (num_atoms, 42)
            ligand_edge_feature_init = dic_ligand['ligand_edge_features']       # (num_atoms，num_atoms, 6)
            # smi = dic_ligand['ligand_smiles']
            ligand_distmap = dic_ligand['ligand_distmap']                       # (num_atoms, num_atoms)分子内的距离
            ligand_match = dic_ligand['ligand_match']                           # (num_matches, num_atoms)
            ligand_position_true = np.zeros((len(ligand_node_feature_init), 3))

            #fragment_src
            if self.fragment:
                fragment_data = np.load(f'{self.f_pkl_path}/{ligand_idx}.npz', allow_pickle=True)   # 这里还要再想一下对semi数据这里要使用ligand的坐标吗
                dic_fragment = fragment_data["frag"].item() 
                fragment_src_pos = dic_fragment['fragment_src_pos']



            aff_true = -1                                                       # semi中特殊设置的内容
            aff_mask = 0
            ref_l_coor = dic_pocket['center_coor']                              # 这里暂时有问题（使用的还是liagnd_true_pos）
            coor_mask = 0


# pdbbind数据
        else:
            dic_data = np.load(f'{self.pdbbind_path}/{f_name}', allow_pickle=True)   #
            if self.fragment:
                fragment_data = np.load(f'{self.f_pkl_path}/{f_name}', allow_pickle=True)   # 这里还要再想一下对semi数据这里要使用ligand的坐标吗
                dic_fragment = fragment_data["frag"].item()
                fragment_src_pos = dic_fragment['fragment_src_pos']

            protein_node_feature_init = dic_data['protein_node_features']
            protein_edge_feature_init = dic_data['protein_edge_features']
            protein_position_true = dic_data['protein_true_posi']
            protein_pdb_info = dic_data['protein_pdb_info']
            ligand_node_feature_init = dic_data['ligand_node_features']
            ligand_edge_feature_init = dic_data['ligand_edge_features']
            ligand_position_true = dic_data['ligand_true_posi']                #（atom_num,3）
            ligand_distmap = dic_data['ligand_distmap']
            ligand_match = dic_data['ligand_match']
            aff_true = dic_data['aff']
            




            coor_mask = 1
            aff_mask = 1
            ref_l_coor = ligand_position_true


        ################################################################################################################
        # sample pocket  (划分口袋范围 ----> CA 原子为基准)
        ################################################################################################################
        if self.sample_pocket_flag and data_type == 'pdbbind':                                      # True
            CA_flag = protein_node_feature_init[:, -3] == 1                                         # 是否CA原子布尔索引
            max_len_protein = self.max_len_before_sampling - len(ligand_node_feature_init)          # 容纳蛋白质原子最大数量
            pocket_sub_index = self.sample_pocket(protein_position_true, ref_l_coor,
                                                  protein_pdb_info, CA_flag, max_len_protein)    # 这里指示以CA原子来计算距离筛选氨基酸，最后返回的数据仍是各种原子，可以称之为CA_pocket_atom_num
            
            protein_node_feature_init = protein_node_feature_init[pocket_sub_index]                                 # (select_atoms, 78)
            protein_edge_feature_init = protein_edge_feature_init[pocket_sub_index, :][:, pocket_sub_index]         # (select_atoms,select_atoms, 6)
            protein_position_true = protein_position_true[pocket_sub_index]                                         # (select_atoms, 3)        


# 所有数据
        ################################################################################################################
        # get length len_protein_before_sampling  （这个指的是节点循环采样之前）
        ################################################################################################################
        len_protein_before_sampling = len(protein_node_feature_init)                                                # 采样前protein_atoms数（最大800）            
        len_ligand = len(ligand_node_feature_init)
        len_complex_before_sampling = len_protein_before_sampling + len_ligand


        ################################################################################################################
        # scale coor / distance
        ################################################################################################################
        protein_position_true = protein_position_true / self.coor_scale
        ligand_position_true = ligand_position_true / self.coor_scale
        ref_l_coor = ref_l_coor / self.coor_scale                                                                   # 这里semi的数据不一样，应该是采用pocket中心
        ligand_distmap = ligand_distmap / self.coor_scale
        
        # 使用新的对数变换方法处理亲和力
        if data_type == 'pdbbind':
            aff_true = math.log(aff_true)
        else:
            aff_true = aff_true
        aff_true = aff_true / self.aff_scale


        ################################################################################################################
        # sampling nodes: core protein atom (CA,CB) retain, random sampling rest protein atom  循环采样
        ################################################################################################################
        # get core atoms
        assert len_ligand < self.max_len_after_sampling                                                             # 最大值：采样前是800，采样后是700
                                                                                                                    # max_len_ligand 是150
        # for CA CB
        core_atom_list = [-3, -33]  # CA -3, CB -33
        protien_core_atom_loc = [np.argwhere(protein_node_feature_init[:, i] == 1) for i in core_atom_list]       # 找到CA和CB原子的原子索引位置((n,1),(m,1）)
        protien_core_atom_loc = np.concatenate([x.reshape(-1) for x in protien_core_atom_loc], axis=-1)      # ((n,1),(m,1）)---->(n+m,)，将CA和CB索引进行拼接
        if len(protien_core_atom_loc) > self.max_len_after_sampling - len_ligand:
            core_atom_list = [-3]  # CA -3, CB -33
            protien_core_atom_loc = [np.argwhere(protein_node_feature_init[:, i] == 1) for i in core_atom_list]
            protien_core_atom_loc = np.concatenate([x.reshape(-1) for x in protien_core_atom_loc], axis=-1)  # 如果现在选取的节点数过多的话就只采用CA原子
        if len(protien_core_atom_loc) > self.max_len_after_sampling - len_ligand:
            protien_core_atom_loc = protien_core_atom_loc[:self.max_len_after_sampling - len_ligand]                # 最后这里是进行了一个兜底裁剪

        # sampling rest atoms(protein)
        rest_loc = np.delete(np.arange(len_protein_before_sampling), protien_core_atom_loc)           # 排除掉核心原子索引（CA,CB）
        rest_sampling_num = min(self.max_len_after_sampling - len_ligand - len(protien_core_atom_loc),
                                len_protein_before_sampling - len(protien_core_atom_loc))                           # 取了一个最小值来作为其他原子的采样数
        rest_atom_loc = [np.random.choice(rest_loc, size=rest_sampling_num, replace=False)                        # 剔除core index后剩余的protein_aton index
                         for _ in range(self.n_cycle)]             # 4                                              # 随机采样其他原子索引，采样次数为n_cycle(每次采样过程中rest原子是不一样的)
        node_sampling_loc_list = [np.concatenate([          # protein采样后和采样前中间的这部分内容还是需要padding的，ligand是从采样后protein的序号接起来的
            protien_core_atom_loc,
            x,                                                                                                      # 实际长度是这样的，索引编号可能会跳，但是总长度是一定的len(protien_core_atom_loc) + rest_sampling_num + len_ligand
            np.arange(len_ligand) + len_protein_before_sampling], axis=-1) # 采样最终的节点范围也是protein（before_sample）+ligand                                   
            for x in rest_atom_loc]                                                                                 # np.arange(len_ligand) + len_protein_before_sampling这里是将原本的ligand编号加上蛋白质采样长度，以免进行索引冲突
        node_sampling_loc = np.stack(node_sampling_loc_list, axis=0)      # [cycle, complex_num_aftersample]                                   # (n_cycle, len_after_sampling)表示的是循环多次后，最终采样的节点索引位置
        len_complex_after_sampling = node_sampling_loc.shape[1]    # 循环采样后的总节点数                                                 # 采样后复合物的节点数量   

        

# before_sample complex_edgefeature

        # cat edge_feature for protein and ligand       拼接protein和ligand的边特征(节点数量以sample前protein数和ligand计量)
        edge_feature_1 = np.concatenate(
            (
                protein_edge_feature_init,
                np.zeros((protein_edge_feature_init.shape[0], len_ligand, protein_edge_feature_init.shape[-1]))                     # 拼接protein与ligand边特征（ligand暂时先用数值0占位）
             ),                                                                                                                           # (N_p, N_p, F) ---> (N_p, N_p + N_l, F)                                                                   
            axis=1)
        edge_feature_2 = np.concatenate(                                                                                        
            (                                                                                                                      # (N_l, N_p, F) ---> (N_p，N_p + N_l, F)
                np.zeros((ligand_edge_feature_init.shape[0], len_protein_before_sampling, ligand_edge_feature_init.shape[-1])),     # ligand与protein边特征（protein暂时先用数值0占位）
                ligand_edge_feature_init
            ),
            axis=1)
        edge_feature_init = np.concatenate((edge_feature_1, edge_feature_2), axis=0)                                               # (N_p + N_l, N_p + N_l, F)                                           

                                                                                                                                          # len_complex_before_sampling这个代表的就是循环采样前的部分！！！！！！！！
# before_sample complex_coor
        ################################################################################################################
        # initialize ligand coor                        拼接protien和ligand pos  （后面要改的应该就是这个地方上）
        ################################################################################################################
        pocket_center = ref_l_coor.mean(axis=0)
        if self.fragment:
            ligand_position_init = fragment_src_pos+ \
                                    np.random.randn(len_ligand, 3) * self.max_ligand_atom_init_distance / self.coor_scale                                                                                                                        # self.max_ligand_atom_init_distance = 10            
        else:
            ligand_position_init = pocket_center + \
                                    np.random.randn(len_ligand, 3) * self.max_ligand_atom_init_distance / self.coor_scale            # 这里是对ligand的原子生成随机坐标，可以根据尺度进行放大缩小，然后移动到口袋中心的位置
        
        protein_position_init = protein_position_true                                                                           # 注意，上边的pocket_center形状是(3,)这种形式的，后面的运算是使用到了广播机制，实现了位置的平移

        coor_init = np.concatenate((protein_position_init, ligand_position_init), axis=0)                                # (len_complex_before_sampling, 3)初始坐标    (N_p + N_l,3)
        coor_true = np.concatenate((protein_position_true, ligand_position_true), axis=0)                                # (len_complex_before_sampling, 3)真实坐标


        ################################################################################################################
        # supply info                                                                                                           # len_complex_after_sampling循环采样后的节点总数
        ################################################################################################################
        # for recycle info, update core atoms, omit rest sampled atoms                                                            这部分是分别对跨cycle不变的节点和变化的节点进行处理(因为进行cycle循环采样的总数是固定的)
       
        # 构建循环采样中不变节点的索引和mask表     这个索引是新构建的
        node_cycling_loc = np.concatenate([np.arange(len(protien_core_atom_loc)),                # 这里表示的是循环采样过程中不变的节点的索引
                                           np.arange(len_complex_after_sampling - len_ligand,                             # 下面就是对不变部分构建mask矩阵了
                                                     len_complex_after_sampling)], axis=-1)
# 采样后core_protein和ligand部分mask
        node_cycling_mask = np.zeros(len_complex_after_sampling)                                                          # 节点索引部分
        node_cycling_mask[:len(protien_core_atom_loc)] = 1                                                                      # 构建mask列表----protein_CA_CB索引
        node_cycling_mask[-len_ligand:] = 1                                                                                     # 构建mask列表----ligand索引    这里进行了叠加，叠加之后就之后补充的蛋白原子索引为0,这里是从倒数len_ligand开始到最后一个赋值为1
        
# 采样后边部分mask边矩阵
        edge_cycling_mask = np.zeros((len_complex_after_sampling, len_complex_after_sampling))  
        edge_cycling_mask[:len(protien_core_atom_loc), :len(protien_core_atom_loc)] = 1         
        edge_cycling_mask[-len_ligand:, -len_ligand:] = 1

# 采样后中ligand部分mask和节点索引 (区分了采样前和采样后)   
        ligand_mask_after_sampling = np.zeros(len_complex_after_sampling)
        ligand_mask_after_sampling[np.arange(len_complex_after_sampling - len_ligand, len_complex_after_sampling)] = 1  # 采样后ligand的mask，标记了ligand部分
        
        ligand_node_loc_after_sampling = np.arange(len_complex_after_sampling - len_ligand, len_complex_after_sampling) # 采样后ligand的节点索引
        ligand_node_loc_before_sampling = np.arange(len_complex_before_sampling - len_ligand,                                # 循环采样前ligand的节点索引
                                                    len_complex_before_sampling)                                              # 采样前ligand原子索引
        # protein部分的节点索引（只有采样前）
        protein_node_loc_before_sampling = np.arange(len_protein_before_sampling)                                            # 采样前蛋白质原子索引

        ligand_match = ligand_match.reshape(-1)        # (num_matches, num_atoms)----> n_match*len_ligand
        n_match = len(ligand_match) // len_ligand
        ligand_nomatch = repeat(torch.arange(0, len_ligand), 'm -> (n m)', n=n_match)                     # 将当前ligand重复n_match次以对应ligand_match


        ################################################################################################################
        # to tensor
        ################################################################################################################
        protein_node_feature_init = torch.from_numpy(protein_node_feature_init).float()
        ligand_node_feature_init = torch.from_numpy(ligand_node_feature_init).float()
        edge_feature_init = torch.from_numpy(edge_feature_init).float()
        coor_init = torch.from_numpy(coor_init).float()
        coor_true = torch.from_numpy(coor_true).float()
        aff_true = torch.Tensor([aff_true]).float()
        protein_position_true = torch.from_numpy(protein_position_true).float()
        ligand_distmap = torch.from_numpy(ligand_distmap).float()

        node_sampling_loc = torch.from_numpy(node_sampling_loc).long()
        node_cycling_loc = torch.from_numpy(node_cycling_loc).long()
        node_cycling_mask = torch.from_numpy(node_cycling_mask).long()
        edge_cycling_mask = torch.from_numpy(edge_cycling_mask).long()
        ligand_mask_after_sampling = torch.from_numpy(ligand_mask_after_sampling).long()
        ligand_node_loc_after_sampling = torch.from_numpy(ligand_node_loc_after_sampling).long()
        protein_node_loc_before_sampling = torch.from_numpy(protein_node_loc_before_sampling).long()
        ligand_node_loc_before_sampling = torch.from_numpy(ligand_node_loc_before_sampling).long()
        ligand_match = torch.from_numpy(ligand_match).long()




# 下面这里有点问题，改成diffusion的话是对这里的选定内容加噪吗
        ################################################################################################################
        # feature masking
        ################################################################################################################
        # 构建监督信号索引       这里的mask是针对F的mask    (F)使用的时候要转为 ---> (1,1,F) 
# protein node feature mask                                                这里这个映射问题还要再看下
        p_atom_label = protein_node_feature_init[:, -37:]                                                                 # 截取protein_atom_label部分的特征
        p_res_label = protein_node_feature_init[:, -57:-37]                                                               # protein_atom_label部分的特征
        for res, a1, a2 in self.fix_p_atom_label_list:                                                                    # 6个等价原子对-----这一步会循环 6 次，每次会给一组三个向量 (res, a1, a2) *6
            equ_loc = ((p_res_label * res).sum(dim=-1) * (p_atom_label * a1).sum(dim=-1)) == 1                            # 返回值是这样的(num_atom,) 每次循环中(p_res_label * res).sum(dim=-1) 这里是对20个特征继续匹配然后求和，原子部分同样，最后再进行一次res与atom之间的匹配
            p_atom_label[equ_loc] = repeat(a2, 'd -> n d', n=int(equ_loc.float().sum()))                 # 这里实际上是将a2的one_hot复制了n次然后赋值到了之前挑选出来的识别到a1的one_hot
        p_x_mask_bool = self.gen_mask_index([p_atom_label, p_res_label], self.mask_rate_p)      # 构建mask列，针对protein
        p_x_mask_label_1 = p_atom_label.argmax(dim=-1)                                                                    # 原子类别标签 （由one_hot转来的）
        p_x_mask_label_2 = p_res_label.argmax(dim=-1)                                                                     # 氨基酸类别标签
                                                                                                   
# ligand node feature mask
        l_atom_label = ligand_node_feature_init[:, :10]                                                                   # 截取ligand_atom_label部分的特征        
        l_x_mask_bool = self.gen_mask_index([l_atom_label], self.mask_rate_l)                   # ligan的mask列
        l_x_mask_label = l_atom_label.argmax(dim=-1)                                                                      # 原子类别标签----->我的理解是将独热编码特征压缩成了一个数字标签列表



# edge mask是以矩阵形式构建的
        # complex edge feature mask                                     # 这里实现的是相加+对称化------->  (i,j) 和 (j,i) 要么一起被 mask，要么一起保留。
        p_edge_label = torch.from_numpy(protein_edge_feature_init)
        p_edge_mask_bool = self.gen_mask_index([p_edge_label], self.mask_rate_p)                # 蛋白质边掩码
        p_edge_mask_bool = (
                    p_edge_mask_bool.triu().float() + p_edge_mask_bool.float().triu().transpose(1, 0)           # 随机张量的上三角（转置之后，相加获得针对protein的边mask）
        ).bool()                                                                                                           
                                                                                                                          
        l_edge_label = torch.from_numpy(ligand_edge_feature_init)                                       
        l_edge_mask_bool = self.gen_mask_index([l_edge_label], self.mask_rate_l)                
        l_edge_mask_bool = (
                    l_edge_mask_bool.triu().float() + l_edge_mask_bool.float().triu().transpose(1, 0)
        ).bool()
# 构建复合物边
        edge_mask_bool = torch.cat([
            F.pad(p_edge_mask_bool, (0, len_ligand), 'constant', False),                              # 只有需要mask的部分是1，不mask和pad部分都是0
            F.pad(l_edge_mask_bool, (len_protein_before_sampling, 0), 'constant', False)              # 通过pad对空值部分进行填充
        ], dim=0).bool()                                                                                                   # (len_complex_before_sampling, len_complex_before_sampling,)最终掩码矩阵返回值                  
        edge_mask_label = edge_feature_init.argmax(dim=-1)                                                                 # 前面是蛋白部分，后面是配体部分




#semi数据
        # mask origin feat，结构掩码 ---->半监督部分使用mask对特征进行掩码
        if data_type == 'semi':
            protein_node_feature_init = F.pad(protein_node_feature_init, (0, 1), 'constant', 0)       # 标记被mask的原子是1，未进行mask的原子是0
            protein_node_feature_init[p_x_mask_bool] = 0         # 这里置零特征的方式有问题    protein_node_feature_init[p_x_mask_bool, -37:] = 0 按道理应该是这样做
            protein_node_feature_init[p_x_mask_bool, -1] = 1
            ligand_node_feature_init = F.pad(ligand_node_feature_init, (0, 1), 'constant', 0)
            ligand_node_feature_init[l_x_mask_bool] = 0
            ligand_node_feature_init[l_x_mask_bool, -1] = 1                                                                

            edge_feature_init = rearrange(F.pad(edge_feature_init, (0, 1), 'constant', 0), 'i j d -> (i j) d')
            edge_feature_init[edge_mask_bool.reshape(-1)] = 0                                                              # 应用边特征（展平之后用索引置零随后恢复）
            edge_feature_init[edge_mask_bool.reshape(-1), -1] = 1       # 
            edge_feature_init = rearrange(edge_feature_init, '(i j) d -> i j d',
                                          i=len_protein_before_sampling + len_ligand)
#pdbbind数据
        else:
            protein_node_feature_init = F.pad(protein_node_feature_init, (0, 1), 'constant', 0)      # 也就是上面构建的mask都是针对semi数据的
            ligand_node_feature_init = F.pad(ligand_node_feature_init, (0, 1), 'constant', 0)
            edge_feature_init = F.pad(edge_feature_init, (0, 1), 'constant', 0)

#所有数据
        # pad mask to complex length------>补齐掩码矩阵到complex的长度
        p_x_mask_bool = F.pad(p_x_mask_bool, (0, len_ligand), 'constant', False)                       # 将蛋白质的mask掩码补齐到复合物长度
        l_x_mask_bool = F.pad(l_x_mask_bool, (len_protein_before_sampling, 0), 'constant', False)      # p_x_mask_label 表示的是掩码前的标签（one_hot转来的）
        p_x_mask_label_1 = F.pad(p_x_mask_label_1, (0, len_ligand), 'constant', 0)
        p_x_mask_label_2 = F.pad(p_x_mask_label_2, (0, len_ligand), 'constant', 0)
        l_x_mask_label = F.pad(l_x_mask_label, (len_protein_before_sampling, 0), 'constant', 0)


        ################################################################################################################
        # protein noise
        ################################################################################################################
        p_coor_true = protein_position_true                                                                                 # (CA_pocket_atom_num, 3) 
        p_coor_noise_bool = self.gen_mask_index([p_res_label], self.mask_rate_p)
        #噪音部分原子
        coor_noise_bool = F.pad(p_coor_noise_bool, (0, len_ligand), 'constant', False)                 # 这里相当于是在protein后面补了ligand的原子部分设置为Flase
        coor_noise_true = F.pad(p_coor_true, (0, 0, 0, len_ligand), 'constant', 0)                     # (CA_pocket_atom_num, 3)，这里是在CA_pocket_atom_num后面补了ligand的原子数
                                                                                                                            #       





        # gen noise for coor init  坐标噪音---->半监督数据                                                                    # 标注ligand位置（以mask形式）
        if data_type == 'semi':                                                             # 针对mask数据添加随机噪声
            coor_init[coor_noise_bool] = coor_init[coor_noise_bool] + torch.randn(
                coor_init[coor_noise_bool].shape) * self.noise_distance / self.coor_scale
            flex_coor_mask = F.pad(p_coor_noise_bool, (0, len_ligand), 'constant', 1).float()
        
        else:
            flex_coor_mask = F.pad(torch.zeros(len_protein_before_sampling), (0, len_ligand), 'constant', 1).to(    
                torch.float)







        # for dist input    边特征中引入dismap（蛋白质就是蛋白质的，liand的dismap是fragment内的的原子之间构建的）
        pocket_dismap = (                                                           # 这里原本是这个样子的coor_init = np.concatenate((protein_position_init, ligand_position_init), axis=0)  
                coor_init[:len_protein_before_sampling].unsqueeze(1) -          # (P, 1, 3) 经过切片之后抛弃了ligand部分的坐标了
                coor_init[:len_protein_before_sampling].unsqueeze(0)            # (1, P, 3)
        ).norm(p=2, dim=-1)                                                         # 这里进行计算的话还用到了广播机制------>(P, P)  计算的是蛋白质原子之间的距离
        edge_feature_init = F.pad(edge_feature_init, (0, 1), 'constant', -1)            # 增加一列--->后面存放了dismap   
        edge_feature_init[:len_protein_before_sampling, :len_protein_before_sampling, -1] = pocket_dismap    # 将dismap赋值到了feature的最后一个特征
        edge_feature_init[-len_ligand:, -len_ligand:, -1] = ligand_distmap          # 这个ligand_dismap主要计算的内容是配体part内的原子距离和part之间的原子距离（其余位置是0）
        
# semi数据的dismap不管是protein还是ligand都设置为0
        if data_type == 'semi':
            edge_feature_init[p_x_mask_bool, :, -1] = -1
            edge_feature_init[:, p_x_mask_bool, -1] = -1
            edge_feature_init[l_x_mask_bool, :, -1] = -1
            edge_feature_init[:, l_x_mask_bool, -1] = -1


        #########################################################################
        # moltree (fragment_part) for ligand
        #########################################################################
        # with open('vocab_processed_path', 'rb') as f:     # vocab_df_crossdock.txt这个是最初的词表文件
        #     # Load the data from the file
        #     vocab_df = pickle.load(f)
        # smile_cluster_list = vocab_df['smile_cluster'].tolist()
        # vocab = Vocab(smile_cluster_list)
        
        
        # complex_idx = f_name[:-4]                               # 这里要想一下半监督数据是不是要改下处理方式
        # ligand_path = os.path.join(f'{self.original_path}/{complex_idx}/{complex_idx}.sdf')
        # mol = Chem.MolFromMolFile(ligand_path, sanitize=False)
        # mol_tree = MolTree(mol, vocab, ligand_path)
        # node_pos = torch.from_numpy(mol_tree.node_pos)
        # node_wid = torch.from_numpy(mol_tree.node_wid)
        # len_node_wid = len(node_wid)
        
        # size_vocab = vocab.size()
        # mol_tree = mol_tree

        ################################################################################################################
        # use Data in PyG
        ################################################################################################################
#所有数据
        complex_graph = dict(
            protein_node_feature_init=protein_node_feature_init,        # 原始数据
            ligand_node_feature_init=ligand_node_feature_init,
            edge_feature_init=edge_feature_init,                        # dismap(被拼接到edge_feature的最后一维),dismap主要是描述
                                                                        # dismap  protein-->protein和ligand-->ligand之间
          # fragment_edge_feature

            coor_init=coor_init,                                        # cycle前的complex_coor
            coor_true=coor_true,

            node_sampling_loc=node_sampling_loc,                        # [cycle, complex_num_aftersample]  用于cycle后的复合物中选取节点
            node_cycling_loc=node_cycling_loc,                          # 用于构造mask，断开的内容mask = 0(但是这个现在是没有使用的)
            
            node_cycling_mask=node_cycling_mask,                        # cycle后的不变的node(core+ligand)
            edge_cycling_mask=edge_cycling_mask,

            ligand_mask_after_sampling=ligand_mask_after_sampling,              # cycle后的ligand的mask
            ligand_node_loc_after_sampling=ligand_node_loc_after_sampling,      # cycle后ligand索引
            ligand_node_loc_before_sampling=ligand_node_loc_before_sampling,    # cycle前ligand索引
            protein_node_loc_before_sampling=protein_node_loc_before_sampling,  # cycle前protein索引
            len_protein_before_sampling=len_protein_before_sampling,

            len_complex_before_sampling=len_complex_before_sampling,
            len_complex_after_sampling=len_complex_after_sampling,

            ligand_match=ligand_match,
            ligand_nomatch=ligand_nomatch,
            len_ligand=len_ligand,

# fragment part(这部分暂时删掉了)

            p_x_mask_bool=p_x_mask_bool,                    # 这个就是用于对目标F进行加噪的索引
            l_x_mask_bool=l_x_mask_bool,
            edge_mask_bool=edge_mask_bool,
            p_x_mask_label_1=p_x_mask_label_1,
            p_x_mask_label_2=p_x_mask_label_2,
            l_x_mask_label=l_x_mask_label,
            edge_mask_label=edge_mask_label,

            coor_noise_bool=coor_noise_bool,
            coor_noise_true=coor_noise_true,
            flex_coor_mask=flex_coor_mask,                  # length_complex的mask，标记了ligand位置

            aff_true=aff_true,
            aff_mask=aff_mask,
            coor_mask=coor_mask,
            coor_scale=self.coor_scale,

            idx=f_name,


        )
        return complex_graph

    def __len__(self):
        return len(self.pdbbind_list)
#到这里dataset部分应该就结束了，complex_graph是字典(保存了张量数据),pdbbind_list(保存了复合物的文件名)






























#工具函数--->采样口袋
    def sample_pocket(self, protein_position_true, ref_l_coor, protein_pdb_info, CA_flag, max_len_protein):
        '''
        sample pocket atoms, len_protein + len_ligand < max_len_before_sampling
        :param protein_position_true: true coor of protein
        :param ref_l_coor: fpocket coor / true ligand coor
        :param protein_pdb_info: ChainId_ResidueNumber
        :param CA_flag: if is C-alpha atom
        :param max_len_protein: max atom allow for protein
        :return: index of selected protein atoms
        '''                                                                             # 这两个是外部定义的
        assert self.select_center_type in ['any_atom', 'geo_center']                    # for ref_l_coor  any_atom
        assert self.select_pocket_type in ['any_atom', 'CA']                            # for protein， CA   第一次提取的数据是针对all_atom的，现在主要是针对CA原子进行的

        if self.select_center_type == 'geo_center':
            ref_l_coor = ref_l_coor.mean(axis=0, keepdim=True)                          # 求配体中心坐标

        p2l_dismap = scipy.spatial.distance.cdist(protein_position_true, ref_l_coor, metric='euclidean').min(axis=-1)   # 初次构建pocket的时候好像计算的是整个ligand的距离画范围找的原子
        df_pocket = pd.DataFrame({'p2l_dismap': p2l_dismap, 'protein_pdb_info': protein_pdb_info, 'CA_flag': CA_flag})   # 构建p2l_dismap，protein_pdb_info，CA_flag的dataframe
                                                                                                                              # protein_pdb_info是一个这样的列表[A_45],表示原子在序列中的顺序
        # get residue distance----->df_sele_res(dataframe)                                                                    # 此时dismap的二维数据会变成一维
        if self.select_pocket_type == 'any_atom':
            df_tmp = df_pocket
        else:                                           # 使用这个
            df_tmp = df_pocket[(df_pocket['CA_flag'] == True)]                            # 挑选出来CA部分的数据，但是dataframe的行索引不会重新排序
        dic_sele_res = {'res': [], 'dis': []}
        for res_i in df_tmp['protein_pdb_info'].unique():
            df_sub = df_tmp[(df_tmp['protein_pdb_info'] == res_i)]                        # 取属于同一个氨基酸的原子构建dataframe
            dic_sele_res['res'].append(res_i)
            dic_sele_res['dis'].append(df_sub['p2l_dismap'].min())                        # 记录氨基酸名字，并将离中心最最近的CA原子距离作为aa的距离dic_sele_res = {'res': [], 'dis': []}
        df_sele_res = pd.DataFrame.from_dict(dic_sele_res)                           # 构建datafram -->sele_res   这个表的元素数量应该是氨基酸数
        df_sele_res = df_sele_res.sort_values(by=['dis'], ascending=[True])               # 按照据距离进行升序排序

        # select protein atoms
        df_sele_res['resi_num_count'] = np.array(
            [len(df_pocket[df_pocket['protein_pdb_info'] == res_i]) for res_i in df_sele_res['res']])   # (resi,)一维数据(数量是resi个) 表示每个氨基酸所含原子数
        df_sele_res['resi_num_cum_count'] = np.cumsum(df_sele_res['resi_num_count'])    # 累加求和
        df_sele_res['sele'] = df_sele_res['resi_num_cum_count'] < max_len_protein         # 进行布尔值判断超出容纳范围的较远的原子节点(以CA排序resi，选择atoms在max_len_protein范围内)
        sele_resi = df_sele_res['res'][df_sele_res['sele']].values                        # 取出 max_len_protein范围内的resi  

        # for dropout
        if self.training:
            random_retain = 1 - np.random.uniform(low=0, high=self.dropout, size=1)       # dropout的保留比例
            sele_resi = np.random.choice(sele_resi, size=max(1, int(random_retain * len(sele_resi))), replace=False)
        assert len(sele_resi) > 0                                                         # 保留resi数---->sele_resi

        # get pocket
        df_pocket['sele_flag'] = [True if i in sele_resi else False for i in df_pocket['protein_pdb_info']]       # 将要采用的resi在pocket的dataframe中打标签
        sub_index = np.argwhere(df_pocket['sele_flag'].values == True).reshape(-1)      # 根据选定的resi在pocket层上进行裁剪，返回的是True的索引 
        return sub_index


#工具函数--->构建蛋白质等价原子对
    def init_protein_pretraining_label(self):
        self.p_res_label_list = np.array(['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                                          'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'])
        self.p_atom_label_list = np.array(
            [' OD2', ' OE1', ' CD1', ' NE1', ' CB ', ' CZ ', ' CH2', ' SG ', ' CG ', ' CZ2',
             ' N  ', ' OG ', ' O  ', ' SD ', ' NE2', ' CE2', ' NZ ', ' OH ', ' NE ', ' CE ',
             ' CD2', ' ND2', ' OXT', ' CG2', ' C  ', ' CE1', ' CD ', ' OG1', ' CZ3', ' NH2',
             ' OE2', ' ND1', ' OD1', ' CE3', ' CA ', ' NH1', ' CG1'])
        self.p_atom_label_equ_list = [['VAL', ' CG1', ' CG2'],                                      # 等价原子对
                                      ['LEU', ' CD1', ' CD2'],
                                      ['PHE', ' CD1', ' CD2'],
                                      ['PHE', ' CE1', ' CE2'],
                                      ['TYR', ' CD1', ' CD2'],
                                      ['TYR', ' CE1', ' CE2']]  # only side chains without electrical charges
        self.fix_p_atom_label_list = []
        for res, a1, a2 in self.p_atom_label_equ_list:  # map a1 to a2                              # 这里相当于是进行一个解包操作
            tmp = [torch.from_numpy(self.p_res_label_list == res).float(),                  # 这里构建one_hot是通过numpy实现一一比对的
                   torch.from_numpy(self.p_atom_label_list == a1).float(),
                   torch.from_numpy(self.p_atom_label_list == a2).float()]
            self.fix_p_atom_label_list.append(tmp)                                                  # tmp----->[(20,)(37,)(37,)]------->fix_p_atom_label_list相当于是存了6个等价原子对




#工具函数---->构建mask矩阵----->随机掩蔽掉部分特征达到一定的自监督性
    def gen_mask_index(self, feat_label_list, mask_rate=0.15):
        '''
        generate mask location
        :param feat_label_list: list of feature tensor, [feat_1, feat_2, ...]
        :return: index , shape = feat.size(0)
        '''
        allow_mask_pos = torch.cat([
            feat_label.sum(dim=-1, keepdim=True) == 1 for feat_label in feat_label_list             # protein_node_feature_init[:, -37:] feat_label_list是一个范围
        ], dim=-1).prod(dim=-1)                                                                     # (nums, 1)  这里实际上是将所有特征进行一个逻辑与运算，只有全部为1的位置才会被选中
        origin_shape = allow_mask_pos.shape
        origin_shape_flat = origin_shape.numel()
        n_mask = max(int(mask_rate * origin_shape_flat), 1)
        mask_index = torch.randperm(origin_shape_flat)[:n_mask]
        mask = torch.zeros(origin_shape_flat).index_fill_(-1, mask_index, 1)
        mask = mask.reshape(origin_shape).bool()
        return mask                                                                                 # (nums,1)  ---> 其中mask_index的部分被填充为1


def assign_struct(mol, coor, min=True):
    AllChem.EmbedMolecule(mol, maxAttempts=10, useRandomCoords=True, clearConfs=False)
    mol_conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        mol_conf.SetAtomPosition(i, coor[i].detach().cpu().numpy().astype(float))
    if min:
        ff = Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(
            mol, Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(mol))
        for atom_i in range(mol.GetNumAtoms()):
            ff.MMFFAddPositionConstraint(atom_i, 1, 100)  # maxDispl: maximum displacement
        ff.Minimize(maxIts=20)
    return mol


@torch.no_grad()
def calc_rmsd(coor_pred, coor_true, match=None):
    if isinstance(match, type(None)):
        match = torch.arange(len(coor_true))
    n_atom = coor_true.size(-2)
    n_match = len(match) // n_atom
    nomatch = repeat(torch.arange(0, coor_true.size(-2)), 'n -> (m n)', m=n_match)

    coor_pred = rearrange(rearrange(coor_pred, 'e n c -> n e c')[match], '(m n) e c -> m n e c', m=n_match)
    coor_true = rearrange(rearrange(coor_true, 'e n c -> n e c')[nomatch], '(m n) e c -> m n e c', m=n_match)

    coor_loss = torch.einsum('m n e c -> m e', (coor_pred - coor_true)**2)
    rmsd_loss = (coor_loss / n_atom)**0.5

    return rmsd_loss, coor_pred, n_match


@torch.no_grad()
def pred_ens(coor_pred, dic_data):
    ens = coor_pred.shape[0]

    coor_pred = rearrange(coor_pred, 'b n c -> (b n) c')[dic_data.ligand_node_loc_after_sampling_flat].reshape(ens, -1, 3)  # to (ens, n_atom, 3)
    ligand_match = dic_data.ligand_match.reshape(ens, -1)[0]

    if ens > 1:
        ens_pred = coor_pred[0]
        first_pred = coor_pred[0]

        rest_pred = coor_pred[1:]

        rmsd_match_ens, tmp_pred, n_match = calc_rmsd(rest_pred,
                                                      repeat(first_pred, 'n c -> e n c', e=rest_pred.size(0)),
                                                      match=ligand_match)  # return [match, ens]
        min_index = rmsd_match_ens.min(dim=0, keepdims=True)[1]
        rest_ens_matched_pred = torch.gather(tmp_pred, dim=0,
                                             index=repeat(min_index, 'm e -> m n e c', n=rest_pred.size(1),
                                                          c=3)).squeeze(0)  # to [n_atom, ens-1, 3]

        ens_pred = torch.cat([first_pred.unsqueeze(1), rest_ens_matched_pred], dim=1).mean(dim=1)
    else:
        ens_pred = coor_pred[0]

    return ens_pred


class ComplexScreeningDataset(torch.utils.data.Dataset):
    def __init__(self, mode, args, data_list, cache_path='./cache', ens=1, specific_list=None):
        self.mode = mode
        self.pdbbind_id_list = [i[:4] for i in data_list]

        # pdbbind data source
        self.pdbbind_path = args.pdbbind_path
        self.allow_dict = pickle.load(open(args.allow_dict_path, 'rb'))

        # model hyperparameters
        self.n_cycle = args.n_cycle
        self.coor_scale = args.coor_scale
        self.aff_scale = args.aff_scale
        self.max_len_before_sampling = args.max_len_before_sampling
        self.max_len_after_sampling = args.max_len_after_sampling if mode == 'train' else args.max_len_after_sampling_for_eval
        self.max_len_ligand = args.max_len_ligand
        self.max_ligand_atom_init_distance = args.max_ligand_atom_init_distance
        self.max_ligand_atom_pretrain_distance = args.max_ligand_atom_pretrain_distance

        # Sample pocket
        self.sample_pocket_flag = args.sample_pocket_flag
        self.select_pocket_type = args.select_pocket_type
        self.select_center_type = args.select_center_type

        # for pocket dropout
        self.training = False
        self.dropout = args.dropout
        self.epoch = 0

        self.ens = ens

        if specific_list is not None:
            self.pdbbind_id_list = specific_list

    def __getitem__(self, i):
        idx = self.pdbbind_id_list[i]
        if self.ens > 1:
            complex_graph = []
            for e in range(self.ens):
                complex_graph.append(self.get_complex(idx))
            complex_graph = collate_screening(complex_graph)  # use collate_dummy in loader
        else:
            complex_graph = self.get_complex(idx)
        return complex_graph

    def get_complex(self, idx):
        if isinstance(idx, tuple):
            # for select pair, (l_idx, r_idx)
            idx, rand_idx = idx  # ligand, protein
        else:
            rand_idx = None

        ################################################################################################################
        # load prepared data
        ################################################################################################################
        dic_data = np.load(f'{self.pdbbind_path}/{idx}.npz')

        ligand_node_feature_init = dic_data['ligand_node_features']
        ligand_edge_feature_init = dic_data['ligand_edge_features']
        ligand_position_true = dic_data['ligand_true_posi']
        ligand_distmap = dic_data['ligand_distmap']
        ligand_match = dic_data['ligand_match']
        aff_true = dic_data['aff']

        screening_label = random.choice([1, 0])
        if rand_idx is None:
            allow_list = copy.deepcopy(self.allow_dict[idx])
            for i in allow_list:
                if i not in self.pdbbind_id_list:
                    allow_list.remove(i)
            if screening_label == 0:
                for _ in range(100):
                    rand_idx = random.choice(self.pdbbind_id_list)
                    if rand_idx not in allow_list:
                        break
            elif screening_label == 1:
                allow_list.remove(idx)
                rand_idx = random.choice(allow_list) if random.choice([True, False]) and len(allow_list) > 0 else idx

        coor_mask = aff_mask = 1 if rand_idx == idx else 0

        dic_data = np.load(f'{self.pdbbind_path}/{rand_idx}.npz')
        protein_node_feature_init = dic_data['protein_node_features']
        protein_edge_feature_init = dic_data['protein_edge_features']
        protein_position_true = dic_data['protein_true_posi']
        protein_pdb_info = dic_data['protein_pdb_info']
        ref_l_coor = dic_data['ligand_true_posi']


        ################################################################################################################
        # sample pocket
        ################################################################################################################
        if self.sample_pocket_flag:
            CA_flag = protein_node_feature_init[:, -3] == 1
            max_len_protein = self.max_len_before_sampling - len(ligand_node_feature_init)
            pocket_sub_index = self.sample_pocket(protein_position_true, ref_l_coor,
                                                  protein_pdb_info, CA_flag, max_len_protein)
            protein_node_feature_init = protein_node_feature_init[pocket_sub_index]
            protein_edge_feature_init = protein_edge_feature_init[pocket_sub_index, :][:, pocket_sub_index]
            protein_position_true = protein_position_true[pocket_sub_index]


        ################################################################################################################
        # get length
        ################################################################################################################
        len_protein_before_sampling = len(protein_node_feature_init)
        len_ligand = len(ligand_node_feature_init)
        len_complex_before_sampling = len_protein_before_sampling + len_ligand


        ################################################################################################################
        # scale coor / distance
        ################################################################################################################
        protein_position_true = protein_position_true / self.coor_scale
        ligand_position_true = ligand_position_true / self.coor_scale
        ref_l_coor = ref_l_coor / self.coor_scale
        ligand_distmap = ligand_distmap / self.coor_scale
        
        aff_true = math.log(aff_true)
        aff_true = aff_true / self.aff_scale


        ################################################################################################################
        # sampling nodes: core protein atom (CA,CB) retain, random sampling rest protein atom
        ################################################################################################################
        # get core atoms
        assert len_ligand < self.max_len_after_sampling

        # for CA CB
        core_atom_list = [-3, -33]  # CA -3, CB -33
        protien_core_atom_loc = [np.argwhere(protein_node_feature_init[:, i] == 1) for i in core_atom_list]
        protien_core_atom_loc = np.concatenate([x.reshape(-1) for x in protien_core_atom_loc], axis=-1)
        if len(protien_core_atom_loc) > self.max_len_after_sampling - len_ligand:
            core_atom_list = [-3]  # CA -3, CB -33
            protien_core_atom_loc = [np.argwhere(protein_node_feature_init[:, i] == 1) for i in core_atom_list]
            protien_core_atom_loc = np.concatenate([x.reshape(-1) for x in protien_core_atom_loc], axis=-1)
        if len(protien_core_atom_loc) > self.max_len_after_sampling - len_ligand:
            protien_core_atom_loc = protien_core_atom_loc[:self.max_len_after_sampling - len_ligand]

        # sampling rest atoms
        rest_loc = np.delete(np.arange(len_protein_before_sampling), protien_core_atom_loc)
        rest_sampling_num = min(self.max_len_after_sampling - len_ligand - len(protien_core_atom_loc),
                                len_protein_before_sampling - len(protien_core_atom_loc))
        rest_atom_loc = [np.random.choice(rest_loc, size=rest_sampling_num, replace=False)
                         for _ in range(self.n_cycle)]
        node_sampling_loc_list = [np.concatenate([
            protien_core_atom_loc,
            x,
            np.arange(len_ligand) + len_protein_before_sampling], axis=-1)
            for x in rest_atom_loc]
        node_sampling_loc = np.stack(node_sampling_loc_list, axis=0)
        len_complex_after_sampling = node_sampling_loc.shape[1]

        # cat edge_feature for protein and ligand
        edge_feature_1 = np.concatenate(
            (
                protein_edge_feature_init,
                np.zeros((protein_edge_feature_init.shape[0], len_ligand, protein_edge_feature_init.shape[-1]))
             ),
            axis=1)
        edge_feature_2 = np.concatenate(
            (
                np.zeros((ligand_edge_feature_init.shape[0], len_protein_before_sampling, ligand_edge_feature_init.shape[-1])),
                ligand_edge_feature_init
            ),
            axis=1)
        edge_feature_init = np.concatenate((edge_feature_1, edge_feature_2), axis=0)


        ################################################################################################################
        # initialize ligand coor
        ################################################################################################################
        pocket_center = ref_l_coor.mean(axis=0)

        ligand_position_init = pocket_center + \
                               np.random.randn(len_ligand, 3) * self.max_ligand_atom_init_distance / self.coor_scale
        protein_position_init = protein_position_true

        coor_init = np.concatenate((protein_position_init, ligand_position_init), axis=0)
        coor_true = np.concatenate((protein_position_true, ligand_position_true), axis=0)


        ################################################################################################################
        # supply info
        ################################################################################################################
        # for recycle info, update core atoms, omit rest sampled atoms
        node_cycling_loc = np.concatenate([np.arange(len(protien_core_atom_loc)),       # 这个实际上是没有永奥，这个可以用于构建采样后的mask
                                           np.arange(len_complex_after_sampling - len_ligand,
                                                     len_complex_after_sampling)], axis=-1)
        node_cycling_mask = np.zeros(len_complex_after_sampling)
        node_cycling_mask[:len(protien_core_atom_loc)] = 1
        node_cycling_mask[-len_ligand:] = 1
        edge_cycling_mask = np.zeros((len_complex_after_sampling, len_complex_after_sampling))
        edge_cycling_mask[:len(protien_core_atom_loc), :len(protien_core_atom_loc)] = 1
        edge_cycling_mask[-len_ligand:, -len_ligand:] = 1

        ligand_mask_after_sampling = np.zeros(len_complex_after_sampling)
        ligand_mask_after_sampling[np.arange(len_complex_after_sampling - len_ligand, len_complex_after_sampling)] = 1
        ligand_node_loc_after_sampling = np.arange(len_complex_after_sampling - len_ligand, len_complex_after_sampling)
        ligand_node_loc_before_sampling = np.arange(len_complex_before_sampling - len_ligand,
                                                    len_complex_before_sampling)
        protein_node_loc_before_sampling = np.arange(len_protein_before_sampling)

        ligand_match = ligand_match.reshape(-1)
        n_match = len(ligand_match) // len_ligand
        ligand_nomatch = repeat(torch.arange(0, len_ligand), 'm -> (n m)', n=n_match)


        ################################################################################################################
        # to tensor
        ################################################################################################################
        protein_node_feature_init = torch.from_numpy(protein_node_feature_init).float()
        ligand_node_feature_init = torch.from_numpy(ligand_node_feature_init).float()
        edge_feature_init = torch.from_numpy(edge_feature_init).float()
        coor_init = torch.from_numpy(coor_init).float()
        coor_true = torch.from_numpy(coor_true).float()
        aff_true = torch.Tensor([aff_true]).float()
        protein_position_true = torch.from_numpy(protein_position_true).float()
        ligand_distmap = torch.from_numpy(ligand_distmap).float()

        node_sampling_loc = torch.from_numpy(node_sampling_loc).long()
        node_cycling_loc = torch.from_numpy(node_cycling_loc).long()
        node_cycling_mask = torch.from_numpy(node_cycling_mask).long()
        edge_cycling_mask = torch.from_numpy(edge_cycling_mask).long()
        ligand_mask_after_sampling = torch.from_numpy(ligand_mask_after_sampling).long()
        ligand_node_loc_after_sampling = torch.from_numpy(ligand_node_loc_after_sampling).long()
        protein_node_loc_before_sampling = torch.from_numpy(protein_node_loc_before_sampling).long()
        ligand_node_loc_before_sampling = torch.from_numpy(ligand_node_loc_before_sampling).long()
        ligand_match = torch.from_numpy(ligand_match).long()


        ################################################################################################################
        # feature masking
        ################################################################################################################
        # protein node feature mask
        p_atom_label = protein_node_feature_init[:, -37:]         #蛋白质所有原子种类的one-hot编码
        p_res_label = protein_node_feature_init[:, -57:-37]       #氨基酸所有种类的one-hot编码
        p_x_mask_bool = self.gen_mask_index([p_atom_label, p_res_label], 0.15)   #对protein-atom和residue进行掩码表，此处是(num_atom, num_res)这种样子的
        p_x_mask_label_1 = p_atom_label.argmax(dim=-1)            #返回每个原子种类的索引 （num_atoms）内部表示的数据是原子类别编号--->0,1,2,3...
        p_x_mask_label_2 = p_res_label.argmax(dim=-1)             #返回每个氨基酸种类的索引

        # ligand node feature mask
        l_atom_label = ligand_node_feature_init[:, :10]           #ligand种类的one-hot编码
        l_x_mask_bool = self.gen_mask_index([l_atom_label], 0.15) #ligand的atom掩码表
        l_x_mask_label = l_atom_label.argmax(dim=-1)              #返回每个原子种类的索引

        # complex edge feature mask
        p_edge_label = torch.from_numpy(protein_edge_feature_init)
        p_edge_mask_bool = self.gen_mask_index([p_edge_label], 0.15)
        p_edge_mask_bool = (
                    p_edge_mask_bool.triu().float() + p_edge_mask_bool.float().triu().transpose(1, 0)
        ).bool()

        l_edge_label = torch.from_numpy(ligand_edge_feature_init)
        l_edge_mask_bool = self.gen_mask_index([l_edge_label], 0.15)
        l_edge_mask_bool = (
                    l_edge_mask_bool.triu().float() + l_edge_mask_bool.float().triu().transpose(1, 0)
        ).bool()

        edge_mask_bool = torch.cat([
            F.pad(p_edge_mask_bool, (0, len_ligand), 'constant', False),
            F.pad(l_edge_mask_bool, (len_protein_before_sampling, 0), 'constant', False)
        ], dim=0).bool()
        edge_mask_label = edge_feature_init.argmax(dim=-1)

        # mask origin feat -----> 在最后一维补一个特征，填充值为0作为mask_token通道
        protein_node_feature_init = F.pad(protein_node_feature_init, (0, 1), 'constant', 0)
        ligand_node_feature_init = F.pad(ligand_node_feature_init, (0, 1), 'constant', 0)
        edge_feature_init = F.pad(edge_feature_init, (0, 1), 'constant', 0)

        # pad mask to complex length ------>这部分构建的label是为了后面计算loss的就是实际上的类别标签
        p_x_mask_bool = F.pad(p_x_mask_bool, (0, len_ligand), 'constant', False)
        l_x_mask_bool = F.pad(l_x_mask_bool, (len_protein_before_sampling, 0), 'constant', False)
        p_x_mask_label_1 = F.pad(p_x_mask_label_1, (0, len_ligand), 'constant', 0)
        p_x_mask_label_2 = F.pad(p_x_mask_label_2, (0, len_ligand), 'constant', 0)
        l_x_mask_label = F.pad(l_x_mask_label, (len_protein_before_sampling, 0), 'constant', 0)


        ################################################################################################################
        # protein noise
        ################################################################################################################
        p_coor_true = protein_position_true
        p_coor_noise_bool = self.gen_mask_index([p_res_label], 0.15)

        coor_noise_bool = F.pad(p_coor_noise_bool, (0, len_ligand), 'constant', False)
        coor_noise_true = F.pad(p_coor_true, (0, 0, 0, len_ligand), 'constant', 0)

        # gen noise for coor init       # 采样前ligand的位置
        flex_coor_mask = F.pad(torch.zeros(len_protein_before_sampling), (0, len_ligand), 'constant', 1).to(torch.float)

        # for dist input
        pocket_dismap = (
                coor_init[:len_protein_before_sampling].unsqueeze(1) -
                coor_init[:len_protein_before_sampling].unsqueeze(0)
        ).norm(p=2, dim=-1)
        edge_feature_init = F.pad(edge_feature_init, (0, 1), 'constant', -1)
        edge_feature_init[:len_protein_before_sampling, :len_protein_before_sampling, -1] = pocket_dismap
        edge_feature_init[-len_ligand:, -len_ligand:, -1] = ligand_distmap


        ################################################################################################################
        # use Data in PyG
        ################################################################################################################
        complex_graph = dict(
            protein_node_feature_init=protein_node_feature_init,
            ligand_node_feature_init=ligand_node_feature_init,
            edge_feature_init=edge_feature_init,

            coor_init=coor_init,
            coor_true=coor_true,

            node_sampling_loc=node_sampling_loc,          #这里就进行了循环采样的步骤后节点的索引
            node_cycling_loc=node_cycling_loc,
            node_cycling_mask=node_cycling_mask,          #这个mask将core atom和ligand部分标记为1
            edge_cycling_mask=edge_cycling_mask,

            ligand_mask_after_sampling=ligand_mask_after_sampling,
            ligand_node_loc_after_sampling=ligand_node_loc_after_sampling,
            ligand_node_loc_before_sampling=ligand_node_loc_before_sampling,
            protein_node_loc_before_sampling=protein_node_loc_before_sampling,
            len_protein_before_sampling=len_protein_before_sampling,

            len_complex_before_sampling=len_complex_before_sampling,
            len_complex_after_sampling=len_complex_after_sampling,

            ligand_match=ligand_match,
            ligand_nomatch=ligand_nomatch,
            len_ligand=len_ligand,

            p_x_mask_bool=p_x_mask_bool,
            l_x_mask_bool=l_x_mask_bool,
            edge_mask_bool=edge_mask_bool,
            p_x_mask_label_1=p_x_mask_label_1,
            p_x_mask_label_2=p_x_mask_label_2,
            l_x_mask_label=l_x_mask_label,
            edge_mask_label=edge_mask_label,

            coor_noise_bool=coor_noise_bool,
            coor_noise_true=coor_noise_true,
            flex_coor_mask=flex_coor_mask,

            aff_true=aff_true,
            aff_mask=aff_mask,
            coor_mask=coor_mask,
            coor_scale=self.coor_scale,

            screening_label=screening_label,

            idx=(idx, rand_idx),
        )
        return complex_graph





    def __len__(self):
        return len(self.pdbbind_id_list)

    def sample_pocket(self, protein_position_true, ref_l_coor, protein_pdb_info, CA_flag, max_len_protein):
        '''
        sample pocket atoms, len_protein + len_ligand < max_len_before_sampling
        :param protein_position_true: true coor of protein
        :param ref_l_coor: fpocket coor / true ligand coor
        :param protein_pdb_info: ChainId_ResidueNumber
        :param CA_flag: if is C-alpha atom
        :param max_len_protein: max atom allow for protein
        :return: index of selected protein atoms
        '''
        assert self.select_center_type in ['any_atom', 'geo_center']  # for ref_l_coor
        assert self.select_pocket_type in ['any_atom', 'CA']  # for protein

        if self.select_center_type == 'geo_center':
            ref_l_coor = ref_l_coor.mean(axis=0, keepdim=True)

        p2l_dismap = scipy.spatial.distance.cdist(protein_position_true, ref_l_coor, metric='euclidean').min(axis=-1)
        df_pocket = pd.DataFrame({'p2l_dismap': p2l_dismap, 'protein_pdb_info': protein_pdb_info, 'CA_flag': CA_flag})

        # get residue distance
        if self.select_pocket_type == 'any_atom':
            df_tmp = df_pocket
        else:
            df_tmp = df_pocket[(df_pocket['CA_flag'] == True)]
        dic_sele_res = {'res': [], 'dis': []}
        for res_i in df_tmp['protein_pdb_info'].unique():
            df_sub = df_tmp[(df_tmp['protein_pdb_info'] == res_i)]
            dic_sele_res['res'].append(res_i)
            dic_sele_res['dis'].append(df_sub['p2l_dismap'].min())
        df_sele_res = pd.DataFrame.from_dict(dic_sele_res)
        df_sele_res = df_sele_res.sort_values(by=['dis'], ascending=[True])

        # select protein atoms
        df_sele_res['resi_num_count'] = np.array(
            [len(df_pocket[df_pocket['protein_pdb_info'] == res_i]) for res_i in df_sele_res['res']])
        df_sele_res['resi_num_cum_count'] = np.cumsum(df_sele_res['resi_num_count'])
        df_sele_res['sele'] = df_sele_res['resi_num_cum_count'] < max_len_protein
        sele_resi = df_sele_res['res'][df_sele_res['sele']].values

        # for dropout
        if self.training:
            random_retain = 1 - np.random.uniform(low=0, high=self.dropout, size=1)
            sele_resi = np.random.choice(sele_resi, size=max(1, int(random_retain * len(sele_resi))), replace=False)
        assert len(sele_resi) > 0

        # get pocket
        df_pocket['sele_flag'] = [True if i in sele_resi else False for i in df_pocket['protein_pdb_info']]
        sub_index = np.argwhere(df_pocket['sele_flag'].values == True).reshape(-1)
        return sub_index



#gen_mask_index感觉就是针对某种特征进行掩码 feat_label_list就是针对的特征部分，[p_res_label]这个就是针对蛋白质res类型部分

    def gen_mask_index(self, feat_label_list, mask_rate=0.15):
        '''
        generate mask location
        :param feat_label_list: list of feature tensor, [feat_1, feat_2, ...]
        :return: index , shape = feat.size(0)
        '''
        allow_mask_pos = torch.cat([
            feat_label.sum(dim=-1, keepdim=True) == 1 for feat_label in feat_label_list         #feat_label.sum(dim=-1, keepdim=True) == 1 挑选出来符合onehot的
        ], dim=-1).prod(dim=-1)                                                                 #挑选出符合onehot的位置(atom_num,)
        origin_shape = allow_mask_pos.shape
        origin_shape_flat = origin_shape.numel()                                                #总节点数(可以进行mask的节点数)                      
        n_mask = max(int(mask_rate * origin_shape_flat), 1)                                     #总mask数量这里是15%
        mask_index = torch.randperm(origin_shape_flat)[:n_mask]                                 #生成一个随机排列的索引       
        mask = torch.zeros(origin_shape_flat).index_fill_(-1, mask_index, 1)                    #这里是按照索引对前面的序列填充为1，构建了一个mask表
        mask = mask.reshape(origin_shape).bool()                                                #再reshape成原本的节点数      
        return mask                                                                             #最终返回的就是对某些原子进行mas的bool值，(num_atoms,)形状是这样的
                                                                                                #布尔序列就相当于是一个 原子编号的掩码标记。

def collate_screening(batch_list):
    dic_data = collate_struct(batch_list)
    dic_data.screening_label = torch.Tensor([g['screening_label'] for g in batch_list]).float()
    return dic_data

def collate_dummy(batch_list):
    return batch_list[0]


def gen_small_dataset(pdbbind_path, pdb_list_path, f_pkl_path, output_path, max_l=64):
    # train_list, val_list, test_list = load_data_split(path=pdb_list_path, blind_training=False)
    # train_list = [i for i in train_list if os.path.exists(os.path.join(pdbbind_path, i))]       # 过滤掉pdb_list中不存在的样本
    # val_list   = [i for i in val_list   if os.path.exists(os.path.join(pdbbind_path, i))]
    pdb_list = os.listdir(pdbbind_path)

    for i in tqdm(pdb_list):
        dic_data = np.load(f'{pdbbind_path}/{i}')
        # len_p = len(dic_data['protein_node_features'])
        ligand_true_posi = dic_data['ligand_true_posi']
        # if len_l >= max_l:
        #     pdbbind_path.remove(i)
        print(f'{ligand_true_posi}节点pos')
    
    for i in tqdm(pdb_list):
        dic_data = np.load(f'{pdbbind_path}/{i}')
        # len_p = len(dic_data['protein_node_features'])
        len_l = len(dic_data['ligand_node_features'])
        # if len_l >= max_l:
        #     pdb_list.remove(i)
        print(f'{len_l}节点长度')
    
    for i in tqdm(pdb_list):
        
        with open(f'{f_pkl_path}/{i}', 'rb') as f:
            fragment_data = np.load(f, allow_pickle=True)  # 解包 tuple
            dict_frag = fragment_data['frag'].item()
        # 如果 fragment_data 是一个 dict，并且里面有 'fragment_pos'
        # 比如 jt.fragment_pos() 返回的是 {'fragment_pos': xxx, ...}
        fragment_src_pos = dict_frag['fragment_src_pos']
        print(f'{fragment_src_pos} fragment_src')


    # print(f'Remain: train-{len(train_list)}, val-{len(val_list)}')
    # save_data_split(train_list, val_list, test_list, path=output_path)




if __name__ == '__main__':
    gen_small_dataset('/home/lpw/ligpose_learn/test_data/pdbbind_path',
                      '/home/lpw/ligpose_learn/test_data/train_list.txt',
                      '/home/lpw/ligpose_learn/test_data/f_pkl_path',
                      output_path='/home/lpw/ligpose_learn/test_data/output',
                      
                      max_l=100)

    pass


















