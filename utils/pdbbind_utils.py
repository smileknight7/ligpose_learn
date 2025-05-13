import os
import shutil
import sys

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

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import torch_geometric

from utils.common import load_idx_list
from utils.data_utils import pad_zeros, batch_index_select_for_edge, batch_index_select
from utils.pdbbind_preprocess import gen_pdbbind_screening_list
from utils.training_utils import load_data_split, save_data_split



# def split_pdbbind(pdbbind_path, data_split_rate, core_list_path=None):
#     if isinstance(core_list_path, type(None)):
#         pdb_list = os.listdir(pdbbind_path)
#         random.shuffle(pdb_list)
#
#         l = len(pdb_list)
#         cut_1 = int(data_split_rate[0] * l)
#         cut_2 = cut_1 + int(data_split_rate[1] * l)
#         train_list = pdb_list[:cut_1]
#         val_list = pdb_list[cut_1:cut_2]
#         test_list = pdb_list[cut_2:]
#     else:
#         pdb_list = os.listdir(pdbbind_path)
#         test_list = [f'{i}.npz' for i in load_idx_list(core_list_path) if f'{i}.npz' in pdb_list]
#         rest_list = [i for i in pdb_list if i not in test_list]
#         random.shuffle(rest_list)
#
#         l = len(rest_list)
#         cut_2 = int(data_split_rate[1] * l)
#         val_list = rest_list[:cut_2]
#         train_list = rest_list[cut_2:]
#
#     return train_list, val_list, test_list


#################################################
########-----------划分数据集------------##########
#################################################

def split_pdbbind_semi(split_rate, pdbbind_path=None, core_list_path=None, labeled_set_path=None, unlabeled_set_path=None):
    """
    划分数据集为有标签和无标签部分
    """
    # 加载有标签数据（refined_set）
    labeled_complexes = []
    if labeled_set_path is not None:
        if os.path.isfile(labeled_set_path):
            # 如果是文件，读取文件内容
            labeled_complexes = [l.strip() for l in open(labeled_set_path, 'r').readlines()]
        else:
            # 如果是目录，获取所有子目录
            labeled_complexes = [os.path.basename(f).replace('.npz', '')
                                 for f in glob.glob(os.path.join(labeled_set_path, '*'))
                                 if os.path.isdir(f) or f.endswith('.npz')]
        print(f"加载有标签数据: {len(labeled_complexes)} 个复合物")

    # 加载无标签数据（general_set）
    unlabeled_complexes = []
    if unlabeled_set_path is not None:
        if os.path.isfile(unlabeled_set_path):
            # 如果是文件，读取文件内容
            unlabeled_complexes = [l.strip() for l in open(unlabeled_set_path, 'r').readlines()]
        else:
            # 如果是目录，获取所有子目录
            unlabeled_complexes = [os.path.basename(f).replace('.npz', '')
                                 for f in glob.glob(os.path.join(unlabeled_set_path, '*'))
                                 if os.path.isdir(f) or f.endswith('.npz')]
        print(f"加载无标签数据: {len(unlabeled_complexes)} 个复合物")

    # 保留核心集测试
    core_list = []
    if core_list_path is not None:
        if os.path.isdir(core_list_path):
            core_list = [os.path.basename(f) for f in glob.glob(os.path.join(core_list_path, '*'))
                         if os.path.isdir(f)]
        else:
            core_list = [l.strip() for l in open(core_list_path, 'r').readlines()]
        print(f"加载核心集数据: {len(core_list)} 个复合物")

    # 划分有标签数据为训练/验证/测试集
    test_list = core_list if core_list else []
    non_test_labeled = [c for c in labeled_complexes if c not in test_list]

    # 按比例划分
    train_size = int(len(non_test_labeled) * split_rate[0])
    val_size = int(len(non_test_labeled) * split_rate[1])

    train_labeled = non_test_labeled[:train_size]
    val_list = non_test_labeled[train_size:train_size + val_size]

    # 若测试集为空，则从剩余有标签数据中划分
    if not test_list:
        test_list = non_test_labeled[train_size + val_size:]

    # 将无标签数据和有标签数据分别标记并存储
    train_list = [(c, 1) for c in train_labeled]  # 1表示有标签
    train_list.extend([(c, 0) for c in unlabeled_complexes])  # 0表示无标签

    return train_list, val_list, test_list


class SemiSupervisedComplexDataset(ComplexDataset):
    def __init__(self, split, args, data_list, cache_path=None):
        """初始化半监督数据集

        Args:
            split: 数据集类型 'train', 'val', 'test'
            args: 参数
            data_list: 数据列表，格式为 [(pdb_id, has_label), ...]
            cache_path: 缓存路径
        """
        self.split = split
        self.args = args
        self.data_list = [item[0] if isinstance(item, tuple) else item for item in data_list]
        self.has_label = [item[1] if isinstance(item, tuple) else True for item in data_list]
        self.cache_path = cache_path
        self.training = split == 'train'

        # 初始化其他成员变量
        super(ComplexDataset, self).__init__()

    def __getitem__(self, idx):
        pdb_id = self.data_list[idx]
        has_label = self.has_label[idx]

        # 加载数据
        data = self._load_data(pdb_id)

        # 如果是无标签数据，将亲和力信息设为None或特殊值
        if not has_label:
            data['aff'] = None

        return data


def collate_struct(batch_list):
    # get max len
    max_len_complex_before_sampling = 0
    max_len_protein_before_sampling = 0
    max_len_ligand = 0
    max_len_complex_after_sampling = 0
    for g in batch_list:
        max_len_complex_before_sampling = max(max_len_complex_before_sampling, g['len_complex_before_sampling'])
        max_len_protein_before_sampling = max(max_len_protein_before_sampling, g['len_protein_before_sampling'])
        max_len_ligand = max(max_len_ligand, g['len_ligand'])
        max_len_complex_after_sampling = max(max_len_complex_after_sampling, g['len_complex_after_sampling'])
        g['node_mask_after_sampling'] = torch.ones(g['len_complex_after_sampling'])
        g['edge_mask_after_sampling'] = torch.ones(g['len_complex_after_sampling'], g['len_complex_after_sampling'])

    # feat & coor
    dic_data = {}
    dic_data.update(pad_zeros(batch_list,
                    [
                        'protein_node_feature_init',
                    ],
                    max_len_protein_before_sampling,
                    collect_dim=-3, data_type='1d', output_dtype=torch.float))
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

    # index padding for cycling
    # node_sampling_loc: [n_cycle, n_loc]
    # Warning: no paddings inside n_loc (i.e. between protein and ligand),
    # nodes have to be selected with idx_remove_middle_pad before
    dic_data['node_sampling_loc'] = torch.stack([
        F.pad(g['node_sampling_loc'],
              (0, max_len_complex_after_sampling - g['node_sampling_loc'].shape[1]),
              'constant',
              max_len_complex_before_sampling - 1)  # max_len_complex_before_sampling-1 used for padding sampling
        for g in batch_list], dim=1).long()  # to (n_cycle, n_batch, n_loc)

    # nodes: [protein, pad, ligand, pad] -> [protein, ligand, pad, pad]
    dic_data['idx_remove_middle_pad'] = torch.stack([
        torch.cat([
            torch.arange(0, g['len_protein_before_sampling']),
            torch.arange(max_len_protein_before_sampling, max_len_protein_before_sampling + g['len_ligand']),
            torch.arange(g['len_protein_before_sampling'], max_len_protein_before_sampling),
            torch.arange(max_len_protein_before_sampling + g['len_ligand'],
                         max_len_protein_before_sampling + max_len_ligand)
        ], dim=0) for g in batch_list], dim=0).long()

    dic_data.update(pad_zeros(batch_list,
                    [
                        'node_cycling_mask', 'ligand_mask_after_sampling',
                        'node_mask_after_sampling',
                    ],
                    max_len_complex_after_sampling,
                    collect_dim=-2, data_type='1d', output_dtype=torch.float))
    dic_data.update(pad_zeros(batch_list,
                    [
                        'edge_cycling_mask', 'edge_mask_after_sampling',
                    ],
                    max_len_complex_after_sampling,
                    collect_dim=-3, data_type='2d', output_dtype=torch.float))
    dic_data.update(pad_zeros(batch_list,
                    [
                        'ligand_node_loc_before_sampling', 'ligand_node_loc_after_sampling',
                    ],
                    max_len_ligand,
                    collect_dim=-2, data_type='1d', output_dtype=torch.long))
    dic_data.update(pad_zeros(batch_list,
                    [
                        'protein_node_loc_before_sampling',
                    ],
                    max_len_protein_before_sampling,
                    collect_dim=-2, data_type='1d', output_dtype=torch.long))

    # For ligand matching
    len_tmp_after_sampling = 0
    len_tmp_ligand = 0
    len_tmp_batch_match = 0
    ligand_node_loc_after_sampling = []
    ligand_match = []
    ligand_nomatch = []
    scatter_ligand_1 = []
    scatter_ligand_2 = []
    for i, g in enumerate(batch_list):
        ligand_node_loc_after_sampling.append(g['ligand_node_loc_after_sampling'] + len_tmp_after_sampling)
        ligand_match.append(g['ligand_match'] + len_tmp_ligand)
        ligand_nomatch.append(g['ligand_nomatch'] + len_tmp_ligand)
        scatter_ligand_1.append(repeat(torch.arange(0, len(g['ligand_match']) // g['len_ligand']),
                                       'i -> (i m)', m=g['len_ligand']) + len_tmp_batch_match)
        scatter_ligand_2.append(torch.zeros(len(g['ligand_match']) // g['len_ligand']) + i)
        len_tmp_after_sampling += max_len_complex_after_sampling
        len_tmp_ligand += g['len_ligand']
        len_tmp_batch_match += len(g['ligand_match']) // g['len_ligand']
    dic_data['ligand_node_loc_after_sampling_flat'] = torch.cat(ligand_node_loc_after_sampling, dim=0).long()
    dic_data['ligand_match'] = torch.cat(ligand_match, dim=0).long()
    dic_data['ligand_nomatch'] = torch.cat(ligand_nomatch, dim=0).long()
    dic_data['scatter_ligand_1'] = torch.cat(scatter_ligand_1, dim=0).long()
    dic_data['scatter_ligand_2'] = torch.cat(scatter_ligand_2, dim=0).long()


    # for suppl info
    dic_data['aff_true'] = torch.cat([g['aff_true'] for g in batch_list], dim=0).float()
    dic_data['aff_mask'] = torch.Tensor([g['aff_mask'] for g in batch_list]).float()
    dic_data['coor_mask'] = torch.Tensor([g['coor_mask'] for g in batch_list]).float()
    dic_data['len_ligand'] = torch.Tensor([g['len_ligand'] for g in batch_list]).float()
    dic_data['idx'] = [g['idx'] for g in batch_list]


    # pre sample edge
    # unlike nodes, no padding between protein and ligand: [protein, ligand, pad]
    dic_data['edge_feature_init_cycle'] = torch.stack([
        batch_index_select_for_edge(dic_data['edge_feature_init'], loc) for loc in dic_data['node_sampling_loc']
    ], dim=0).float()


    # for masking
    dic_data.update(pad_zeros(batch_list,
                    [
                        'p_x_mask_bool', 'l_x_mask_bool',
                    ],
                    max_len_complex_before_sampling,
                    collect_dim=-2, data_type='1d', value=False, output_dtype=torch.bool))
    dic_data.update(pad_zeros(batch_list,
                    [
                        'edge_mask_bool',
                    ],
                    max_len_complex_before_sampling,
                    collect_dim=-3, data_type='2d', value=False, output_dtype=torch.bool))
    dic_data.update(pad_zeros(batch_list,
                    [
                        'p_x_mask_label_1', 'p_x_mask_label_2', 'l_x_mask_label',
                    ],
                    max_len_complex_before_sampling,
                    collect_dim=-2, data_type='1d', output_dtype=torch.long))
    dic_data.update(pad_zeros(batch_list,
                    [
                        'edge_mask_label',
                    ],
                    max_len_complex_before_sampling,
                    collect_dim=-3, data_type='2d', output_dtype=torch.long))


    # for masking pre-sampling
    dic_data['p_x_mask_bool_cycle'] = torch.stack([
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


    # addidional batch info
    dic_data['x_batch_info'] = repeat(torch.arange(len(batch_list)), 'b -> c b n',
                                      c=dic_data['node_sampling_loc'].size(0),
                                      n=max_len_complex_before_sampling).long()
    dic_data['edge_batch_info'] = repeat(torch.arange(len(batch_list)), 'b -> c b i j',
                                         c=dic_data['node_sampling_loc'].size(0),
                                         i=max_len_complex_before_sampling,
                                         j=max_len_complex_before_sampling).long()
    dic_data['x_batch_info_cycle'] = torch.gather(dic_data['x_batch_info'], dim=-1, index=dic_data['node_sampling_loc']).long()
    edge_batch_info_cycle = torch.gather(dic_data['edge_batch_info'], dim=-1,
                                         index=repeat(dic_data['node_sampling_loc'], 'c b j -> c b i j',
                                                      i=max_len_complex_before_sampling))
    dic_data['edge_batch_info_cycle'] = torch.gather(edge_batch_info_cycle, dim=-2,
                                         index=repeat(dic_data['node_sampling_loc'], 'c b i -> c b i j',
                                                      j=edge_batch_info_cycle.size(-1))).long()


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
                        'flex_coor_mask',
                    ],
                    max_len_complex_before_sampling,
                    collect_dim=-2, data_type='1d', output_dtype=torch.float))
    dic_data['flex_coor_mask_cycle'] = torch.stack([
        torch.gather(dic_data['flex_coor_mask'], dim=-1, index=loc) for loc in dic_data['node_sampling_loc']
    ], dim=0).float()

    batch_data = torch_geometric.data.Data(**dic_data)
    return batch_data


class ComplexStructDataset(torch.utils.data.Dataset):
    def __init__(self, mode, args, data_list, cache_path='./cache'):
        self.mode = mode
        self.pdbbind_list = data_list

        # pdbbind data source
        self.pdbbind_path = args.pdbbind_path

        # unlabeled data source
        self.semi_list = [i.split('.')[0] for i in os.listdir(args.c_npz_path) if i.endswith('npz')]
        self.l_npz_path = args.l_npz_path
        self.p_npz_path = args.p_npz_path
        self.c_npz_path = args.c_npz_path

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

        # for masking
        self.semi_rate = args.semi_rate
        self.mask_rate_l = args.mask_rate_l
        self.mask_rate_p = args.mask_rate_p

        # for protein pretraining
        self.init_protein_pretraining_label()
        self.noise_distance = args.noise_distance

        # for pocket dropout
        self.training = False
        self.dropout = args.dropout
        self.epoch = 0


    def __getitem__(self, i):
        if self.mode == 'train':
            f_name = self.pdbbind_list[i] if torch.rand(1) > self.semi_rate else random.choice(self.semi_list)
            try:
                complex_graph = self.get_complex(f_name)
            except:
                for _ in range(5):
                    try:
                        complex_graph = self.get_complex(random.choice(self.pdbbind_list))
                        break
                    except:
                        raise RuntimeError('Error in dataloader')
        else:
            f_name = self.pdbbind_list[i]
            complex_graph = self.get_complex(f_name)
        return complex_graph

    def get_complex(self, f_name):
        ################################################################################################################
        # load prepared data
        ################################################################################################################
        if '-' in f_name:
            data_type = 'semi'
        else:
            data_type = 'pdbbind'

        if data_type == 'semi':
            complex_idx = f_name[:-4]
            protein_idx = complex_idx.split('-')[0]
            ligand_idx = complex_idx.split('-')[1]

            dic_pocket = np.load(f'{self.p_npz_path}/{protein_idx}.npz', allow_pickle=False)
            protein_node_feature_init = dic_pocket['protein_node_features']
            protein_edge_feature_init = dic_pocket['protein_edge_features']
            protein_position_true = dic_pocket['protein_true_posi']
            protein_pdb_info = dic_pocket['protein_pdb_info']

            dic_ligand = np.load(f'{self.l_npz_path}/{ligand_idx}.npz', allow_pickle=True)
            ligand_node_feature_init = dic_ligand['ligand_node_features']
            ligand_edge_feature_init = dic_ligand['ligand_edge_features']
            # smi = dic_ligand['ligand_smiles']
            ligand_distmap = dic_ligand['ligand_distmap']
            ligand_match = dic_ligand['ligand_match']
            ligand_position_true = np.zeros((len(ligand_node_feature_init), 3))

            aff_true = -1
            aff_mask = 0
            ref_l_coor = dic_pocket['center_coor']
            coor_mask = .0

        else:
            dic_data = np.load(f'{self.pdbbind_path}/{f_name}')

            protein_node_feature_init = dic_data['protein_node_features']
            protein_edge_feature_init = dic_data['protein_edge_features']
            protein_position_true = dic_data['protein_true_posi']
            protein_pdb_info = dic_data['protein_pdb_info']
            ligand_node_feature_init = dic_data['ligand_node_features']
            ligand_edge_feature_init = dic_data['ligand_edge_features']
            ligand_position_true = dic_data['ligand_true_posi']
            ligand_distmap = dic_data['ligand_distmap']
            ligand_match = dic_data['ligand_match']
            aff_true = dic_data['aff']

            coor_mask = 1
            aff_mask = 1
            ref_l_coor = ligand_position_true


        ################################################################################################################
        # sample pocket
        ################################################################################################################
        if self.sample_pocket_flag and data_type == 'pdbbind':
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
        node_cycling_loc = np.concatenate([np.arange(len(protien_core_atom_loc)),
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
        p_atom_label = protein_node_feature_init[:, -37:]
        p_res_label = protein_node_feature_init[:, -57:-37]
        for res, a1, a2 in self.fix_p_atom_label_list:
            equ_loc = ((p_res_label * res).sum(dim=-1) * (p_atom_label * a1).sum(dim=-1)) == 1
            p_atom_label[equ_loc] = repeat(a2, 'd -> n d', n=int(equ_loc.float().sum()))
        p_x_mask_bool = self.gen_mask_index([p_atom_label, p_res_label], self.mask_rate_p)
        p_x_mask_label_1 = p_atom_label.argmax(dim=-1)
        p_x_mask_label_2 = p_res_label.argmax(dim=-1)

        # ligand node feature mask
        l_atom_label = ligand_node_feature_init[:, :10]
        l_x_mask_bool = self.gen_mask_index([l_atom_label], self.mask_rate_l)
        l_x_mask_label = l_atom_label.argmax(dim=-1)

        # complex edge feature mask
        p_edge_label = torch.from_numpy(protein_edge_feature_init)
        p_edge_mask_bool = self.gen_mask_index([p_edge_label], self.mask_rate_p)
        p_edge_mask_bool = (
                    p_edge_mask_bool.triu().float() + p_edge_mask_bool.float().triu().transpose(1, 0)
        ).bool()

        l_edge_label = torch.from_numpy(ligand_edge_feature_init)
        l_edge_mask_bool = self.gen_mask_index([l_edge_label], self.mask_rate_l)
        l_edge_mask_bool = (
                    l_edge_mask_bool.triu().float() + l_edge_mask_bool.float().triu().transpose(1, 0)
        ).bool()

        edge_mask_bool = torch.cat([
            F.pad(p_edge_mask_bool, (0, len_ligand), 'constant', False),
            F.pad(l_edge_mask_bool, (len_protein_before_sampling, 0), 'constant', False)
        ], dim=0).bool()
        edge_mask_label = edge_feature_init.argmax(dim=-1)

        # mask origin feat
        if data_type == 'semi':
            protein_node_feature_init = F.pad(protein_node_feature_init, (0, 1), 'constant', 0)
            protein_node_feature_init[p_x_mask_bool] = 0
            protein_node_feature_init[p_x_mask_bool, -1] = 1
            ligand_node_feature_init = F.pad(ligand_node_feature_init, (0, 1), 'constant', 0)
            ligand_node_feature_init[l_x_mask_bool] = 0
            ligand_node_feature_init[l_x_mask_bool, -1] = 1

            edge_feature_init = rearrange(F.pad(edge_feature_init, (0, 1), 'constant', 0), 'i j d -> (i j) d')
            edge_feature_init[edge_mask_bool.reshape(-1)] = 0
            edge_feature_init[edge_mask_bool.reshape(-1), -1] = 1
            edge_feature_init = rearrange(edge_feature_init, '(i j) d -> i j d',
                                          i=len_protein_before_sampling + len_ligand)
        else:
            protein_node_feature_init = F.pad(protein_node_feature_init, (0, 1), 'constant', 0)
            ligand_node_feature_init = F.pad(ligand_node_feature_init, (0, 1), 'constant', 0)
            edge_feature_init = F.pad(edge_feature_init, (0, 1), 'constant', 0)

        # pad mask to complex length
        p_x_mask_bool = F.pad(p_x_mask_bool, (0, len_ligand), 'constant', False)
        l_x_mask_bool = F.pad(l_x_mask_bool, (len_protein_before_sampling, 0), 'constant', False)
        p_x_mask_label_1 = F.pad(p_x_mask_label_1, (0, len_ligand), 'constant', 0)
        p_x_mask_label_2 = F.pad(p_x_mask_label_2, (0, len_ligand), 'constant', 0)
        l_x_mask_label = F.pad(l_x_mask_label, (len_protein_before_sampling, 0), 'constant', 0)


        ################################################################################################################
        # protein noise
        ################################################################################################################
        p_coor_true = protein_position_true
        p_coor_noise_bool = self.gen_mask_index([p_res_label], self.mask_rate_p)

        coor_noise_bool = F.pad(p_coor_noise_bool, (0, len_ligand), 'constant', False)
        coor_noise_true = F.pad(p_coor_true, (0, 0, 0, len_ligand), 'constant', 0)


        # gen noise for coor init
        if data_type == 'semi':
            coor_init[coor_noise_bool] = coor_init[coor_noise_bool] + torch.randn(
                coor_init[coor_noise_bool].shape) * self.noise_distance / self.coor_scale
            flex_coor_mask = F.pad(p_coor_noise_bool, (0, len_ligand), 'constant', 1).float()
        else:
            flex_coor_mask = F.pad(torch.zeros(len_protein_before_sampling), (0, len_ligand), 'constant', 1).to(
                torch.float)

        # for dist input
        pocket_dismap = (
                coor_init[:len_protein_before_sampling].unsqueeze(1) -
                coor_init[:len_protein_before_sampling].unsqueeze(0)
        ).norm(p=2, dim=-1)
        edge_feature_init = F.pad(edge_feature_init, (0, 1), 'constant', -1)
        edge_feature_init[:len_protein_before_sampling, :len_protein_before_sampling, -1] = pocket_dismap
        edge_feature_init[-len_ligand:, -len_ligand:, -1] = ligand_distmap
        if data_type == 'semi':
            edge_feature_init[p_x_mask_bool, :, -1] = -1
            edge_feature_init[:, p_x_mask_bool, -1] = -1
            edge_feature_init[l_x_mask_bool, :, -1] = -1
            edge_feature_init[:, l_x_mask_bool, -1] = -1


        ################################################################################################################
        # use Data in PyG
        ################################################################################################################
        complex_graph = dict(
            protein_node_feature_init=protein_node_feature_init,
            ligand_node_feature_init=ligand_node_feature_init,
            edge_feature_init=edge_feature_init,

            coor_init=coor_init,
            coor_true=coor_true,

            node_sampling_loc=node_sampling_loc,
            node_cycling_loc=node_cycling_loc,
            node_cycling_mask=node_cycling_mask,
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

            idx=f_name,
        )
        return complex_graph

    def __len__(self):
        return len(self.pdbbind_list)

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

    def init_protein_pretraining_label(self):
        self.p_res_label_list = np.array(['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                                          'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'])
        self.p_atom_label_list = np.array(
            [' OD2', ' OE1', ' CD1', ' NE1', ' CB ', ' CZ ', ' CH2', ' SG ', ' CG ', ' CZ2',
             ' N  ', ' OG ', ' O  ', ' SD ', ' NE2', ' CE2', ' NZ ', ' OH ', ' NE ', ' CE ',
             ' CD2', ' ND2', ' OXT', ' CG2', ' C  ', ' CE1', ' CD ', ' OG1', ' CZ3', ' NH2',
             ' OE2', ' ND1', ' OD1', ' CE3', ' CA ', ' NH1', ' CG1'])
        self.p_atom_label_equ_list = [['VAL', ' CG1', ' CG2'],
                                      ['LEU', ' CD1', ' CD2'],
                                      ['PHE', ' CD1', ' CD2'],
                                      ['PHE', ' CE1', ' CE2'],
                                      ['TYR', ' CD1', ' CD2'],
                                      ['TYR', ' CE1', ' CE2']]  # only side chains without electrical charges
        self.fix_p_atom_label_list = []
        for res, a1, a2 in self.p_atom_label_equ_list:  # map a1 to a2
            tmp = [torch.from_numpy(self.p_res_label_list == res).float(),
                   torch.from_numpy(self.p_atom_label_list == a1).float(),
                   torch.from_numpy(self.p_atom_label_list == a2).float()]
            self.fix_p_atom_label_list.append(tmp)

    def gen_mask_index(self, feat_label_list, mask_rate=0.15):
        '''
        generate mask location
        :param feat_label_list: list of feature tensor, [feat_1, feat_2, ...]
        :return: index , shape = feat.size(0)
        '''
        allow_mask_pos = torch.cat([
            feat_label.sum(dim=-1, keepdim=True) == 1 for feat_label in feat_label_list
        ], dim=-1).prod(dim=-1)
        origin_shape = allow_mask_pos.shape
        origin_shape_flat = origin_shape.numel()
        n_mask = max(int(mask_rate * origin_shape_flat), 1)
        mask_index = torch.randperm(origin_shape_flat)[:n_mask]
        mask = torch.zeros(origin_shape_flat).index_fill_(-1, mask_index, 1)
        mask = mask.reshape(origin_shape).bool()
        return mask


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
        node_cycling_loc = np.concatenate([np.arange(len(protien_core_atom_loc)),
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
        p_atom_label = protein_node_feature_init[:, -37:]
        p_res_label = protein_node_feature_init[:, -57:-37]
        p_x_mask_bool = self.gen_mask_index([p_atom_label, p_res_label], 0.15)
        p_x_mask_label_1 = p_atom_label.argmax(dim=-1)
        p_x_mask_label_2 = p_res_label.argmax(dim=-1)

        # ligand node feature mask
        l_atom_label = ligand_node_feature_init[:, :10]
        l_x_mask_bool = self.gen_mask_index([l_atom_label], 0.15)
        l_x_mask_label = l_atom_label.argmax(dim=-1)

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

        # mask origin feat
        protein_node_feature_init = F.pad(protein_node_feature_init, (0, 1), 'constant', 0)
        ligand_node_feature_init = F.pad(ligand_node_feature_init, (0, 1), 'constant', 0)
        edge_feature_init = F.pad(edge_feature_init, (0, 1), 'constant', 0)

        # pad mask to complex length
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

        # gen noise for coor init
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

            node_sampling_loc=node_sampling_loc,
            node_cycling_loc=node_cycling_loc,
            node_cycling_mask=node_cycling_mask,
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

    def gen_mask_index(self, feat_label_list, mask_rate=0.15):
        '''
        generate mask location
        :param feat_label_list: list of feature tensor, [feat_1, feat_2, ...]
        :return: index , shape = feat.size(0)
        '''
        allow_mask_pos = torch.cat([
            feat_label.sum(dim=-1, keepdim=True) == 1 for feat_label in feat_label_list
        ], dim=-1).prod(dim=-1)
        origin_shape = allow_mask_pos.shape
        origin_shape_flat = origin_shape.numel()
        n_mask = max(int(mask_rate * origin_shape_flat), 1)
        mask_index = torch.randperm(origin_shape_flat)[:n_mask]
        mask = torch.zeros(origin_shape_flat).index_fill_(-1, mask_index, 1)
        mask = mask.reshape(origin_shape).bool()
        return mask


def collate_screening(batch_list):
    dic_data = collate_struct(batch_list)
    dic_data.screening_label = torch.Tensor([g['screening_label'] for g in batch_list]).float()
    return dic_data

def collate_dummy(batch_list):
    return batch_list[0]


def gen_small_dataset(pdbbind_path, pdb_list_path, output_path, max_l=64):
    train_list, val_list, test_list = load_data_split(path=pdb_list_path, blind_training=False)

    for i in tqdm(train_list):
        dic_data = np.load(f'{pdbbind_path}/{i}')
        # len_p = len(dic_data['protein_node_features'])
        len_l = len(dic_data['ligand_node_features'])
        if len_l >= max_l:
            train_list.remove(i)

    for i in tqdm(val_list):
        dic_data = np.load(f'{pdbbind_path}/{i}')
        # len_p = len(dic_data['protein_node_features'])
        len_l = len(dic_data['ligand_node_features'])
        if len_l >= max_l:
            val_list.remove(i)

    print(f'Remain: train-{len(train_list)}, val-{len(val_list)}')
    save_data_split(train_list, val_list, test_list, path=output_path)




if __name__ == '__main__':
    # gen_small_dataset('/home/dtj/work_site/test/tmp',
    #                   '../eval/pdbbind/core_test',
    #                   '../eval/pdbbind/reduced_core_test',
    #                   max_l=100)

    pass


















