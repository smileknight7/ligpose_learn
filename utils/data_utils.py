import numpy as np
from rdkit import Chem

import torch
import torch.nn.functional as F
from einops import rearrange, repeat



# 构建x的索引矩阵
def batch_index_select(x, idx):
    # select data with gather
    # x =  torch.stack([torch.index_select(x_i, 0, idx_i) for x_i, idx_i in zip(x, idx)], dim=0)
    assert len(x.shape) - len(idx.shape) == 1                               # 使用assert进行判断，维度是否统一
    idx = repeat(idx, 'b n -> b n d', d=x.size(-1))          # 这里是取x的最后一个维度进行复制到idx中以方便使用矩阵索引
    new_x = torch.gather(x, dim=-2, index=idx)                        # 根据索引找到x中倒数第二个维度中的内容
    return new_x                          


#这个函数是重稠密边张量中抽取子图（先按行筛选再按列筛选）
                                                                            # 分两种模式----->1有边特征的
                                                                            #          ----->2索引矩阵
                                                                            # 这个函数应用到这里了dic_data['edge_feature_init_cycle'] = torch.stack([
def batch_index_select_for_edge(x, idx, mask=False):              # x------>(B, N, N, D)
    # select data with gather, for edge (2D data)                           # idx------>(B, n_loc)
    if not mask:
        index = repeat(idx,
                       'b i -> b i j d',
                       j=x.size(-2),
                       d=x.size(-1))
    else:
        index = repeat(idx,
                       'b i -> b i j',
                       j=x.size(2))                                       # 这里这样做的原因是gather的index维度不一样，但是其他维度大小要和原张量一致
    x_1 = torch.gather(x, dim=1, index=index)                       # (B, N, N, D) ----> (B, n_loc, N, D)
    if not mask:
        index = repeat(idx,
                       'b j -> b i j d',
                       i=x_1.size(-3),
                       d=x_1.size(-1))
    else:
        index = repeat(idx,
                       'b j -> b i j',
                       i=x_1.size(1))
    x_2 = torch.gather(x_1, dim=2, index=index)                    # (B, n_loc, N, D) ----> (B, n_loc, n_loc, D)
    return x_2



# 将一个batch的节点按照batch节点数最大的dataset进行补齐
# 批处理padding工具(这个函数是用于对batch_list中的数据进行padding处理)
def pad_zeros(batch_list, keys, max_len, collect_dim=-3, data_type='1d', cat=False, value=0, output_dtype=None):
    # 1d: set of [..., pad_dim, ...], 2d: set of [..., pad_dim, pad_dim + 1, ...]
    # To:
    # 1d: [..., collect_dim, pad_dim, ...], 2d: [..., collect_dim, pad_dim, pad_dim + 1, ...]
    assert collect_dim < 0
    pad_dim = collect_dim + 1                                               # (B,N,D)   padding维度是-2的话那堆叠维度应该是-3

    collect = torch.concat if cat else torch.stack

    dic_data = {}
    for k in keys:
        if data_type == '1d':
            collection = collect([F.pad(g[k],
                   tuple([0] * (np.abs(pad_dim) - 1) * 2 + [0, max_len - g[k].shape[pad_dim]]),     #这里这样写是为符合F.pad的要求，保证后边的维度不受pad影响，而目标pad维度进行pad
                   'constant', value)       # [0] * (np.abs(pad_dim) - 1) * 2前面的位置补齐，[0, max_len - g[k].shape[pad_dim]]真实要补的位置
                   for g in batch_list], dim=collect_dim)
        if data_type == '2d':
            collection = collect([F.pad(g[k],
                   tuple([0] * (np.abs(pad_dim) - 2) * 2 + [0, max_len - g[k].shape[pad_dim]]*2),
                   'constant', value)
                   for g in batch_list], dim=collect_dim)
        else:
            assert data_type in ['1d', '2d']

        if not isinstance(output_dtype, type(None)):
            collection = collection.to(output_dtype)
        dic_data[k] = collection

    return dic_data



def read_rdkit_mol(mol):
    if mol.endswith('pdb'):
        mol = Chem.MolFromPDBFile(mol)
    elif mol.endswith('mol'):
        mol = Chem.MolFromMolFile(mol)
    elif mol.endswith('mol2'):
        mol = Chem.MolFromMol2File(mol)
    elif mol.endswith('sdf'):
        SD = Chem.SDMolSupplier(mol)
        mol = [x for x in SD][0]
    else:
        mol = Chem.MolFromSmiles(mol)
    return mol






