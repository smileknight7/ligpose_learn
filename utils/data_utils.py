import numpy as np
from rdkit import Chem

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


#########------------------------########
#读取.npz文件
#########------------------------########


data = np.load('your_file.npz')

# 查看文件中包含的数组名称
print(data.files)

# 访问特定的数组
array1 = data['array_name1']
array2 = data['array_name2']

# 使用完毕后关闭文件
data.close()




def batch_index_select(x, idx):
    # select data with gather
    # x =  torch.stack([torch.index_select(x_i, 0, idx_i) for x_i, idx_i in zip(x, idx)], dim=0)
    assert len(x.shape) - len(idx.shape) == 1
    idx = repeat(idx, 'b n -> b n d', d=x.size(-1))
    new_x = torch.gather(x, dim=-2, index=idx)
    return new_x


def batch_index_select_for_edge(x, idx, mask=False):
    # select data with gather, for edge (2D data)
    if not mask:
        index = repeat(idx,
                       'b i -> b i j d',
                       j=x.size(-2),
                       d=x.size(-1))
    else:
        index = repeat(idx,
                       'b i -> b i j',
                       j=x.size(2))
    x_1 = torch.gather(x, dim=1, index=index)
    if not mask:
        index = repeat(idx,
                       'b j -> b i j d',
                       i=x_1.size(-3),
                       d=x_1.size(-1))
    else:
        index = repeat(idx,
                       'b j -> b i j',
                       i=x_1.size(1))
    x_2 = torch.gather(x_1, dim=2, index=index)
    return x_2


def pad_zeros(batch_list, keys, max_len, collect_dim=-3, data_type='1d', cat=False, value=0, output_dtype=None):
    # 1d: set of [..., pad_dim, ...], 2d: set of [..., pad_dim, pad_dim + 1, ...]
    # To:
    # 1d: [..., collect_dim, pad_dim, ...], 2d: [..., collect_dim, pad_dim, pad_dim + 1, ...]
    assert collect_dim < 0
    pad_dim = collect_dim + 1

    collect = torch.concat if cat else torch.stack

    dic_data = {}
    for k in keys:
        if data_type == '1d':
            collection = collect([F.pad(g[k],
                   tuple([0] * (np.abs(pad_dim) - 1) * 2 + [0, max_len - g[k].shape[pad_dim]]),
                   'constant', value)
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






