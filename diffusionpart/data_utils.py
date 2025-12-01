import os
import pickle
import random
import torch
from torch.utils import data
import numpy as np
import rdkit.Chem as Chem


def collate_struct(batch_list,args):
    max_len_fragment = max([item['fragment_feature'].shape[0] for item in batch_list])
    max_len_pos = max([item['fragment_pos'].shape[0] for item in batch_list])
    fragment_feat_tensor = torch.zeros(len(batch_list), max_len_fragment, args.feature_size)             # batch_list[0]['fragment_feature'].shape[1])
    fragment_feat_mask = torch.zeros((len(batch_list), max_len_fragment,1), dtype=torch.bool)       # 这里为什么要选取最大长度呢
    fragment_pos_tensor = torch.zeros((len(batch_list), max_len_pos, 3))
    fragment_pos_mask = torch.zeros((len(batch_list), max_len_pos,1), dtype=torch.bool)             # emmm这里需要再考虑一下好吧
    adj_matrix_tensor = torch.zeros((len(batch_list), max_len_fragment, max_len_fragment))
    adj_matrix_mask = torch.zeros((len(batch_list), max_len_fragment, max_len_fragment), dtype=torch.bool) # 这个连接矩阵需要构建一下的

    max_len_protein = max([item['protein_feat'].shape[0] for item in batch_list])
    protein_feat_tensor = torch.zeros(len(batch_list), max_len_protein, args.protein_feature_size)  # batch_list[0]['protein_feat'].shape[1])
    protein_pos_tensor = torch.zeros((len(batch_list), max_len_protein, 3))
    protein_feat_mask = torch.zeros((len(batch_list), max_len_protein,1), dtype=torch.bool)
    protein_edge_mask = torch.zeros((len(batch_list), max_len_protein, max_len_protein), dtype=torch.bool)

# assignment data
    for i, sample in enumerate(batch_list):
        fragment_len = sample['fragment_feature'].shape[0]
        fragment_feat_tensor[i, :fragment_len, :] = torch.tensor(sample['fragment_feature'], dtype=torch.float)
        fragment_feat_mask[i, :fragment_len, :] = 1
        fragment_pos_tensor[i, :fragment_len, :] = torch.tensor(sample['fragment_pos'], dtype=torch.float)
        fragment_pos_mask[i, :fragment_len, :] = 1
        # construct adjacency matrix
        adj_matrix_tensor[i, :sample['adj_matrix'].shape[0], :sample['adj_matrix'].shape[1]] = torch.tensor(sample['adj_matrix'])
        adj_matrix_mask[i, :sample['adj_matrix'].shape[0], :sample['adj_matrix'].shape[1]] = 1 - torch.eye(sample['adj_matrix'].shape[0])       # 这里是使用数学操作将对角线上的值变为0，其余位置变为1 ---->屏蔽自环的mask

        

        protein_len = sample['protein_feat'].shape[0]
        protein_feat_tensor[i, :protein_len, :] = torch.tensor(sample['protein_feat'], dtype=torch.float)
        protein_pos_tensor[i, :protein_len, :] = torch.tensor(sample['protein_pos'], dtype=torch.float)
        protein_feat_mask[i, :protein_len, :] = 1
        # construct protein edge mask
        protein_edge_mask[i, :sample['protein_feat'].shape[0], :sample['protein_feat'].shape[0]] = 1 - torch.eye(sample['protein_feat'].shape[0])
    
    return{'fragment_feat_tensor': fragment_feat_tensor,
           'fragment_feat_mask': fragment_feat_mask,
           'fragment_pos_tensor': fragment_pos_tensor,
            'fragment_pos_mask': fragment_pos_mask,
            'fragment_edge_tensor': adj_matrix_tensor,
            'fragment_edge_mask': adj_matrix_mask,
                
            'protein_feat_tensor': protein_feat_tensor,
            'protein_pos_tensor': protein_pos_tensor,
            'protein_feat_mask': protein_feat_mask,
            'protein_edge_mask': protein_edge_mask
            }


class Fragment(data.Dataset):
    def __init__(self, mode, data_list, fragment_path):
        self.mode = mode
        self.data_list = data_list
        self.fragment_path = fragment_path


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        
        if self.mode == 'train':
            file_name = random.choice(self.data_list)
            try:
                return self.get_frag_complex(file_name)
        
            except Exception as e:
                print(f"[WARNING] Failed loading {file_name}: {e}")
        else:
            file_name = self.data_list[idx]
            return self.get_frag_complex(file_name)

    def get_frag_complex(self, file_name):
        
        dic_data = np.load(os.path.join(self.fragment_path, file_name.npz), allow_pickle=True).item()
        fragment_feature = dic_data['fragment_feature']
        fragment_pos = dic_data['fragment_pos']
        protein_feat = dic_data['protein_feat']
        protein_pos = dic_data['protein_pos']
        return {'fragment_feature': fragment_feature,'fragment_pos': fragment_pos,'protein_feat': protein_feat,'protein_pos': protein_pos}