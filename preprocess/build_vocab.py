import pickle
import pandas as pd
import copy
import pickle
from tqdm.auto import tqdm
import numpy as np
import torch
import random
from utils.chemutils import get_clique_mol, tree_decomp, get_mol, get_smiles, set_atommap, get_clique_mol_simple, allowable_features
from utils.mol_tree import MolTree_process
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
import os
import matplotlib.pyplot as plt
from rdkit import Chem
import argparse


def add_motif_feature(vocab_df, vocab_processed_path) -> None:

    with open(vocab_df, 'rb') as f:
        vocab_df = pickle.load(f)
    # smile_cluster_list = vocab_df['smile_cluster'].tolist()
    # vocab = Vocab(smile_cluster_list)

    all_vocab_feature = []

    for _, row in vocab_df.iterrows():
        mol = row['mol']
        smile_cluster = row['smile_cluster']
        print(row['smile_cluster'])
        ComputeGasteigerCharges(mol)  # they are Nan for 93 molecules in all of PDBbind. We put a 0 in that case.
        try:
            sssr = Chem.GetSymmSSSR(mol)
            num_rings = len(sssr)
            ring_sizes = [len(list(r)) for r in sssr]

            is_in_ring3 = allowable_features['possible_is_in_ring3_list'].index(3 in ring_sizes)
            is_in_ring4 = allowable_features['possible_is_in_ring4_list'].index(4 in ring_sizes)
            is_in_ring5 = allowable_features['possible_is_in_ring5_list'].index(5 in ring_sizes)
            is_in_ring6 = allowable_features['possible_is_in_ring6_list'].index(6 in ring_sizes)
        except Exception as e:
            num_rings = 0
            is_in_ring3 =0
            is_in_ring4 =0
            is_in_ring5 =0
            is_in_ring6 =0

        weight = Chem.Descriptors.MolWt(mol)

        all_vocab_feature.append({
            'mol' : mol,
            'smile_cluster': smile_cluster,
            'num_rings': num_rings,
            'is_in_ring3': is_in_ring3,
            'is_in_ring4': is_in_ring4,
            'is_in_ring5': is_in_ring5,
            'is_in_ring6': is_in_ring6,
            'weight': weight,
        })

    all_vocab_feature = pd.DataFrame(all_vocab_feature)

    # Min-max normalization
    min_weight = all_vocab_feature['weight'].min()
    max_weight = all_vocab_feature['weight'].max()
    if max_weight != min_weight:
        all_vocab_feature['weight'] = (all_vocab_feature['weight'] - min_weight) / (max_weight - min_weight)
    else:
        all_vocab_feature['weight'] = 0.0
    print(all_vocab_feature)

    with open(vocab_processed_path, 'wb') as f:
        pickle.dump(all_vocab_feature, f)

def get_crossdock_vocab(index_path, origin_path, vocab_path):
    seed = 2022
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    vocab_smile = {}            # smile -->frequency
    vocab_mol = {}              # smile -->mol
    cnt = 0
    rot = 0
    index_names = []
    
    with open(index_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
        # 每行第一个字段是pdb编号
            index = line.split()[0]
            index_names.append(index)

    for i, index in enumerate(tqdm(index_names)):
        if index is None: continue
        
        sdf_path = origin_path + '/' + index + '/' + index + '_lig.mol2'
        
        try:
            mol = Chem.MolFromMolFile(sdf_path, sanitize=False)
            moltree = MolTree_process(mol)
            cnt += 1
            if moltree.num_rotatable_bond > 0:
                rot += 1
        except Exception as e:
            print(e)
            continue

        for c in moltree.nodes:
            smile_cluster = c.smiles
            if smile_cluster not in vocab_smile:
                vocab_smile[smile_cluster] = 1
                vocab_mol[smile_cluster] = c.mol

            else:
                vocab_smile[smile_cluster] += 1
    # 重构字典
    # vocab_smile = dict(sorted(vocab_smile.items(), key=lambda kv: (kv[1], kv[0]), reverse=True))
    data = [{'smile_cluster': smile_cluster, 'frequence': vocab_smile[smile_cluster], 'mol': vocab_mol[smile_cluster]} for smile_cluster in vocab_smile]
    sorted_data = sorted(data, key=lambda x: x['frequence'], reverse=True)
    vocab_df = pd.DataFrame(sorted_data)

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab_df, f)

    # for k, v in vocab.items():
    #     filename.write(k + ':' + str(v))
    #     filename.write('\n')
    # filename.close()

    # number of molecules and vocab
    print('Size of the motif vocab:', len(vocab_smile))
    print('Total number of molecules', cnt)
    print('Percent of molecules with rotatable bonds:', rot / cnt)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--origin_path', type=str,
                        default='/data/lpw/ligpose/data/work_file/tmp',   # 这里注意更改
                        help='origin_pdbbind_path')
    parser.add_argument('--index_path', type=str,
                            default='/data/lpw/ligpose/data/work_file/INDEX_refined_set.txt',   # 
                            help='index_to_pdbbind_refined')
    parser.add_argument('--vocab_path', type=str,
                        default='/data/lpw/ligpose/data/work_file/pdbbind_vocab.pkl',   # 
                        help='index_to_pdbbind_refined')
    parser.add_argument('--vocab_processed_path', type=str,
                    default='/data/lpw/ligpose/data/work_file/pdbbind_vocab.pkl',   # 
                    help='index_to_pdbbind_refined')
    
    args = parser.parse_args()


    vocab_df = get_crossdock_vocab(args.index_path, args.origin_path, args.vocab_path)
    
    with open(args.vocab_path, 'wb') as f:
        pickle.dump(vocab_df, f)
    print('[saved] vocab df ->', args.vocab_path)
    
    add_motif_feature(args.vocab_path, args.vocab_processed_path)