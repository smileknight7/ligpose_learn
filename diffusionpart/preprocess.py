
import copy
import pickle
import torch
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from utils.mol_tree import MolTree_process
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
import matplotlib.pyplot as plt
from rdkit import Chem
from collections import Counter
from biopandas.pdb import PandasPdb
import logging
from utils.chemutils import tree_decomp, get_mol, get_smiles, get_clique_mol, allowable_features
import os
import traceback


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

fh = logging.FileHandler('preprocess.log')
fh.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


RESIDUE_LIST = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]

def get_slots(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]


class Vocab(object):

    def __init__(self, fp_df):
        self.fp_df = fp_df



class MolTreeNode(object):

    def __init__(self, c, cmol, centers, hbd):
        self.clique = c
        self.smiles = Chem.MolToSmiles(cmol, kekuleSmiles=True, canonical=True)
        print(self.smiles)
        self.pos = np.array(centers)
        self.hbd = hbd
        
        # should restrict to single bond, but double bond is ok



def get_protein_features(protein_path, mol, raidus): 

    pdb_df = PandasPdb().read_pdb(protein_path)
    protein_coords = pdb_df.df['ATOM'][['x_coord', 'y_coord', 'z_coord']]           # 拆分成不同的dataframe
    atom_types = pdb_df.df['ATOM']['element_symbol']
    residue_name = pdb_df.df['ATOM']['chain_id'] + pdb_df.df['ATOM']['residue_number'].astype(str)
    residue_type = pdb_df.df['ATOM']['residue_name']

    ligand_coords = np.array(mol.GetConformer().GetPositions())
    protein_coords = np.array(protein_coords)
    pocket_residues = set()
    for i in range(len(protein_coords)):
        for j in range(len(ligand_coords)):
            dist = np.linalg.norm(protein_coords[i] - ligand_coords[j])
            if dist <= raidus:
                pocket_residues.add(residue_name[i])
    protein_atom_idx = [i for i, res in enumerate(residue_name) if res in pocket_residues]
    protein_feature ={
        'atom_types': atom_types[protein_atom_idx],
        'residue_name': residue_name[protein_atom_idx],
        'residue_type': residue_type[protein_atom_idx],
        'coord':protein_coords[protein_atom_idx]
    }
    CA_index = [i for i, atom in enumerate(atom_types[protein_atom_idx]) if atom == 'CA']
    protein_CA = {
        'residue_type': residue_type[protein_atom_idx][CA_index],                       # 如果使用iloc进行切片索引不太一样
        'CA_coord': protein_coords[protein_atom_idx][CA_index],
        'ligand_name': protein_path.split('/')[-1].split('.')[0],                                                                                                                            # protein的名字
    }
    return protein_CA


class MolTree():                                                                        # vocab这里可能后面还是要重建一下！！
    def __init__(self, mol, vocab=None, ligand_path=None):
        try:
            self.smiles = Chem.MolToSmiles(mol)
        
        except Exception as e:
            supplier = Chem.ForwardSDMolSupplier(ligand_path, removeHs=True)
            for mol in supplier:
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            self.smiles = smiles

        self.mol = mol
        for atom in mol.GetAtoms():
            print(atom.GetIdx())
        self.num_rotatable_bond = 0
        self.vocab = vocab

        # use vanilla tree decomposition for simplicity                
        cliques, edges, cluster_centers, = tree_decomp(self.mol, reference_vocab=None)
        self.nodes = []
        root = 0

        for i, c in enumerate(cliques):                                         # node部分内容
            Chem.AddHs(self.mol)
            mol_checkH = self.mol
            Chem.AddHs(mol_checkH)
            Hydro_start = ('O', 'N', 'S', 'P')
            node_hbd = 0
            for atom_idx in c:
                atom = mol_checkH.GetAtomWithIdx(atom_idx)
                if atom.GetSymbol() in Hydro_start:
                    node_hbd += atom.GetTotalNumHs()
            print(c,"cliques")
            cmol = get_clique_mol(self.mol, c)
            node = MolTreeNode(c, cmol, cluster_centers[i],node_hbd)
            self.nodes.append(node)
        for i,c in enumerate(cliques):
            print(node.smiles,"smiles")

        # for x, y in edges:
        #     self.nodes[x].add_neighbor(self.nodes[y])                                   # 树分解中构建边的点添加到node的邻居列表
        #     self.nodes[y].add_neighbor(self.nodes[x])

        # if root > 0:
        #     self.nodes[0], self.nodes[root] = self.nodes[root], self.nodes[0]

        # for i, node in enumerate(self.nodes):
        #     node.nid = i + 1
        #     node.wid = vocab.get_index(node.smiles)

        # # assign node IDs to atoms
        # atom_cluster_map = {}
        # atom_vocab_map = {}
        # for i, node in enumerate(self.nodes):
        #     for atom_idx in node.atom_indices:
        #         atom_cluster_map[atom_idx] = node.nid                       # 做一个atom到nid的映射（表示atom的node归属）
        #         atom_vocab_map[atom_idx] = node.wid                         

        # # create sorted cluster ID array mapped by atom ID
        # n_atoms = self.mol.GetNumAtoms()
        # atom_cluster_array = np.zeros(n_atoms, dtype=np.int64)
        # for atom_idx, cluster_id in atom_cluster_map.items():               # 构建atom到cluster_id的映射
        #     atom_cluster_array[atom_idx] = cluster_id                       
        #     atom_cluster_array[np.argsort(np.arange(n_atoms))]
        # self.atom_cluster_array = atom_cluster_array                                   
        # atom_vocab_array = np.zeros(n_atoms, dtype=np.int64)
        # for atom_idx, vocab_id in atom_vocab_map.items():                   # 构建atom到vocab_id的映射
        #     atom_vocab_array[atom_idx] = vocab_id                           
        #     atom_vocab_array[np.argsort(np.arange(n_atoms))]    
        
        # self.atom_vocab_array = atom_vocab_array

        # every cluster has cluster_id, how to get the cluster_id for every atom in mol?         ！！！！！！！

    def size(self):
        return len(self.nodes)


# 这里可以写两个，一个处理成moltree级别一个处理为dict级别
def process_fragment(pdb_id, data_path, radius, vocab_path=None, moltree=False):
    print(pdb_id)
    vocab = Vocab( fp_df=pd.read_csv(vocab_path, index_col=0))
    mol_path = f'{data_path}/{pdb_id}/{pdb_id}_ligand.sdf'
    protein_path = f'{data_path}/{pdb_id}/{pdb_id}_protein.pdb'
    print(vocab.fp_df.index)
    print(vocab.fp_df.head())

    mol_suppl = Chem.SDMolSupplier(mol_path)
    mol_list = [x for x in mol_suppl if x is not None]
    if len(mol_list) == 0:
        print(f'Cannot read molecule from {mol_path}')
        logger.info(f'Cannot read molecule from {mol_path}')
        return None
    mol = mol_list[0]
    protein_data = get_protein_features(protein_path, mol, radius)
    protein_feat = protein_data['residue_type']
    protein_feat = torch.tensor([RESIDUE_LIST.index(residue) for residue in protein_feat], dtype=torch.long)
    protein_pos = torch.tensor(protein_data['CA_coord'], dtype=torch.float)
    
    if moltree:
        try:
            jtree = MolTree(mol, vocab, ligand_path=mol_path)
            return(jtree, protein_data)
        
        except Exception as e:
            return None
    else:
        try:
            Tree = MolTree(mol, vocab, ligand_path=mol_path)
            for node in Tree.nodes:
                    try:
                        fp_fix =  np.array(vocab.fp_df.loc[node.smiles])  
                    except KeyError:
                        print(f'Smiles {node.smiles} not found in vocabulary.')
                        fp_fix = np.zeros((vocab.fp_df.shape[1],), dtype=np.int64)          # 但是将未知片段置零会不会影响训练呢？
                    atom_TPSA = Chem.rdMolDescriptors._CalcTPSAContribs(Tree.mol)
                    atom_ASA = Chem.rdMolDescriptors._CalcLabuteASAContribs(Tree.mol)
                    tpsa = sum([atom_TPSA[idx] for idx in node.clique])/10
                    asa = (sum([list(atom_ASA[0])[i] for i in node.clique]) + atom_ASA[1])/10
                    node.fp = np.concatenate((np.array([node.hbd]), fp_fix, np.array([tpsa]), np.array([asa])))
            fragment_feature = []
            fragment_pos = []
            for node in Tree.nodes:
                fragment_feature.append(torch.tensor(node.fp))
                fragment_pos.append(torch.tensor(node.pos))
            return{
                'fragment_feature': fragment_feature,
                'fragment_pos': fragment_pos,
                'protein_feat': protein_feat,                     
                'protein_pos': protein_pos                   
                }  
       
        except Exception as e:
            logger.info(msg=f'Error processing {pdb_id}: {e}')
            traceback.print_exc()
            return None