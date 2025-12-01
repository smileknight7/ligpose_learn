import pandas as pd
import copy
import pickle
from tqdm.auto import tqdm
import numpy as np
import torch
import random
from utils.chemutils import tree_decomp, get_mol, get_smiles, get_clique_mol_simple, allowable_features
from utils.mol_tree import MolTree_process
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
import matplotlib.pyplot as plt
from rdkit import Chem
from collections import Counter

def get_slots(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]


class Vocab(object):

    def __init__(self, smiles_list):
        self.vocab = smiles_list
        self.vmap = {x: i for i, x in enumerate(self.vocab)}
        # self.slots = [get_slots(smiles) for smiles in self.vocab]

    def get_index(self, smiles):
        try:
            index = self.vmap[smiles]
            return index
        except Exception as e:
            return 0

    def get_smiles(self, idx):
        return self.vocab[idx]

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

    def size(self):
        return len(self.vocab)


class MolTreeNode(object):

    def __init__(self, mol, cmol, clique, cluster_center, atom_indices):
        self.smiles = Chem.MolToSmiles(cmol, canonical=True)
        self.mol = cmol
        self.clique = [x for x in clique]  # copy
        self.cluster_center = cluster_center
        self.atom_indices = atom_indices

        self.neighbors = []
        self.rotatable = False
        if len(self.clique) == 2:
            if mol.GetAtomWithIdx(self.clique[0]).GetDegree() >= 2 and mol.GetAtomWithIdx(
                    self.clique[1]).GetDegree() >= 2:
                self.rotatable = True
        # should restrict to single bond, but double bond is ok







class MolTree(object):
    def __init__(self, mol, vocab=None, ligand_path=None):
        try:
            self.smiles = Chem.MolToSmiles(mol)
        except Exception as e:
            supplier = Chem.ForwardSDMolSupplier(ligand_path, removeHs=True)
            for mol in supplier:
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            self.smiles = smiles
            # print(e)

        self.mol = mol
        self.num_rotatable_bond = 0
        self.vocab = vocab
        '''
        # use reference_vocab and threshold to control the size of vocab
        reference_vocab = np.load('./utils/reference.npy', allow_pickle=True).item()
        reference = defaultdict(int)
        for k, v in reference_vocab.items():
            reference[k] = v'''

        # use vanilla tree decomposition for simplicity
        cliques, edges, cluster_centers, atom_indices= tree_decomp(self.mol, reference_vocab=None)
        self.nodes = []
        root = 0

        for i, c in enumerate(cliques):
            cmol = get_clique_mol_simple(self.mol, c)
            node = MolTreeNode(self.mol, cmol, c, cluster_centers[i], atom_indices[i])
            self.nodes.append(node)
            if min(c) == 0:                                                 # 原子序号最小的cluster作为root
                root = i

        for node in self.nodes:
            if node.rotatable:
                self.num_rotatable_bond += 1

        for x, y in edges:
            self.nodes[x].add_neighbor(self.nodes[y])                       # 树分解中构建边的点添加到node的邻居列表
            self.nodes[y].add_neighbor(self.nodes[x])

        if root > 0:
            self.nodes[0], self.nodes[root] = self.nodes[root], self.nodes[0]

        for i, node in enumerate(self.nodes):
            node.nid = i + 1
            node.wid = vocab.get_index(node.smiles)
            '''
            if len(node.neighbors) > 1:  # Leaf node mol is not marked
                set_atommap(node.mol, node.nid)
            node.is_leaf = (len(node.neighbors) == 1)'''
        # assign node IDs to atoms
        atom_cluster_map = {}
        atom_vocab_map = {}
        for i, node in enumerate(self.nodes):
            for atom_idx in node.atom_indices:
                atom_cluster_map[atom_idx] = node.nid                       # 做一个atom到nid的映射（表示atom的node归属）
                atom_vocab_map[atom_idx] = node.wid                         # key是atom，value是nid

        # create sorted cluster ID array mapped by atom ID
        n_atoms = self.mol.GetNumAtoms()
        atom_cluster_array = np.zeros(n_atoms, dtype=np.int64)
        for atom_idx, cluster_id in atom_cluster_map.items():
            atom_cluster_array[atom_idx] = cluster_id                       # 构建atom到cluster_id的映射
            atom_cluster_array[np.argsort(np.arange(n_atoms))]
        self.atom_cluster_array = atom_cluster_array
                                                                            # 这里的构建还是要考虑一下（可能不需要？）
        atom_vocab_array = np.zeros(n_atoms, dtype=np.int64)
        for atom_idx, vocab_id in atom_vocab_map.items():
            atom_vocab_array[atom_idx] = vocab_id                           # 构建atom到vocab_id的映射
            atom_vocab_array[np.argsort(np.arange(n_atoms))]    
        self.atom_vocab_array = atom_vocab_array
        # every cluster has cluster_id, how to get the cluster_id for every atom in mol?  md这个就是我之前想的问题啊

        self.node_pos = np.array(cluster_centers, dtype=np.float32)
        self.node_wid = np.array([node.wid for node in self.nodes], dtype=np.int64)


    def size(self):
        return len(self.nodes)

    def recover(self):
        for node in self.nodes:
            node.recover(self.mol)

    def assemble(self):
        for node in self.nodes:
            node.assemble()


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

