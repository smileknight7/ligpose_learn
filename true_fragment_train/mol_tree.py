# 细粒度优化过程中的数据预处理
# 对比一下和endiffusion/dataset/mol_tree.py的区别
import copy
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from collections import Counter
import logging

import traceback
import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import torch
import tqdm
from biopandas.pdb import PandasPdb

sys.path.append('.')
from data_utils.chemutils import (decode_stereo, enum_assemble, get_clique_mol,
                                  get_mol, get_smiles, set_atommap,
                                  tree_decomp)


def read_pdb(path, mol, raid=6.0):                          # 新增蛋白质读入与口袋截取
    pdb_df = PandasPdb().read_pdb(path)
    print(pdb_df.df.keys())
    protein_coord = pdb_df.df['ATOM'][['x_coord', 'y_coord', 'z_coord']]
    atom_type = pdb_df.df['ATOM']['atom_name']
    residue_name = pdb_df.df['ATOM']['chain_id'] + pdb_df.df['ATOM']['residue_number'].astype(str)   # 拼接原子的链和残基
    residue_type = pdb_df.df['ATOM']['residue_name']
    
    ligand_coord = np.array(mol.GetConformer().GetPositions())
    protein_coord = np.array(protein_coord)
    pocket_residue = set()
    for i in range(len(protein_coord)):
        for j in range(len(ligand_coord)):
            if np.linalg.norm(protein_coord[i] - ligand_coord[j]) < raid:
                pocket_residue.add(residue_name[i])
    protein_atom_idx = [i for i,r in enumerate(residue_name) if r in pocket_residue]         # pocket_residue是使用set()处理后的残基集合        
    protein = {
        'atom_type': [atom_type[i] for i in protein_atom_idx],
        'residue_name': [residue_name[i] for i in protein_atom_idx],
        'residue_type': [residue_type[i] for i in protein_atom_idx],
        'coord': [protein_coord[i] for i in protein_atom_idx]
    }
    CA_idex = [i for i,atom in enumerate(protein['atom_type']) if atom == 'CA']
    protein_CA = {
        'residue_type': [protein['residue_type'][i] for i in CA_idex],
        'coord': [protein['coord'][i] for i in CA_idex],
        'ligand_name': path.split('/')[-1].split('.')[0],
        'pocket_name': path.split('/')[-2]
    }
    return protein_CA


def read_protein_mol(mol_path, protein_path, name, data_name):
    
    mol_suppl = Chem.SDMolSupplier(mol_path)                            
    mol_list = [x for x in mol_suppl if x is not None]
    if len(mol_list) == 0:                                                  
        logger.info(f"[EMPTY_MOL] {data_name} file={mol_path}")
        return None
    
    mol = mol_list[0]
    protein_data = read_pdb(protein_path, mol)
    
    try:
        jt = MolTree(mol, vocab=vocab)
        fragment_data = jt.fragment_pos()
        return fragment_data, protein_data, data_name
    
    except Exception as e:
        
        lig_id = os.path.basename(mol_path)
        bad_ligands.append(lig_id)
        logger.info(f"[BAD_LIGAND] {data_name} file={mol_path}")
        logger.info(f"Error: {repr(e)}")
        logger.info(traceback.format_exc())
        lig_id = os.path.basename(mol_path)
        bad_ligands.append(lig_id)
        print(f"[BAD_LIGAND] data_name={data_name}, lig_file={lig_id}")
        print("incompatible mol", e)
        print(bad_ligands)
        traceback.print_exc()
        return None

def get_slots(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]

class Vocab(object):

    def __init__(self, smiles_list, fp_df):
        self.vocab = smiles_list
        self.vmap = {x:i for i,x in enumerate(self.vocab)}
        self.slots = [get_slots(smiles) for smiles in self.vocab]
        self.fp_df = fp_df                      # self.fp_df ----> 是一个dataframe
        self.fps = [np.array(self.fp_df.loc[smiles]) for smiles in self.vocab]      # 这里新增缓存了片段指纹和ligand大小
        self.mol_sizes = [Chem.MolFromSmiles(smiles).GetNumHeavyAtoms() for smiles in self.vocab]

        
    def get_index(self, smiles):
        return self.vmap[smiles]

    def get_size(self, size):                                           # 下面三个是检索函数
        return [i for i,x in enumerate(self.mol_sizes) if x==size]
    
    def get_array(self, array):
        return [i for i,x in enumerate(self.fps) if np.array_equal(x, array)]
    
    def compute_size_for_idx(self, idx):
        return Chem.MolFromSmiles(self.vocab[idx]).GetNumHeavyAtoms()

    def get_smiles(self, idx):
        return self.vocab[idx]
    
    def get_fp(self, smiles):
        return np.array(self.fp_df.loc[smiles])

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

    def size(self):
        return len(self.vocab)

class MolTreeNode(object):

    def __init__(self, smiles, pos, clique=[], vocab=None, hbd=None):
        self.smiles = smiles
        self.mol = get_mol(self.smiles)
        self.wid = None
        self.fp = None
        if vocab:
            self.fp = vocab.fp_df.loc[smiles]#new embedding
            self.wid = vocab.get_index(smiles)  #这里的embedding少了hbd

        self.clique = [x for x in clique] #copy
        self.neighbors = []
        self.pos = pos
        self.hbd = hbd
        
    def add_neighbor(self, nei_node):
        self.neighbors.append(nei_node)

    def recover(self, original_mol):
        clique = []
        clique.extend(self.clique)
        if not self.is_leaf:
            for cidx in self.clique:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(self.nid)

        for nei_node in self.neighbors:
            clique.extend(nei_node.clique)
            if nei_node.is_leaf: #Leaf node, no need to mark 
                continue
            for cidx in nei_node.clique:
                #allow singleton node override the atom mapping
                if cidx not in self.clique or len(nei_node.clique) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node.nid)

        clique = list(set(clique))
        label_mol = get_clique_mol(original_mol, clique)
        self.label = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol)))
        self.label_mol = get_mol(self.label)

        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return self.label


# assemble(self):  这部分构建树的内容没有assemble过程，但是新增下面这个  MolTreeNode_blur



class MolTreeNode_blur(object):

    def __init__(self, fp, pos, size):
        self.fp = fp
        self.wid = None#check this to differentiate from MolTreeNode
        self.neighbors = []
        self.pos = pos
        self.size = size
        
    def add_neighbor(self, nei_node):
        self.neighbors.append(nei_node)
    
logger = logging.getLogger("mol_tree")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("bad_ligand.log")
fh.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)




class MolTree(object):

    def __init__(self, mol, nodes=None, edge_index=None, vocab=None):
        if mol:#use for data preprocess
            self.smiles = Chem.MolToSmiles(mol)
            self.mol3D = mol
            self.mol3D = Chem.RemoveHs(self.mol3D)
            Chem.Kekulize(self.mol3D)

            cliques, edges = tree_decomp(self.mol3D)            # 构建tree过程是和粗粒度之间一样（但是node部分不太一样）
            self.cliques = cliques
            self.adj_matrix = np.zeros((len(cliques), len(cliques)))
            self.nodes = []
            root = 0
            for i,c in enumerate(cliques):
                Chem.AddHs(self.mol3D)
                mol3D_checkH = self.mol3D
                Chem.AddHs(mol3D_checkH)
                Hydro_start = ('O', 'N', 'S', 'P')
                node_hbd = 0
                for atom_idx in c:
                    atom = mol3D_checkH.GetAtomWithIdx(atom_idx)
                    if atom.GetSymbol() in Hydro_start:
                        node_hbd += atom.GetTotalNumHs()
                
                cmol = get_clique_mol(self.mol3D, c)
                try:
                    node_pos = np.mean([self.mol3D.GetConformer().GetAtomPosition(x) for x in c], axis=0)
                except:
                    #print('Bad Conformer, init 0 position \r')
                    node_pos = np.zeros((1,3))
                node = MolTreeNode(get_smiles(cmol), node_pos, c, hbd=node_hbd)    # node部分有一定区别
                self.nodes.append(node)
                if min(c) == 0:
                    root = i

            for x,y in edges:
                self.nodes[x].add_neighbor(self.nodes[y])        # 对每个node添加邻居  使用这个属性.add_neighbor
                self.nodes[y].add_neighbor(self.nodes[x])
                self.adj_matrix[x, y] = 1
                self.adj_matrix[y, x] = 1#met bug before         # 在tree中添加这个.adj_matrix属性
            if root > 0:                                         # root是根部团簇索引，下面进行重新赋值和重排
                self.nodes[0],self.nodes[root] = self.nodes[root],self.nodes[0]
                self.adj_matrix[[0, root], :] = self.adj_matrix[[root, 0], :]
                self.adj_matrix[:, [0, root]] = self.adj_matrix[:, [root, 0]]

            for i,node in enumerate(self.nodes):
                node.nid = i + 1
                if len(node.neighbors) > 1: #Leaf node mol is not marked
                    set_atommap(node.mol, node.nid)
                node.is_leaf = (len(node.neighbors) == 1)



#用于重构分子
        elif nodes is not None:#use for reconstruction
            #self.nodes = [MolTreeNode(node[0], node[1], vocab=vocab) for node in nodes]
            self.nodes = nodes
            for i in range(len(self.nodes)):
                self.nodes[i].idx = i
            self.adj_matrix = np.zeros((len(nodes), len(nodes)))
            self.decode_adj_matrix = np.zeros((len(nodes), len(nodes)))
            if edge_index is not None:
                exist_edge = set()
                for ind in range(edge_index[0].shape[0]):
                    i, j = edge_index[0][ind], edge_index[1][ind]
                    self.adj_matrix[i, j] = 1
                    self.adj_matrix[j, i] = 1
                    if (i, j) not in exist_edge:
                        self.nodes[i].add_neighbor(self.nodes[j])
                        exist_edge.add((i, j))
                    if (j, i) not in exist_edge:
                        self.nodes[j].add_neighbor(self.nodes[i])
                        exist_edge.add((j, i))
            
        else:
            raise ValueError('Invalid input for MolTreeNodes')
    def add_node(self, node, link_index=None):
        if link_index is not None:
            for i in link_index:
                self.nodes[i].add_neighbor(node)
                node.add_neighbor(self.nodes[i])
            new_adj_matrix = np.zeros((len(self.nodes) + 1, len(self.nodes) + 1))
            new_adj_matrix[:self.adj_matrix.shape[0], :self.adj_matrix.shape[1]] = self.adj_matrix
            new_decode_adj_matrix = np.zeros((len(self.nodes) + 1, len(self.nodes) + 1))
            new_decode_adj_matrix[:self.adj_matrix.shape[0], :self.adj_matrix.shape[1]] = self.decode_adj_matrix
            for i in link_index:
                new_adj_matrix[-1, i] = 1
                new_adj_matrix[i, -1] = 1
                new_decode_adj_matrix[i, -1] = 1
            self.adj_matrix = new_adj_matrix
            self.decode_adj_matrix = new_decode_adj_matrix

        self.nodes.append(node)
    
    def add_edge(self, i, j):
        self.adj_matrix[i, j] = 1
        self.adj_matrix[j, i] = 1
        self.nodes[i].add_neighbor(self.nodes[j])
        self.nodes[j].add_neighbor(self.nodes[i])
        self.decode_adj_matrix[i, j] = 1

    def size(self):
        return len(self.nodes)

    def recover(self):
        for node in self.nodes:
            node.recover(self.mol3D)

    def assemble(self):
        for node in self.nodes:
            node.assemble()
# 构建fragment ----> atom映射

    def fragment_pos(self):                                 # 感觉这里应该加一个当前node内原子的坐标，这样可能不会出错

        num_atoms = self.mol3D.GetNumAtoms()
        atom_true_pos = np.array([list(self.mol3D.GetConformer().GetAtomPosition(i)) for i in range(num_atoms)])
        fragment_src_pos = np.zeros((num_atoms, 3))
        nei_list_pos = [[] for _ in range(num_atoms)]
        #node_idx = list(range(len(self.nodes)))

        for node_ids, atom_ids in enumerate(self.cliques):
            node = self.nodes[node_ids]  # [3]
            node_pos = node.pos  # [3]
            for a in atom_ids:
                nei_list_pos[a].append(node_pos )

        for a in range(num_atoms):
            if len(nei_list_pos[a]) == 0:
                # 按理说不会发生（因为 tree_decomp 会覆盖所有原子）
                # 保险一点：退回原始坐标或 0
                fragment_src_pos[a] = 0.0
            
            elif len(nei_list_pos[a]) == 1:
                fragment_src_pos[a] = nei_list_pos[a][0]
            else:
                # 属于多个 fragment → 取中心的平均
                fragment_src_pos[a] = np.mean(nei_list_pos[a], axis=0)
        
        print("sucess")
        print(nei_list_pos)

        fragement_pos = {
            'fragment_pos': nei_list_pos,
            'fragment_src_pos': fragment_src_pos,
            'atom_true_pos': atom_true_pos}
        
        return fragement_pos






if __name__ == "__main__":
    import sys
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    vocab_dir = 'dataset/vocab.txt'
    with open(vocab_dir, 'r') as f:
        vocab = [x.strip() for x in f.readlines()]
    vocab_fp = pd.read_csv('dataset/vocab_blur_fps_updated.csv', index_col=0)           # 这里使用的片段编码和粗粒度过程不一样
    vocab = Vocab(vocab, vocab_fp)
    data_name = sys.argv[1] if len(sys.argv) > 1 else 'GEOM_drug'    # 默认使用GEOM_drug数据集
    if data_name == 'crossdock':
        crossdock_dir_sdf = 'data/crossdock_mols.sdf'
        suppl = Chem.SDMolSupplier(crossdock_dir_sdf)
        mol_list = [x for x in suppl if x is not None]
        output_dir = 'data/crossdock_blur_trees'
            #code used for save mol trees of crossdock dataset
        for i, mol in enumerate(tqdm.tqdm(mol_list)):
            tree_list = []
            try:#check vocab compat automatically
                jt = MolTree(mol)           # 这里暂时没传vocab
                tree_list.append(jt)
            except:
                continue
            with open(os.path.join(output_dir, f'{i}_drug_trees.pkl'), 'wb') as f:
                    pickle.dump(tree_list, f)
        print(f'{len(tree_list)} trees saved')


    elif data_name == 'GEOM_drug':
        base_dir = 'data/GEOM/rdkit_folder/drugs/'
        output_dir = 'data/GEOM_drugs_trees_blur_correct_adj'
        pickled_path = [os.path.join(base_dir, d) for d in os.listdir(base_dir)]
        start_time = time.time()
        for i, p in enumerate(tqdm.tqdm(pickled_path)):
            tree_list = []
            with open(p, 'rb') as f:
                try:
                    file = pickle.load(f)
                    mols = [file['conformers'][i]['rd_mol'] for i in range(len(file['conformers']))]
                    random.shuffle(mols)        # ['conformers'][i]['rd_mol']这里是取value索引再取value
                    if len(mols) > 4:
                        mols = mols[:4]
                except:
                    continue #some pickled files are corrupted
                for mol in mols:
                    try:
                        jt = MolTree(mol, vocab=vocab)  #构建分子树
                        tree_list.append(jt)
                    except:
                        continue#some mols are corrupted
                    
            if len(tree_list) > 1:
                with open(os.path.join(output_dir, f'{i}_drug_trees.pkl'), 'wb') as f:
                    pickle.dump(tree_list, f)
        
    elif data_name == 'crossdock_cond':
        crossdock_dir_split = "data/split_by_name.pt"
        crossdock_dir_data = "data/crossdocked_pocket10"
        split_paths = torch.load(crossdock_dir_split)
        output_trees = {"train": [], "test": []}
        num_workers = 64
        if not num_workers:
            for name in split_paths.keys():
                for line in tqdm.tqdm(split_paths[name]):
                    mol_path = os.path.join(crossdock_dir_data, line[1])
                    protein_path = os.path.join(crossdock_dir_data, line[0])
                    jt, protein_data = read_protein_mol(mol_path, protein_path, name)
                    
        else:
            #multiprocessing
            pool = mp.Pool(processes=num_workers)
            for name in split_paths.keys():
                batch_paths = []
                for line in tqdm.tqdm(split_paths[name]):
                    mol_path = os.path.join(crossdock_dir_data, line[1])
                    
                    protein_path = os.path.join(crossdock_dir_data, line[0])
                    batch_paths.append((mol_path, protein_path, name))
                    if len(batch_paths) == num_workers:
                        results = pool.starmap(read_protein_mol, batch_paths)
                        for result in results:
                            if result is not None:
                                output_trees[name].append(result)
                        batch_paths = []
                if len(batch_paths) > 0:
                    results = pool.starmap(read_protein_mol, batch_paths)
                    for result in results:
                        if result is not None:
                            output_trees[name].append(result)
                
        for i, d in tqdm.tqdm(enumerate(output_trees["train"])):
            with open(f"data/crossdock_trees_cond_name/train_{i}", "wb") as f:
                pickle.dump(d, f)
        for i, d in tqdm.tqdm(enumerate(output_trees["test"])):
            with open(f"data/crossdock_trees_cond_name/test_{i}", "wb") as f:
                pickle.dump(d, f)
   


    elif data_name == 'PDBbind_cond':
        base_dir = ''
        pdbbind_split = base_dir + "data/pdbbind_split_whole.pt"
        pdbbind_data = base_dir
        split_paths = torch.load(pdbbind_split)
        output_trees = []                                                   # 当前修改这部分不区分rain和test，再dataloader前引入
        num_workers = 4
        bad_ligands = []
        if not num_workers:
            for name in split_paths.keys():
                for line in tqdm.tqdm(split_paths[name]):
                    mol_path = os.path.join(pdbbind_data, line[1])
                    protein_path = os.path.join(pdbbind_data, line[0])
                    result = read_protein_mol(mol_path, protein_path, name , data_name)
                    if result is None:
                        continue

                    filename = os.path.basename(line[1])
                    data_name = filename[:4]

                                                                    # 这里有点冗余，想想可以把这里直接改掉了后面还要再拆出来
                    
                    output_trees.append(result)

        else:
                #multiprocessing
                pool = mp.Pool(processes=num_workers)
                for name in split_paths.keys():
                    batch_paths = []
                    for line in tqdm.tqdm(split_paths[name]):
                        mol_path = os.path.join(pdbbind_data, line[1])
                        protein_path = os.path.join(pdbbind_data, line[0])
                        filename = os.path.basename(line[1])
                        data_name = filename[:4] 

                        batch_paths.append((mol_path, protein_path, name, data_name))
                        if len(batch_paths) == num_workers:                     # 数据加载够了才会使用并行运算
                            results = pool.starmap(read_protein_mol, batch_paths)
                            for result in results:
                                if result is not None:
                                    output_trees.append(result)
                            batch_paths = []
                    if len(batch_paths) > 0:
                        results = pool.starmap(read_protein_mol, batch_paths)
                        for result in results:
                            if result is not None:
                                output_trees.append(result)
            
        # except Exception as e:
        # print("incompatible mol:", e)
        # traceback.print_exc()
        # return None

        for frag, protein, data_name in output_trees:
            np.savez(                                                       # 看下ligpose那里这部分是怎么写的
                f"data/dataset/train/{data_name}.npz",
                frag=np.array(frag, dtype=object),
                protein=np.array(protein, dtype=object)
            )

        ##校验是否读取正常
        data = np.load(f"data/dataset/train/{data_name}.npz", allow_pickle=True)      
        data_frag = data['frag'].item()
        data_frag_pos = data_frag['fragment_src_pos']                     # 原子数量是对的，但是拆分方式可能有问题（应该要换一种拆分方式，将片段拆分的大一点）
        atom_true_pos = data_frag['atom_true_pos']
        print(data_frag_pos)
        print(atom_true_pos)
        for frag, protein, data_name in output_trees:
            np.savez(
                f"data/dataset/train/{data_name}.npz",
                frag=np.array(frag, dtype=object),
                protein=np.array(protein, dtype=object)
            )

        
        
        # for i, d in tqdm.tqdm(enumerate(output_trees["test"])):
        #     with open(f"data/crossdock_trees_cond_name/test_{i}", "wb") as f:
        #         pickle.dump(d, f)


    else:
        raise ValueError('Wrong data name')
