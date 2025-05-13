import os
import sys
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-1]))
#这段代码是将该文件所在文件夹中的python脚本作为模块导入到当前脚本中
import pickle
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import re
import numpy as np
import scipy
import scipy.spatial
import pandas as pd
from biopandas.pdb import PandasPdb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')



# 加载 pdbbind_screening_allow_dict.pkl
# with open('/root/root/autodl-tmp/suppl/pdbbind_screening_allow_dict.pkl', 'rb') as f:
#     allow_dict = pickle.load(f)
#
#
# print(allow_dict)

#提取PDB结构中的原子坐标信息（将含有ATOM的行进行抽提）
def protein_atom_filter(pdb_lines):
    pdb_lines = [line for line in pdb_lines if line.startswith('ATOM')]
    # pdb_lines = [line for line in pdb_lines if line[13:16] in ['CA ']] #['CA ', 'C  ', 'N  ', 'CB ']
    return pdb_lines

#设计独热编码，方便后边对元素类型进行标记
def onehot_with_allowset(x, allowset, with_unk=True):
    if x not in allowset and with_unk:
        x = allowset[0]  # UNK first
    return list(map(lambda s: x == s, allowset))

#配体生成特征向量（其实是对分子的每个原子生成了特征向量，原子种类,氢键等等）
def get_ligand_node_feature(atom, idx, ring_info, canonical_rank):
    # encode with rich features
    atom_features = \
        onehot_with_allowset(atom.GetSymbol(), ['UNK', 'C', 'O', 'N', 'S', 'F', 'Cl', 'Br', 'B', 'I'], with_unk=True) + \
        onehot_with_allowset(atom.GetTotalDegree(), ['UNK', 1, 2, 3, 4, 5], with_unk=True) + \
        onehot_with_allowset(atom.GetFormalCharge(), ['UNK', -1, -2, 0, 1, 2], with_unk=True) + \
        onehot_with_allowset(atom.GetImplicitValence(), ['UNK', 0, 1, 2, 3], with_unk=True) + \
        onehot_with_allowset(atom.GetTotalNumHs(), ['UNK', 0, 1, 2, 3], with_unk=True) + \
        onehot_with_allowset(atom.GetHybridization(), \
                             ['UNK',
                              Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3,
                              Chem.rdchem.HybridizationType.SP3D,
                              Chem.rdchem.HybridizationType.SP3D2], with_unk=True) + \
        [atom.GetIsAromatic()] + \
        [ring_info.IsAtomInRingOfSize(idx, 3),
         ring_info.IsAtomInRingOfSize(idx, 4),
         ring_info.IsAtomInRingOfSize(idx, 5),
         ring_info.IsAtomInRingOfSize(idx, 6),
         ring_info.IsAtomInRingOfSize(idx, 7),
         ring_info.IsAtomInRingOfSize(idx, 8)]
    atom_features = np.array(atom_features).astype(np.float32)
    return atom_features


def get_protein_node_feature(atom):  # simple feature
    atom_features = \
        onehot_with_allowset(atom.GetSymbol(),
                             ['C', 'O', 'N', 'S'], \
                             with_unk=False) + \
        onehot_with_allowset(atom.GetTotalDegree(), [1, 2, 3, 4, 6], with_unk=False) + \
        onehot_with_allowset(atom.GetImplicitValence(), [0, 1, 2, 3, 4], with_unk=False) + \
        onehot_with_allowset(atom.GetTotalNumHs(), [0, 1, 2, 3, 4], with_unk=False) + \
        onehot_with_allowset(atom.GetHybridization(), \
                             ['UNK',
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3],
                             with_unk=True) + \
        onehot_with_allowset(atom.GetPDBResidueInfo().GetResidueName(), \
                             ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', \
                              'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'], \
                             with_unk=False) + \
        onehot_with_allowset(atom.GetPDBResidueInfo().GetName(), \
                             [' OD2', ' OE1', ' CD1', ' NE1', ' CB ', ' CZ ', ' CH2', ' SG ', ' CG ', ' CZ2', \
                              ' N  ', ' OG ', ' O  ', ' SD ', ' NE2', ' CE2', ' NZ ', ' OH ', ' NE ', ' CE ', \
                              ' CD2', ' ND2', ' OXT', ' CG2', ' C  ', ' CE1', ' CD ', ' OG1', ' CZ3', ' NH2', \
                              ' OE2', ' ND1', ' OD1', ' CE3', ' CA ', ' NH1', ' CG1'],
                             with_unk=False)
    atom_features = np.array(atom_features).astype(np.float32)
    return atom_features

#调用之前提取边特征的函数来实现节点特征的提取—————蛋白质和小分子
def get_node_feature(mol, mol_type):
    if mol_type == 'protein':
        node_features = [get_protein_node_feature(atom) for atom in mol.GetAtoms()]
    elif mol_type == 'ligand':
        ring_info = mol.GetRingInfo()
        canonical_rank = Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=True)
        node_features = [get_ligand_node_feature(atom, idx, ring_info, canonical_rank) for idx, atom in
                         zip(range(mol.GetNumAtoms()), mol.GetAtoms())]
    else:
        assert mol_type in ['protein', 'ligand']
    node_features = np.stack(node_features)
    return node_features


def get_full_connecte_dge(mol):
    # include self-loop
    edge = np.array([[i, j] for i in range(mol.GetNumAtoms()) for j in range(mol.GetNumAtoms())])
    return edge


def get_have_bond(mol, idx_1, idx_2):
    bond = mol.GetBondBetweenAtoms(int(idx_1), int(idx_2))
    if bond == None:
        return 0
    else:
        return 1


def get_atom_distance(mol_conf, idx_1, idx_2):
    coor_1 = mol_conf.GetAtomPosition(int(idx_1))
    coor_2 = mol_conf.GetAtomPosition(int(idx_2))
    return coor_1.Distance(coor_2)

#调用之前提取边特征的函数来实现边特征的提取——————蛋白质
def get_protein_bond_feature(mol, mol_conf, idx_1, idx_2):
    bond = mol.GetBondBetweenAtoms(int(idx_1), int(idx_2))
    if bond == None:
        edge_feature = [1] + [0] * 5
    else:
        edge_feature = [0]
        edge_feature += onehot_with_allowset(bond.GetBondType(), ['UNK', \
                                                                  Chem.rdchem.BondType.SINGLE, \
                                                                  Chem.rdchem.BondType.DOUBLE, \
                                                                  Chem.rdchem.BondType.TRIPLE, \
                                                                  Chem.rdchem.BondType.AROMATIC], with_unk=True)
    edge_feature = np.array(edge_feature).astype(np.float32)
    return edge_feature


def get_protein_edge_feature(mol):
    edge = get_full_connecte_dge(mol)
    mol_conf = mol.GetConformer()
    edge_features = np.array([get_protein_bond_feature(mol, mol_conf, i, j) for i, j in edge])
    edge_features = edge_features.reshape(mol.GetNumAtoms(), mol.GetNumAtoms(), -1)
    return edge, edge_features


def get_ligand_bond_feature(mol, idx_1, idx_2):
    bond = mol.GetBondBetweenAtoms(int(idx_1), int(idx_2))
    if bond == None:
        edge_feature = [1] + [0] * 5
    else:
        edge_feature = [0]
        edge_feature += onehot_with_allowset(bond.GetBondType(), ['UNK', \
                                                                  Chem.rdchem.BondType.SINGLE, \
                                                                  Chem.rdchem.BondType.DOUBLE, \
                                                                  Chem.rdchem.BondType.TRIPLE, \
                                                                  Chem.rdchem.BondType.AROMATIC], with_unk=True)
    edge_feature = np.array(edge_feature).astype(np.float32)
    return edge_feature


def get_ligand_edge_feature(mol):
    edge = get_full_connecte_dge(mol)
    edge_features = np.array([get_ligand_bond_feature(mol, i, j) for i, j in edge])
    edge_features = edge_features.reshape(mol.GetNumAtoms(), mol.GetNumAtoms(), -1)
    return edge, edge_features


def get_true_posi(mol):
    mol_conf = mol.GetConformer()
    node_posi = np.array([mol_conf.GetAtomPosition(int(idx)) for idx in range(mol.GetNumAtoms())])
    return node_posi


def get_liagnd_match(mol):
    matches = mol.GetSubstructMatches(mol, uniquify=False, useChirality=True)
    return np.array(matches)


def get_pocket(df_protein, center_coor, dis, any_atom=False):
    df_pocket = df_protein.copy()
    if not any_atom:
        df_tmp = df_pocket[df_pocket['atom_name'] == 'CA'].copy()
    else:
        df_tmp = df_pocket.copy()

    p_coor = df_tmp[['x_coord', 'y_coord', 'z_coord']].values
    p2l_dismap = scipy.spatial.distance.cdist(p_coor, center_coor, metric='euclidean')
    p2l_dis = p2l_dismap.min(-1)
    df_tmp['pocket_flag'] = (p2l_dis < dis)
    pocket_res_chain = []
    for res_i in df_tmp['residue_number'].unique():
        for chain_i in df_tmp['chain_id'].unique():
            df_sub = df_tmp[(df_tmp['residue_number'] == res_i) & (df_tmp['chain_id'] == chain_i)]
            tmp_dis = df_sub['pocket_flag'].values
            if any(tmp_dis):
                pocket_res_chain.append(f'{res_i},{chain_i}')

    df_pocket['pocket_flag'] = [True if str(df_pocket.loc[i, 'residue_number']) + ',' + str(
        df_pocket.loc[i, 'chain_id']) in pocket_res_chain else False
                                for i in range(len(df_pocket))]
    df_pocket = df_pocket[df_pocket['pocket_flag'] == True][df_pocket.columns[:-1]]
    return df_pocket


def get_pocket_pdb_info(protein_mol):
    return [f'{atom.GetPDBResidueInfo().GetChainId()}_{atom.GetPDBResidueInfo().GetResidueNumber()}' for atom in
            protein_mol.GetAtoms()]


def read_mol_with_pdb_smi(pdb_path, smiles):
    ligand_mol = Chem.MolFromPDBFile(pdb_path)
    ligand_template = Chem.MolFromSmiles(smiles)
    ligand_mol = AllChem.AssignBondOrdersFromTemplate(ligand_template, ligand_mol)
    assert ligand_mol != None
    return ligand_mol


def get_ligand_unrotable_distance(ligand_mol):
    # find rotable bonds
    # '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]' / '[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]'
    rot_patt = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
    patt = Chem.MolFromSmarts(rot_patt)
    hit_bonds = ligand_mol.GetSubstructMatches(patt)
    em = Chem.EditableMol(ligand_mol)
    for (idx1, idx2) in hit_bonds:
        em.RemoveBond(idx1, idx2)
    p = em.GetMol()
    part_list = Chem.GetMolFrags(p, asMols=False)

    added_part_list = []
    for part in part_list:
        tmp = list(part)
        for bonds in hit_bonds:
            i, j = bonds
            if i in part:
                tmp.append(j)
            elif j in part:
                tmp.append(i)
        added_part_list.append(tmp)

    n_atoms = ligand_mol.GetNumAtoms()
    dist_map = np.zeros((n_atoms, n_atoms)) + -1
    mol_conf = ligand_mol.GetConformer()
    for part in added_part_list:
        for i in part:
            for j in part:
                dist_map[i, j] = get_atom_distance(mol_conf, i, j)
    return dist_map


def read_mol_from_pdbbind(data_path, pdb_id):
    ligand_mol2_path = f'{data_path}/{pdb_id}/{pdb_id}_ligand.mol2'
    ligand_mol = Chem.MolFromMol2File(ligand_mol2_path)
    if ligand_mol == None:
        ligand_pdbpath = f'{data_path}/{pdb_id}/{pdb_id}_ligand.pdb'
        ligand_smiles_path = f'{data_path}/{pdb_id}/{pdb_id}_ligand.smi'
        ligand_smiles = open(ligand_smiles_path, 'r').readlines()[0].split('\t')[0]
        ligand_mol = read_mol_with_pdb_smi(ligand_pdbpath, ligand_smiles)
    return ligand_mol


# def get_aff(info_path):
#     lines = open(info_path, 'r').readlines()
#     dic_aff = {line[:4]: float(line[18:23]) for line in lines if not line.startswith('#')}
#     return dic_aff


#info_path = r'/root/ligpose_data/INDEX_refined_set_10.txt'

#############------------############
#此处进行了修改（将pdb信息中的亲和力数据以浮点数的形式进行提取）
############-------------############

info_path = r'/root/ligpose_data/INDEX_refined_set_10.txt'

def get_aff(info_path):
    lines = open(info_path, 'r').readlines()
    dic_aff = {}
    unit_conversion = {
        'nM': 1,
        'uM': 1e3,
        'mM': 1e6,
        'pM': 1e-3,
    }
#建立字典来方便对单位进行换算

    for line in lines:
        if not line.startswith('#'):
            pdb_id = line[:4]
            # 使用正则表达式提取数值和单位
            match = re.search(r'(K[di]=)([\d\.]+)([a-zA-Z]+)', line)
            if match:
                _, value_str, unit = match.groups()
                #这里是进行了解包操作使用_, 放弃对第一个元素解包，try:可以用来捕捉异常数据
                try:
                    value = float(value_str)  # 转换为浮点数
                    if unit in unit_conversion:
                        value *= unit_conversion[unit]  # 转换为 nM 单位
                        dic_aff[pdb_id] = value
                    else:
                        print(f"未知单位: {unit}，跳过该行")
                except ValueError:
                    print(f"无法解析数值: {value_str}，跳过该行")
            else:
                print(f"未找到数值和单位，跳过该行: {line.strip()}")
    print(dic_aff)
    return dic_aff




def process_pdbbind(pdb_id, data_path, suppl_path, cache_path, dis=15):
    # =========== ligand encoding ===========
    ligand_mol = read_mol_from_pdbbind(data_path, pdb_id)
    ligand_true_posi = get_true_posi(ligand_mol)
    ligand_node_features = get_node_feature(ligand_mol, 'ligand')
    ligand_edge, ligand_edge_features = get_ligand_edge_feature(ligand_mol)
    ligand_match = get_liagnd_match(ligand_mol)
    ligand_distmap = get_ligand_unrotable_distance(ligand_mol)


    # =========== protein encoding ===========
    pdb_in_path = f'{data_path}/{pdb_id}/{pdb_id}_protein.pdb'
    biodf_protein = PandasPdb().read_pdb(pdb_in_path)
    df_protein = biodf_protein.df['ATOM']

    df_pocket = get_pocket(df_protein, ligand_true_posi, dis=dis, any_atom=True)
    biodf_protein.df['ATOM'] = df_pocket
    tmp_pocket_file = cache_path + f'/{pdb_id}.pdb'
    biodf_protein.to_pdb(tmp_pocket_file)

    protein_lines = open(tmp_pocket_file, 'r').readlines()
    protein_string = ''.join(protein_atom_filter(protein_lines))
    protein_mol = Chem.MolFromPDBBlock(protein_string)
    if protein_mol == None:
        # use default pocket
        lines = open(f'{data_path}/{pdb_id}/{pdb_id}_pocket.pdb', 'r').readlines()
        protein_string = ''.join(protein_atom_filter(lines))
        protein_mol = Chem.MolFromPDBBlock(protein_string)

    protein_node_features = get_node_feature(protein_mol, 'protein')
    protein_edge, protein_edge_features = get_protein_edge_feature(protein_mol)
    protein_true_posi = get_true_posi(protein_mol)
    protein_pdb_info = get_pocket_pdb_info(protein_mol)


    # =========== suppl info ===========
    aff = get_aff(suppl_path)[pdb_id]




    return dict(protein_node_features=protein_node_features,
                protein_edge_features=protein_edge_features,
                protein_true_posi=protein_true_posi,
                protein_pdb_info=protein_pdb_info,

                ligand_node_features=ligand_node_features,
                ligand_edge_features=ligand_edge_features,
                ligand_true_posi=ligand_true_posi,
                ligand_match=ligand_match,
                ligand_distmap=ligand_distmap,

                aff=aff,
                )


def gen_pdbbind_screening_list(protein_path, ligand_path, prepared_pdbbind_path, save_path):
    protein_info = [line[:-1] for line in open(protein_path, 'r').readlines() if not line.startswith('#')]
    ligand_info = [line[:-1] for line in open(ligand_path, 'r').readlines() if not line.startswith('#')]

    protein_info = {line[:4]: (line[12:18], line[20:]) for line in protein_info}  # pdb_id : (uniprot_id, name)
    ligand_info = {line[:4]: line.split(' ')[-1][1:-1] for line in ligand_info}  # pdb_id : name

    allow_pair = []  # (protein_pdb_id, ligand_pdb_id)
    for l_pdb_id, l_name in tqdm(ligand_info.items()):
        # find pdb_id with same ligand name
        allow_p_pdb_id = [pdb_id for pdb_id, name in ligand_info.items() if name == l_name] if 'mer' not in l_name else [l_pdb_id]

        # add other proteins with same uniprot_id or name to allow_p_pdb_id
        allow_p_uniprot_name = [uniprot_name for pdb_id, uniprot_name in protein_info.items() if pdb_id in allow_p_pdb_id]
        allow_p_uniprot_id = [uniprot_name[0] for uniprot_name in allow_p_uniprot_name]
        allow_p_name = [uniprot_name[1] for uniprot_name in allow_p_uniprot_name]
        allow_p_pdb_id = []
        not_allow_p_pdb_id = []
        for p in protein_info.items():
            p_pdb_id, uniprot_name = p
            p_uniprot_id, p_name = uniprot_name

            if p_uniprot_id in allow_p_uniprot_id or p_name in allow_p_name:
                allow_p_pdb_id.append(p_pdb_id)
            else:
                not_allow_p_pdb_id.append(p_pdb_id)

        allow_pair += [(p_pdb_id, l_pdb_id) for p_pdb_id in allow_p_pdb_id]

    allow_pair = list(set(allow_pair))
    # print('allow_pair:', len(allow_pair))

    prepared_pdb_id_list = [i.split('.')[0] for i in os.listdir(prepared_pdbbind_path)]
    allow_dict = defaultdict(list)
    for p_pdb_id, l_pdb_id in tqdm(allow_pair):
        if p_pdb_id not in prepared_pdb_id_list or l_pdb_id not in prepared_pdb_id_list:
            continue
        if l_pdb_id not in allow_dict.keys():
            allow_dict[l_pdb_id] = [p_pdb_id]
        else:
            allow_dict[l_pdb_id] += [p_pdb_id]

    pickle.dump(allow_dict, open(save_path, 'wb'))


# def try_prepare_pdbbind(intup):
#     f, task = intup
#     try:
#         f(task)
#         return True
#     except:
#         return False

def try_prepare_pdbbind(task):
    try:
        func, args = task
        return func(args)
    except Exception as e:
        import traceback
        print(f"处理任务时出错: {e}")
        print(traceback.format_exc())  # 打印完整的错误堆栈
        return False









