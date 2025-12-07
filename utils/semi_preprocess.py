import io
import os
import lmdb
import tqdm
import numpy as np
import pickle
import gzip
import random
import io
import sys
import logging
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Geometry import Point3D
from biopandas.pdb import PandasPdb
from utils.pdbbind_preprocess import get_true_posi, get_node_feature, get_ligand_edge_feature,\
      get_liagnd_match, get_ligand_unrotable_distance,get_protein_edge_feature, get_pocket_pdb_info,\
      get_semi_pocket_center



def geom_item_to_mol(item, conf_idx=0):
    store = item._store
    elements = store["element"]
    bond_index = store["bond_index"]
    bond_type = store["bond_type"]
    pos_all_confs = store["pos_all_confs"]
    smiles = store["smiles"]
    mol_id = store["mol_id"]

    from rdkit.Chem import rdmolops

    rw_mol = Chem.RWMol()
    
    for z in elements.tolist():
        rw_mol.AddAtom(Chem.Atom(int(z)))

    
    def map_bond_type(bt):
        bt = int(bt)
        if bt == 1: return rdchem.BondType.SINGLE
        if bt == 2: return rdchem.BondType.DOUBLE
        if bt == 3: return rdchem.BondType.TRIPLE
        if bt == 4: return rdchem.BondType.AROMATIC
        return rdchem.BondType.SINGLE

    src, dst = bond_index[0].tolist(), bond_index[1].tolist()
    bt_list = bond_type.tolist()
    for u, v, bt in zip(src, dst, bt_list):
        if u < v:
            rw_mol.AddBond(int(u), int(v), map_bond_type(bt))

    mol = rw_mol.GetMol()


    conf = Chem.Conformer(len(elements))
    coords = pos_all_confs[conf_idx]                            # [n_atoms, 3]
    for i_atom in range(len(elements)):
        x, y, z = coords[i_atom].tolist()
        conf.SetAtomPosition(i_atom, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)

    mol.SetProp("_Name", f"geom_{mol_id}")
    mol.SetProp("smiles", smiles)
    Chem.SanitizeMol(mol)
    return mol


# for geom_data extract                     # 这里的lmdb数据是有点问题的，所以要做一个伪装文件进行提取
class DummyGeomObject:
    pass

class GeomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "utils.data":
            return DummyGeomObject
        return super().find_class(module, name)




def preprocess_ligand_geom(i, item, save_dir):
    try:
        ligand_mol = geom_item_to_mol(item, conf_idx=0)
        
    except Exception as e:

        return None

    ligand_true_posi = get_true_posi(ligand_mol)                           
    ligand_node_features = get_node_feature(ligand_mol, 'ligand')  
    ligand_edge, ligand_edge_features = get_ligand_edge_feature(ligand_mol) 
    ligand_match = get_liagnd_match(ligand_mol)                            
    ligand_distmap = get_ligand_unrotable_distance(ligand_mol)       

    save_path = os.path.join(save_dir, f"{i:06d}.npz")
    np.savez_compressed(save_path, ligand_true_posi=ligand_true_posi,
                        ligand_node_features=ligand_node_features,
                        ligand_edge=ligand_edge,
                        ligand_edge_features=ligand_edge_features,
                        ligand_match=ligand_match,
                        ligand_distmap=ligand_distmap)
    return True
    


def preprocess_protein_pdb(gz_path, save_dir):
    with gzip.open(gz_path, "rt") as f:
        file_name = gz_path.name.split(".")[0]
        tmp_pdb_path = os.path.join(save_dir, f"{file_name}_tmp.pdb")
        
        pdb_str = f.read()
        biodf_protein = PandasPdb().read_pdb_from_list(pdb_str.splitlines())
        df_protein = biodf_protein.df['ATOM']
        chains = df_protein['chain_id'].unique()
        chain = random.choice(chains)
        df_chain = df_protein[df_protein['chain_id'] == chain].reset_index(drop=True)           # 使用bool索引进行取值
        residues = df_chain['residue_number'].unique()
        
        if len(residues) < 100:
            print("protein_resi<100", len(residues))
            return True
        
        else:
            max_start = len(residues) - 100
            start_idx = random.randint(0, max_start)
            selected_residues = residues[start_idx : start_idx + 100]
            df_sub = df_chain[df_chain['residue_number'].isin(selected_residues)].copy()
            df_sub = df_sub.reset_index(drop=True)
        
        file_name = gz_path.name.split(".")[0]
        tmp_pdb_path = os.path.join(save_dir, f"{file_name}_tmp.pdb")
        ppdb_sub = PandasPdb()                                                                  # 这个dataframe一定要用pandaspdb进行包装一下才能转为pdb格式
        ppdb_sub.df['ATOM'] = df_sub
        ppdb_sub.to_pdb(tmp_pdb_path, records=['ATOM'])                                    # 这里一定要指定临时文件位置

        protein_mol = Chem.MolFromPDBFile(tmp_pdb_path)
        if protein_mol is None:
            print("RDKit parse failed:", gz_path)
            return True
        
        protein_node_features = get_node_feature(protein_mol, 'protein')    
        protein_edge, protein_edge_features = get_protein_edge_feature(protein_mol) 
        protein_true_posi = get_true_posi(protein_mol)                               
        protein_pdb_info = get_pocket_pdb_info(protein_mol)                  
        center_coor = get_semi_pocket_center(tmp_pdb_path)

        file_name = gz_path.name.split(".")[0] 
        save_path = os.path.join(save_dir, f"{file_name}.npz")
        np.savez_compressed(save_path,
                            protein_node_features=protein_node_features,
                            protein_edge_features=protein_edge_features,
                            protein_true_posi=protein_true_posi,
                            protein_pdb_info=protein_pdb_info,
                            center_coor=center_coor)
        os.remove(tmp_pdb_path)
        return True










# DEBUG
# class DummyGeomObject:
#     """用来接住 LMDB 里所有来自 utils.data 的对象."""
#     pass


# class GeomUnpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         # 只要是原来模块是 utils.data 的类，统统用 DummyGeomObject 来代替
#         if module == "utils.data":
#             # 你也可以 print 一下看看 name 是什么：
#             # print(f"[DEBUG] unpickle from utils.data: {name}")
#             return DummyGeomObject
#         # 其他模块正常处理
#         return super().find_class(module, name)


# with env.begin() as txn:
#     cursor = txn.cursor()
#     for i, (key, value) in enumerate(cursor):
#         try:
#             item = GeomUnpickler(io.BytesIO(value)).load()
#         except Exception as e:
#             print("\n========== PICKLE DEBUG ==========")
#             print("Error:", e)
#             import traceback
#             traceback.print_exc()
#             print("==================================\n")
#             raise

#         # 👇 print for 5 items information
#         if i < 5:
            
#             print(f"[DEBUG] i={i}, key={key!r}, item_dict_keys={item.__dict__.keys()}", flush=True)
#             print("item.__dict__:", list(item.__dict__.keys()))
#             print("[DEBUG] store keys:", item._store.keys())


#         # test for 20 items
#         if i >= 20:
#             print("Checked first 20 entries, looks OK, breaking loop.", flush=True)
#             break

