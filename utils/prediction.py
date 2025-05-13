import os

import numpy as np
import pandas as pd
from ray.util.multiprocessing import Pool

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import torch_geometric

from model.layers import LigPose
from utils.data_utils import pad_zeros, batch_index_select_for_edge, read_rdkit_mol
from utils.pdbbind_preprocess import *
from utils.pdbbind_utils import collate_dummy, pred_ens, assign_struct
from common import *

# if is_notebook():
#     from tqdm.notebook import tqdm
# else:
from tqdm import tqdm


def set_device(device):
    if device == 'cpu':
        torch.set_num_threads(16)
    else:
        torch.cuda.set_device(device)


def load_LigPose(param_path):
    ligpose = LigPose(param_path=param_path)
    return ligpose


def get_input(protein=None,
              ligand=None,
              ref_pocket_center=None,
              batch_csv=None):
    if batch_csv is not None:
        df_input = pd.read_csv(batch_csv)
        protein_list = df_input['protein'].values
        ligand_list = df_input['ligand'].values
        ref_pocket_center_list = df_input['ref_pocket_center'].values
    else:
        assert protein is not None and ligand is not None and ref_pocket_center is not None
        if not isinstance(protein, list):
            protein_list = [protein]
        else:
            protein_list = protein
        if not isinstance(ligand, list):
            ligand_list = [ligand]
        else:
            ligand_list = ligand
        if not isinstance(ref_pocket_center, list):
            ref_pocket_center_list = [ref_pocket_center]
        else:
            ref_pocket_center_list = ref_pocket_center

    input_list = [(i, j, k) for i, j, k in zip(protein_list, ligand_list, ref_pocket_center_list)]
    return input_list


def prepare_single_input(tupin, dis=15):
    f_name_list, idx, cache_path = tupin
    p_path, l_path, ref_path = f_name_list

    # =========== ligand encoding ===========
    ligand_mol = read_rdkit_mol(l_path)
    AllChem.EmbedMolecule(ligand_mol, maxAttempts=10, useRandomCoords=True, clearConfs=False)
    ff = Chem.rdForceFieldHelpers.MMFFGetMoleculeForceField(
        ligand_mol, Chem.rdForceFieldHelpers.MMFFGetMoleculeProperties(ligand_mol))
    for atom_i in range(ligand_mol.GetNumAtoms()):
        ff.MMFFAddPositionConstraint(atom_i, 1, 100)  # maxDispl: maximum displacement
    ff.Minimize(maxIts=20)

    ligand_node_features = get_node_feature(ligand_mol, 'ligand')
    ligand_edge, ligand_edge_features = get_ligand_edge_feature(ligand_mol)
    ligand_match = get_liagnd_match(ligand_mol)
    ligand_distmap = get_ligand_unrotable_distance(ligand_mol)

    ref_lig = read_rdkit_mol(l_path)
    ref_pocket_center = get_true_posi(ref_lig)
    smi = l_path


    # =========== protein encoding ===========
    pdb_in_path = p_path
    biodf_protein = PandasPdb().read_pdb(pdb_in_path)
    df_protein = biodf_protein.df['ATOM']

    df_pocket = get_pocket(df_protein, ref_pocket_center, dis=dis, any_atom=True)
    biodf_protein.df['ATOM'] = df_pocket
    tmp_pocket_file = cache_path + f'/{idx}_protein_tmp.pdb'
    biodf_protein.to_pdb(tmp_pocket_file)

    protein_lines = open(tmp_pocket_file, 'r').readlines()
    protein_string = ''.join(protein_atom_filter(protein_lines))
    protein_mol = Chem.MolFromPDBBlock(protein_string)

    protein_node_features = get_node_feature(protein_mol, 'protein')
    protein_edge, protein_edge_features = get_protein_edge_feature(protein_mol)
    protein_true_posi = get_true_posi(protein_mol)
    protein_pdb_info = get_pocket_pdb_info(protein_mol)

    dic_input = dict(
        protein_node_features=protein_node_features,
        protein_edge_features=protein_edge_features,
        protein_true_posi=protein_true_posi,
        protein_pdb_info=protein_pdb_info,

        ligand_node_features=ligand_node_features,
        ligand_edge_features=ligand_edge_features,
        ligand_match=ligand_match,
        ligand_distmap=ligand_distmap,

        ref_pocket_center=ref_pocket_center,

        smi=smi,
    )

    os.remove(tmp_pocket_file)
    np.savez_compressed(f'{cache_path}/{idx}.npz', **dic_input)
    return True


def prepare_input(input_list, cache_path, prepare_data_with_multi_cpu=False):
    delmkdir(cache_path)

    tasks = []
    for idx, f_name_list in enumerate(input_list):
        tasks.append((prepare_single_input, (f_name_list, idx, cache_path)))

    fail = 0
    if prepare_data_with_multi_cpu:
        pool = Pool()
        print('Preparing data...')
        for r in pool.map(try_prepare_pdbbind, tasks):
            if not r:
                fail += 1
    else:
        for task in tqdm(tasks, desc='Preparing data'):
            r = try_prepare_pdbbind(task)
            if not r:
                fail += 1

    print(f'Prepared data: {len(tasks) - fail}/{len(tasks)}, {(len(tasks) - fail) / len(tasks) * 100:.2f}%')



def collate_input(batch_list):
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
                                  'coor_init',
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
    dic_data['len_ligand'] = torch.Tensor([g['len_ligand'] for g in batch_list]).float()
    dic_data['idx'] = [g['idx'] for g in batch_list]
    dic_data['smi'] = [g['smi'] for g in batch_list]

    # pre sample edge
    # unlike nodes, no padding between protein and ligand: [protein, ligand, pad]
    dic_data['edge_feature_init_cycle'] = torch.stack([
        batch_index_select_for_edge(dic_data['edge_feature_init'], loc) for loc in dic_data['node_sampling_loc']
    ], dim=0).float()

    # addidional batch info
    dic_data['x_batch_info'] = repeat(torch.arange(len(batch_list)), 'b -> c b n',
                                      c=dic_data['node_sampling_loc'].size(0),
                                      n=max_len_complex_before_sampling).long()
    dic_data['edge_batch_info'] = repeat(torch.arange(len(batch_list)), 'b -> c b i j',
                                         c=dic_data['node_sampling_loc'].size(0),
                                         i=max_len_complex_before_sampling,
                                         j=max_len_complex_before_sampling).long()
    dic_data['x_batch_info_cycle'] = torch.gather(dic_data['x_batch_info'], dim=-1,
                                                  index=dic_data['node_sampling_loc']).long()
    edge_batch_info_cycle = torch.gather(dic_data['edge_batch_info'], dim=-1,
                                         index=repeat(dic_data['node_sampling_loc'], 'c b j -> c b i j',
                                                      i=max_len_complex_before_sampling))
    dic_data['edge_batch_info_cycle'] = torch.gather(edge_batch_info_cycle, dim=-2,
                                                     index=repeat(dic_data['node_sampling_loc'], 'c b i -> c b i j',
                                                                  j=edge_batch_info_cycle.size(-1))).long()

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

class InferDataset(torch.utils.data.Dataset):
    def __init__(self, args, cache_path, ens=1):
        self.cache_path = cache_path
        self.data_list = [i.split('.')[0] for i in os.listdir(cache_path)]
        self.ens = ens

        # model hyperparameters
        self.n_cycle = args.n_cycle
        self.coor_scale = args.coor_scale
        self.aff_scale = args.aff_scale
        self.max_len_before_sampling = args.max_len_before_sampling
        self.max_len_after_sampling = args.max_len_after_sampling_for_eval
        self.max_len_ligand = args.max_len_ligand
        self.max_ligand_atom_init_distance = args.max_ligand_atom_init_distance
        self.max_ligand_atom_pretrain_distance = args.max_ligand_atom_pretrain_distance

        # Sample pocket
        self.sample_pocket_flag = args.sample_pocket_flag
        self.select_pocket_type = args.select_pocket_type
        self.select_center_type = args.select_center_type

        self.training = False

    def __getitem__(self, i):
        if self.ens > 1:
            complex_graph = []
            for e in range(self.ens):
                complex_graph.append(self.get_complex(self.data_list[i]))
            complex_graph = collate_input(complex_graph)  # use collate_dummy in loader
        else:
            complex_graph = self.get_complex(self.data_list[i])
        return complex_graph

    def get_complex(self, idx):
        ################################################################################################################
        # load prepared data
        ################################################################################################################
        dic_data = np.load(f'{self.cache_path}/{idx}.npz')

        protein_node_feature_init = dic_data['protein_node_features']
        protein_edge_feature_init = dic_data['protein_edge_features']
        protein_position_true = dic_data['protein_true_posi']
        protein_pdb_info = dic_data['protein_pdb_info']
        ligand_node_feature_init = dic_data['ligand_node_features']
        ligand_edge_feature_init = dic_data['ligand_edge_features']
        ref_l_coor = dic_data['ref_pocket_center']
        ligand_distmap = dic_data['ligand_distmap']
        ligand_match = dic_data['ligand_match']
        smi = dic_data['smi']


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
        ref_l_coor = ref_l_coor / self.coor_scale
        ligand_distmap = ligand_distmap / self.coor_scale


        ################################################################################################################
        # sampling nodes: core protein atom (CA,CB) retain, random sampling rest protein atom
        ################################################################################################################
        # get core atoms
        assert len_ligand < self.max_len_after_sampling, f'Ligand too big, heavy atoms: {len_ligand}'

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
        # for mask padding
        protein_node_feature_init = F.pad(protein_node_feature_init, (0, 1), 'constant', 0)
        ligand_node_feature_init = F.pad(ligand_node_feature_init, (0, 1), 'constant', 0)
        edge_feature_init = F.pad(edge_feature_init, (0, 1), 'constant', 0)

        # gen noise for coor init
        flex_coor_mask = F.pad(torch.zeros(len_protein_before_sampling), (0, len_ligand), 'constant', 1).to(torch.float)

        # for dismap
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

            flex_coor_mask=flex_coor_mask,

            coor_scale=self.coor_scale,

            idx=idx,
            smi=smi,
        )
        return complex_graph

    def __len__(self):
        return len(self.data_list)

    def sample_pocket(self, protein_position_true, ref_l_coor, protein_pdb_info, CA_flag, max_len_protein):
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


def save_struct_single(coor, f_name, smi, output_structure_path, min):
    mol = read_rdkit_mol(smi)
    mol = assign_struct(mol, coor, min=min)
    Chem.MolToPDBFile(mol, f'{output_structure_path}/{f_name}.pdb')


def save_struct(struct_pred_list, idx_list, smi_list, ens, struct_suppl_list, output_structure_path, args, min):
    for struct_pred, idx, smi, struct_suppl in zip(struct_pred_list, idx_list, smi_list, struct_suppl_list):
        if ens == 1:
            for i in range(len(idx)):
                save_struct_single(
                    struct_pred[0][-1, i][struct_suppl.ligand_node_loc_after_sampling[i]] * args.coor_scale,
                    str(idx[i]), str(smi[i]), output_structure_path, min,
                )
        else:
            ens_pred = pred_ens(struct_pred[0][-1], struct_suppl)
            save_struct_single(ens_pred * args.coor_scale, str(idx[0]), str(smi[0]), output_structure_path, min,)


def save_screening(screening_pred_list, idx_list, ens, args):
    pred_list = []
    for screening_pred, idx in zip(screening_pred_list, idx_list):
        if ens == 1:
            for i in range(len(idx)):
                pred_list.append(
                    (idx[i], screening_pred[0][i].item() * args.aff_scale * screening_pred[1][i].sigmoid().item())
                )
        else:
            pred_list.append(
                (idx[0], (screening_pred[0].mean() * args.aff_scale * screening_pred[1].sigmoid().mean()).item())
            )
    return pred_list



def predict(
        param_path,
        device='cpu',
        protein=None,
        ligand=None,
        ref_pocket_center=None,
        batch_csv=None,
        prepare_data_with_multi_cpu=False,
        cache_path='./cache',
        ens=1,
        batch_size=5,
        num_workers=8,
        pred_type=None,
        output_structure_path='./ouput_structure',
        output_result_path=None,
        seed=42,
        min=True,
    ):
    set_device(device)
    set_all_seed(seed)
    ligpose = load_LigPose(param_path).to(device)
    ligpose.train(False)
    args = ligpose.args

    input_list = get_input(protein, ligand, ref_pocket_center, batch_csv)
    prepare_input(input_list, cache_path, prepare_data_with_multi_cpu)

    infer_dataset = InferDataset(args, cache_path, ens=ens)
    infer_loader = torch.utils.data.DataLoader(
        dataset=infer_dataset, batch_size=batch_size if ens == 1 else 1,
        num_workers=num_workers, shuffle=False,
        pin_memory=False, persistent_workers=6, prefetch_factor=2,
        collate_fn=collate_input if ens == 1 else collate_dummy)
    infer_loader.dataset.training = False


    dic_result = defaultdict(list)
    with torch.no_grad():
        for g in tqdm(infer_loader, desc='Predicting'):
            g = g.to(device)
            dic_pred = ligpose.infer(g, pred_type=pred_type)

            for k, v in dic_pred.items():
                dic_result[k].append(v)
            dic_result['idx'].append(g.idx)
            dic_result['smi'].append(g.smi)
            dic_result['struct_suppl'].append(
                torch_geometric.data.Data(
                    **dict(
                        ligand_node_loc_after_sampling_flat=g.ligand_node_loc_after_sampling_flat,
                        ligand_node_loc_after_sampling=g.ligand_node_loc_after_sampling,
                        ligand_match=g.ligand_match,
                    )
                )
            )

    if 'structure' in dic_result.keys():
        delmkdir(output_structure_path)
        save_struct(dic_result['structure'], dic_result['idx'], dic_result['smi'],
                    ens, dic_result['struct_suppl'], output_structure_path, args, min)

    protein = []
    ligand = []
    for p, l, _ in input_list:
        protein.append(p)
        ligand.append(l)
    df_output = pd.DataFrame.from_dict(
        dict(
            index=np.arange(len(input_list)),
            protein=protein,
            ligand=ligand,
        )
    )

    if 'screening' in dic_result.keys():
        screening_score = save_screening(dic_result['screening'], dic_result['idx'], ens, args)
        df_output['screening_score'] = np.nan
        loc = []
        score = []
        for i, j in screening_score:
            loc.append(int(i))
            score.append(float(j))
        df_output.loc[loc, 'screening_score'] = score

    if output_result_path is not None:
        df_output.to_csv(output_result_path, index=False)

    shutil.rmtree(cache_path)
    print('DONE')






























