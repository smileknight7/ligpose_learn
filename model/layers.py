import random

import pandas as pd
import torch
import torch.nn.functional as F

from model.GNN import *
from utils.data_utils import batch_index_select



class UpdateBlock(torch.nn.Module):
    def __init__(self,
                 n_block,
                 node_hidden,
                 edge_hidden,
                 n_head,
                 head_hidden,
                 dropout,
                 ):
        super(UpdateBlock, self).__init__()
        self.n_block = n_block - 1
        self.att_layers = torch.nn.ModuleList([
            GateNormAttention(
                node_hidden,
                edge_hidden,
                n_head,
                head_hidden,
                dropout,
                only_coor_out=False,) for _ in range(self.n_block)
        ])
        self.node_FF_layers = torch.nn.ModuleList([
            GateNormFeedForward(node_hidden, dropout) for _ in range(self.n_block)
        ])
        self.edge_FF_layers = torch.nn.ModuleList([
            GateNormFeedForward(edge_hidden, dropout) for _ in range(self.n_block)
        ])
        self.last_update = GateNormAttention(
            node_hidden,
            edge_hidden,
            n_head,
            head_hidden,
            dropout,
            only_coor_out=True,
        )

    def forward(self, complex_graph):
        coor_hidden = []
        for i in range(self.n_block):
            complex_graph = self.att_layers[i](complex_graph)
            complex_graph.x = self.node_FF_layers[i](complex_graph.x)
            complex_graph.edge_attr = self.edge_FF_layers[i](complex_graph.edge_attr)
            coor_hidden.append(complex_graph.coor)
        complex_graph = self.last_update(complex_graph)
        coor_hidden.append(complex_graph.coor)
        complex_graph.coor_hidden = torch.stack(coor_hidden, dim=0)
        return complex_graph


class LigPoseBase(torch.nn.Module):
    def __init__(self, args):
        super(LigPoseBase, self).__init__()
        self.n_cycle = args.n_cycle
        # embed
        self.protein_embed = make_embed(args.protein_input_channel + 1, args.node_hidden)
        self.ligand_embed = make_embed(args.ligand_input_channel + 1, args.node_hidden)
        self.edge_embed = make_embed(args.edge_input_channel + 2, args.edge_hidden)

        # cycle
        self.cycle_node_gate = GateResidue(args.node_hidden)
        self.cycle_edge_gate = GateResidue(args.edge_hidden)
        self.cycle_node_norm = torch.nn.LayerNorm(args.node_hidden)
        self.cycle_edge_norm = torch.nn.LayerNorm(args.edge_hidden)

        # feature and coordinate update block
        self.complex_net = UpdateBlock(args.n_block,
                                       args.node_hidden,
                                       args.edge_hidden,
                                       args.n_head,
                                       args.node_hidden // args.n_head,
                                       args.dropout
                                       )

    def forward(self, complex_graph):
        # embed
        complex_graph = complex_graph
        embed_protein_node_feature_init = self.protein_embed(complex_graph.protein_node_feature_init)
        embed_ligand_node_feature_init = self.ligand_embed(complex_graph.ligand_node_feature_init)
        middle_pad_embed_node_feature_init = torch.cat(
            [embed_protein_node_feature_init, embed_ligand_node_feature_init], dim=-2)
        complex_graph.embed_node_feature_init = batch_index_select(middle_pad_embed_node_feature_init,
                                                                   complex_graph.idx_remove_middle_pad)
        # cycle
        if self.training:  # MC-like style
            cycle_num = random.sample(range(1, self.n_cycle + 1), 1)[0]
            with torch.no_grad():
                for cycle_i in range(cycle_num - 1):
                    complex_graph = self.run_cycle(complex_graph, cycle_i)
            complex_graph = self.run_cycle(complex_graph, cycle_num - 1)
        else:
            for cycle_i in range(self.n_cycle):
                complex_graph = self.run_cycle(complex_graph, cycle_i)

        return complex_graph

    def run_cycle(self, complex_graph, cycle_i):
        x_cycle = batch_index_select(complex_graph.embed_node_feature_init, complex_graph.node_sampling_loc[cycle_i])
        edge_attr_cycle = self.edge_embed(complex_graph.edge_feature_init_cycle[cycle_i])
        coor_init_cycle = batch_index_select(complex_graph.coor_init, complex_graph.node_sampling_loc[cycle_i])

        if cycle_i > 0:
            x_cycle = x_cycle * rearrange(complex_graph.node_cycling_mask == 0, 'b i -> b i ()') + \
                      self.cycle_node_gate(x_cycle, complex_graph.x) * rearrange(complex_graph.node_cycling_mask,
                                                                                 'b i -> b i ()')
            edge_attr_cycle = edge_attr_cycle * rearrange(complex_graph.edge_cycling_mask == 0, 'b i j -> b i j ()') + \
                              self.cycle_edge_gate(edge_attr_cycle, complex_graph.edge_attr) * rearrange(
                complex_graph.edge_cycling_mask, 'b i j -> b i j ()')

            # map updated protein and ligand coor to original coor (for noise protein coor)
            # coor_before_sampling_flat = rearrange(complex_graph.coor_before_sampling, 'b n c -> (b n) c')
            # coor_last_update_after_sampling = rearrange(complex_graph.coor, 'b n c -> (b n) c')
            # coor_before_sampling_flat[complex_graph.node_sampling_loc_flat[cycle_i]] = coor_last_update_after_sampling
            # complex_graph.coor_before_sampling = rearrange(coor_before_sampling_flat, '(b n) c -> b n c', b=x_cycle.size(0))
            coor_init_cycle = coor_init_cycle * rearrange(complex_graph.node_cycling_mask == 0, 'b i -> b i ()') + \
                              complex_graph.coor * rearrange(complex_graph.node_cycling_mask, 'b i -> b i ()')

        complex_graph.x = self.cycle_node_norm(x_cycle)
        complex_graph.edge_attr = self.cycle_edge_norm(edge_attr_cycle)
        complex_graph.coor = coor_init_cycle

        complex_graph.cycle_i = cycle_i
        complex_graph.flex_coor_mask_after_sampling = complex_graph.flex_coor_mask_cycle[cycle_i]
        complex_graph = self.complex_net(complex_graph)
        return complex_graph


class LigPoseStruct(torch.nn.Module):
    def __init__(self, args):
        super(LigPoseStruct, self).__init__()
        self.main_net = LigPoseBase(args)

        # for affinity prediction
        self.aff_layers = torch.nn.Sequential(
            torch.nn.LayerNorm(args.node_hidden + args.edge_hidden),
            torch.nn.Linear(args.node_hidden + args.edge_hidden, args.node_hidden + args.edge_hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(args.node_hidden + args.edge_hidden, 1))

        self.p_x_pretrain_1 = torch.nn.Sequential(
            torch.nn.LayerNorm(args.node_hidden),
            torch.nn.Linear(args.node_hidden, args.node_hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(args.node_hidden, 37))
        self.p_x_pretrain_2 = torch.nn.Sequential(
            torch.nn.LayerNorm(args.node_hidden),
            torch.nn.Linear(args.node_hidden, args.node_hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(args.node_hidden, 20))
        self.l_x_pretrain = torch.nn.Sequential(
            torch.nn.LayerNorm(args.node_hidden),
            torch.nn.Linear(args.node_hidden, args.node_hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(args.node_hidden, 10))
        self.edge_pretrain = torch.nn.Sequential(
            torch.nn.LayerNorm(args.edge_hidden),
            torch.nn.Linear(args.edge_hidden, args.edge_hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(args.edge_hidden, 6))

    def forward(self, complex_graph, return_graph=False):
        complex_graph = self.main_net(complex_graph)

        # for aff
        node_mean_pooling = torch.einsum('b n d -> b d',
                                         complex_graph.x * rearrange(complex_graph.node_mask_after_sampling,
                                                                     'b n -> b n ()')) / \
                            torch.einsum('b n -> b', complex_graph.node_mask_after_sampling).unsqueeze(dim=-1)
        edge_mean_pooling = torch.einsum('b i j d -> b d',
                                         complex_graph.edge_attr * rearrange(complex_graph.edge_mask_after_sampling,
                                                                             'b i j -> b i j ()')) / \
                            torch.einsum('b i j -> b', complex_graph.edge_mask_after_sampling).unsqueeze(dim=-1)
        complex_graph.node_edge_mean_pooling = torch.cat([node_mean_pooling, edge_mean_pooling], dim=-1)
        complex_graph.aff_pred = self.aff_layers(complex_graph.node_edge_mean_pooling).squeeze(dim=-1) if self.training \
            else F.relu(self.aff_layers(complex_graph.node_edge_mean_pooling)).squeeze(dim=-1)

        cycle_i = complex_graph.cycle_i
#模型通过掩码来区分蛋白质和配体的节点
        p_x_masked = rearrange(complex_graph.x, 'b n d -> (b n) d')[
            complex_graph.p_x_mask_bool_cycle[cycle_i].reshape(-1)]
        l_x_masked = rearrange(complex_graph.x, 'b n d -> (b n) d')[
            complex_graph.l_x_mask_bool_cycle[cycle_i].reshape(-1)]
        edge_masked = rearrange(complex_graph.edge_attr, 'b i j d -> (b i j) d')[
            complex_graph.edge_mask_bool_cycle[cycle_i].reshape(-1)]

        p_x_pred_1 = self.p_x_pretrain_1(p_x_masked)
        p_x_pred_2 = self.p_x_pretrain_2(p_x_masked)
        l_x_pred = self.l_x_pretrain(l_x_masked)
        edge_pred = self.edge_pretrain(edge_masked)

        if return_graph:
            return complex_graph
        else:
            return (complex_graph.coor_hidden, complex_graph.aff_pred, p_x_pred_1, p_x_pred_2, l_x_pred, edge_pred)

    @torch.no_grad()
    def infer(self, complex_graph, return_graph=False):
        complex_graph = self.main_net(complex_graph)

        node_mean_pooling = torch.einsum('b n d -> b d',
                                         complex_graph.x * rearrange(complex_graph.node_mask_after_sampling,
                                                                     'b n -> b n ()')) / \
                            torch.einsum('b n -> b', complex_graph.node_mask_after_sampling).unsqueeze(dim=-1)
        edge_mean_pooling = torch.einsum('b i j d -> b d',
                                         complex_graph.edge_attr * rearrange(complex_graph.edge_mask_after_sampling,
                                                                             'b i j -> b i j ()')) / \
                            torch.einsum('b i j -> b', complex_graph.edge_mask_after_sampling).unsqueeze(dim=-1)
        complex_graph.node_edge_mean_pooling = torch.cat([node_mean_pooling, edge_mean_pooling], dim=-1)
        complex_graph.aff_pred = self.aff_layers(complex_graph.node_edge_mean_pooling).squeeze(dim=-1) if self.training \
            else F.relu(self.aff_layers(complex_graph.node_edge_mean_pooling)).squeeze(dim=-1)

        if return_graph:
            return complex_graph
        else:
            return (complex_graph.coor_hidden, complex_graph.aff_pred)


class LigPoseScr(torch.nn.Module):
    def __init__(self, args):
        super(LigPoseScr, self).__init__()
        self.main_net = LigPoseStruct(args)

        # for screening prediction
        self.scr_layers = torch.nn.Sequential(
            torch.nn.LayerNorm(args.node_hidden + args.edge_hidden),
            torch.nn.Linear(args.node_hidden + args.edge_hidden, args.node_hidden + args.edge_hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(args.node_hidden + args.edge_hidden, 1))

    def forward(self, complex_graph):
        complex_graph = self.main_net(complex_graph, return_graph=True)
        complex_graph.scr_pred = self.scr_layers(complex_graph.node_edge_mean_pooling).squeeze(dim=-1)
        return (complex_graph.coor_hidden, complex_graph.aff_pred, complex_graph.scr_pred)

    @torch.no_grad()
    def infer(self, complex_graph):
        complex_graph = self.main_net.infer(complex_graph, return_graph=True)
        complex_graph.scr_pred = self.scr_layers(complex_graph.node_edge_mean_pooling).squeeze(dim=-1)
        return (complex_graph.aff_pred, complex_graph.scr_pred)


class LigPose(torch.nn.Module):
    def __init__(self, args=None, param_path=None):
        super(LigPose, self).__init__()
        assert args is not None or param_path is not None
        self.load_param(args, param_path)

    def forward(self, complex_graph):
        struct_pred = self.pred_struct(complex_graph)
        scr_pred = self.pred_screening(complex_graph)
        return struct_pred, scr_pred

    def load_param(self, args=None, param_path=None):
        if args is not None:
            self.ligpose_struct = LigPoseStruct(args)
            self.ligpose_scr = LigPoseScr(args)
            self.args = args
        elif param_path is not None:
            params = torch.load(param_path, map_location='cpu')
            self.ligpose_struct = LigPoseStruct(params['struct_args'])
            self.ligpose_struct.load_state_dict(params['struct_state_dict'], strict=True)
            self.ligpose_scr = LigPoseScr(params['screen_args'])
            self.ligpose_scr.load_state_dict(params['screen_state_dict'], strict=True)
            self.args = params['struct_args']
            del params

    def infer(self, complex_graph, pred_type=None):
        for p in pred_type:
            assert p in ['structure', 'screening']

        dic_pred = {}
        for p in pred_type:
            if p == 'structure':
                dic_pred['structure'] = self.pred_struct(complex_graph)
            elif p == 'screening':
                dic_pred['screening'] = self.pred_screening(complex_graph)

        return dic_pred

    def pred_struct(self, complex_graph):
        return self.ligpose_struct.infer(complex_graph.__deepcopy__(None))

    def pred_screening(self, complex_graph):
        return self.ligpose_scr.infer(complex_graph.__deepcopy__(None))




















