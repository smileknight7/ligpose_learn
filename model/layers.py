import random

import pandas as pd
import torch
import torch.nn.functional as F

from model.GNN import *
from utils.data_utils import batch_index_select




#继承一个来自nn.module的函数并声明其构造函数
class UpdateBlock(torch.nn.Module):
    def __init__(self,
                 n_block,                           #此处的n_block指堆叠次数，也就是说UpdateBlock会有多少个重复单元
                 node_hidden,
                 edge_hidden,
                 n_head,
                 head_hidden,                       #node_hidden//n_head----->768//8=96
                 dropout,
                 ):
        #初始化父类函数，并构建构造函数
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
        #此处的FF是指前馈神经网(由全连接，非线性以及dropout构成，负责对每个节点或者边特征进行变换或者增强)

        self.last_update = GateNormAttention(
            node_hidden,
            edge_hidden,
            n_head,
            head_hidden,
            dropout,
            only_coor_out=True,
        )
#整体上，这个更新模块是由一层注意力，一层点节点层FF，一层边节点层FF
#Message Passing (Attention) + Point-wise Feature Update (FF)
#这种结构是参考transformer实现的，注意力层主要是用于节点和边的信息交互，而前馈神经网络层是用来进行节点和边的更新的
        
#FF就是前馈神经网络，也就是Feedforward层
#最后通过一层注意力层进行输出
        



#前向传播
    def forward(self, complex_graph):
        coor_hidden = []
        for i in range(self.n_block):
            complex_graph = self.att_layers[i](complex_graph)
            complex_graph.x = self.node_FF_layers[i](complex_graph.x)
            complex_graph.edge_attr = self.edge_FF_layers[i](complex_graph.edge_attr)
            
            coor_hidden.append(complex_graph.coor)
        complex_graph = self.last_update(complex_graph)
        coor_hidden.append(complex_graph.coor)
        complex_graph.coor_hidden = torch.stack(coor_hidden, dim=0)      #这里会构建一个(n_block+1, batch, node, 3)的张量----->coor_hidden
        return complex_graph


#这里有一点点没看懂，为什么对complex_graph进行了堆叠并赋值，很奇怪，还搞了返回值

class LigPoseBase(torch.nn.Module):
    def __init__(self, args):
        super(LigPoseBase, self).__init__()
        self.n_cycle = args.n_cycle
        # embed
        self.protein_embed = make_embed(args.protein_input_channel + 1, args.node_hidden)   #79----->768
        self.ligand_embed = make_embed(args.ligand_input_channel + 1, args.node_hidden)     #45----->768
        self.edge_embed = make_embed(args.edge_input_channel + 2, args.edge_hidden)         #6----->384

        # cycle
        self.cycle_node_gate = GateResidue(args.node_hidden) #------768
        self.cycle_edge_gate = GateResidue(args.edge_hidden) #------384
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
        # embed   全图部分
        complex_graph = complex_graph
        embed_protein_node_feature_init = self.protein_embed(complex_graph.protein_node_feature_init)
        embed_ligand_node_feature_init = self.ligand_embed(complex_graph.ligand_node_feature_init)            #---->(batch,num_atoms, 768)应该是这个样子的
        middle_pad_embed_node_feature_init = torch.cat(
            [embed_protein_node_feature_init, embed_ligand_node_feature_init], dim=-2)
        
        complex_graph.embed_node_feature_init = batch_index_select(middle_pad_embed_node_feature_init,      # 按照有效节点进行重排序（有意义的protein（before_sample）和ligand在前面）
                                                                   complex_graph.idx_remove_middle_pad)   
        # 构建complex的embedding索引 embed_node_feature_init
        
        
        # cycle
        if self.training:  # MC-like style
            cycle_num = random.sample(range(1, self.n_cycle + 1), 1)[0]                                     #----->n_cycle=4
            with torch.no_grad():                                                                           # 这里引入了蒙特卡洛随机采样，随机决定运行几轮cycle，前面的随机采样是应用于推理的，最后一次随机采样才用于训练过程
                for cycle_i in range(cycle_num - 1):                                                        # 假设取到了cycle_num = 3
                    complex_graph = self.run_cycle(complex_graph, cycle_i)            # 这里会依次用到cycle_i=0,1,2 进行三次次更新, 只有等于2的时候记录grad
            complex_graph = self.run_cycle(complex_graph, cycle_num - 1)                
        else:                                                                       #推理模式
            for cycle_i in range(self.n_cycle):
                complex_graph = self.run_cycle(complex_graph, cycle_i)

        return complex_graph


# 采样部分数据对目标core_atom和ligand的消息传递和融合(这里是要按照cycle选定原子)                                                    # 这里要注意这个n_loc不是连续的中间又断开)
    def run_cycle(self, complex_graph, cycle_i):                                                                        # index = (n_cycle, B, n_loc)
        x_cycle = batch_index_select(complex_graph.embed_node_feature_init, complex_graph.node_sampling_loc[cycle_i])    # (B, n_loc, F)取出当前cycle的节点特征
        edge_attr_cycle = self.edge_embed(complex_graph.edge_feature_init_cycle[cycle_i])                                      # (B, n_loc, n_loc, F)                            之前计算的张量形状就是(cycle_batch, N, N, edge_dim)
        coor_init_cycle = batch_index_select(complex_graph.coor_init, complex_graph.node_sampling_loc[cycle_i])          # (B, n_loc, 3)

        if cycle_i > 0:                                                                                                        # 最开始cycle_i 不走这个，直接走定义complex_graph.x这部分的内容                                                                                                     #* rearrange这里是新增一个维度，用于通过广播机制进行筛选节点，mask的形状变成 (B,N,1),这一段是要
            x_cycle = x_cycle * rearrange(complex_graph.node_cycling_mask == 0, 'b i -> b i ()') + \
                      self.cycle_node_gate(x_cycle, complex_graph.x) * rearrange(complex_graph.node_cycling_mask,       # 前半部分：布尔掩码（== 0是对掩码进行了转置）保留mask=0部分，其余部分归零                                                                                                                              
                                                                                 'b i -> b i ()')                      # 后半部分：布尔掩码（mask=1)对core_atom和ligand部分进行更新(引入GateResidue)，通过gate控制更新部分
            edge_attr_cycle = edge_attr_cycle * rearrange(complex_graph.edge_cycling_mask == 0, 'b i j -> b i j ()') + \
                              self.cycle_edge_gate(edge_attr_cycle, complex_graph.edge_attr) * rearrange(        
                complex_graph.edge_cycling_mask, 'b i j -> b i j ()')

            # map updated protein and ligand coor to original coor (for noise protein coor)
            # coor_before_sampling_flat = rearrange(complex_graph.coor_before_sampling, 'b n c -> (b n) c')
            # coor_last_update_after_sampling = rearrange(complex_graph.coor, 'b n c -> (b n) c')
            # coor_before_sampling_flat[complex_graph.node_sampling_loc_flat[cycle_i]] = coor_last_update_after_sampling
            # complex_graph.coor_before_sampling = rearrange(coor_before_sampling_flat, '(b n) c -> b n c', b=x_cycle.size(0))
            coor_init_cycle = coor_init_cycle * rearrange(complex_graph.node_cycling_mask == 0, 'b i -> b i ()') + \
                              complex_graph.coor * rearrange(complex_graph.node_cycling_mask, 'b i -> b i ()')  #坐标这里没有使用门控进行更新，是直接使用了硬替换

#归一化                                                              # complex_graph.x是上一轮的， x_cycle是当前本轮的  
        complex_graph.x = self.cycle_node_norm(x_cycle)             # x_cycle--->c_cycle状态下的node状态                                                           #这个complex_graph.x，是在这里进行定义的，每次使用的都是上次定义的内容
        complex_graph.edge_attr = self.cycle_edge_norm(edge_attr_cycle)
        complex_graph.coor = coor_init_cycle                        # cycle=0的时候是初始坐标coor_init(select)，cycle>0的时候是上次更新的坐标

        complex_graph.cycle_i = cycle_i
        complex_graph.flex_coor_mask_after_sampling = complex_graph.flex_coor_mask_cycle[cycle_i]       # (n_cycle, batch_list, node_sampling_loc）相当于是从这里取出当前循环层
        #flex_coor_mask_after_sampling就是标记当前cycle下的ligand节点位置
        complex_graph = self.complex_net(complex_graph)             # 这里要使用到上面complex.x
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

    def forward(self, complex_graph, return_graph=False):        # complex_graph = dic_data
        complex_graph = self.main_net(complex_graph)

        # for aff
        node_mean_pooling = torch.einsum('b n d -> b d',                                                           
                                         complex_graph.x * rearrange(complex_graph.node_mask_after_sampling,   #    
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

    def load_param(self, args=True, param_path=None):   #这里比较重要好吧，可能还是要再考虑一下是怎么传进来参数的
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
            self.args = params['struct_args']#s['struct_args']这里之前是使用这个字段
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




















