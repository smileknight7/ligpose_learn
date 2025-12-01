import torch
import numpy as np
import torch.nn as nn
import torch_scatter as scatter
from diffusionpart.utils import *



# compute distance and coord difference
def coord2diff(x, edge_index, norm_constant=1):                 
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)       
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff                                   

# 这里应该是这样参数化？
# in_node_nf, context_node_nf,
#                  n_dims, hidden_nf=64,                      
#                  act_fn="silu", n_layers=4, attention=False,
#                  condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
#                  inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'

class EGNN_dynamics(nn.Module):
    def __init__(self, args):
        super(EGNN_dynamics,self).__init__()

        self.egnn = EGNN(
            in_node_nf=args.in_node_nf, in_edge_nf=1,
            hidden_nf=args.hidden_nf, act_fn=args.act_fn,
            n_layers=args.n_layers, attention=args.attention, tanh=args.tanh, norm_constant=args.norm_constant,
            inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
            normalization_factor=args.normalization_factor,
            aggregation_method=args.aggregation_method)
        
        self.in_node_nf = args.in_node_nf
        # self.device = device
        self.n_dims = args.n_dims            # 3 cfg文件中配置过了
        self._edges_dict = {}
        self.condition_time = args.condition_time

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):                  # 
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, node_mask, edge_mask, context, mol_shape=None):
        bs, n_nodes, dims = xh.shape            # 这里最开始的时候使用的是 t, x, node_mask, edge_mask, context, mol_shape
        #get the node 
        h_dims = dims - self.n_dims             
        # to do put the edge into the dataloader part. 
        edges = self.get_adj_matrix(n_nodes, bs)        # 构建节点的全连接矩阵
        edges = [x.to(xh.device) for x in edges]                           # 但是节点序号映射到batch上方便区分batch内不同dataset的节点

        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask     
        x = xh[:, 0:self.n_dims].clone()                                   # self.n_dims = 3
        if h_dims == 0:                                     
            h = torch.ones(bs*n_nodes, 1).to(xh.device)              # 如果没有h就使用全1来代替
        else:
            h = xh[:, self.n_dims:].clone()

        if self.condition_time:                                            # True
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)

# put in egnn                                                                           
        h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask) 
        if mol_shape is not None:
            x_final = x_final.view(bs, n_nodes, -1)     
            x = x.view(bs, n_nodes, -1)
            x_final[:, mol_shape:, :] = x[:, mol_shape:, :]
            x_final = x_final.view(bs*n_nodes, -1)
            x = x.view(bs*n_nodes, -1)
        vel = (x_final - x) * node_mask     

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]               # 这样切片是排除了最后一列（时间维度）

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if h_dims == 0:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            return torch.cat([vel, h_final], dim=2)


# 构建batch中的全连接矩阵
    def get_adj_matrix(self, n_nodes, batch_size):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):                          # batch_size=2  n_nodes=3
                            rows.append(i + batch_idx * n_nodes)          # rows = [0,0,0,1,1,1,2,2,2, 3,3,3,4,4,4,5,5,5]   
                            cols.append(j + batch_idx * n_nodes)          # cols = [0,1,2,0,1,2,0,1,2, 3,4,5,3,4,5,3,4,5]      
                edges = [torch.LongTensor(rows),                                           
                         torch.LongTensor(cols)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size)
        

class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method, # input,output和hidden都采用self.hidden_nf
                 edges_in_d=0, nodes_att_dim=0, act_fn="silu", attention=False):         # attention = True
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        if act_fn == "silu":
            act_fn = nn.SiLU()
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method            # SUM
        self.attention = attention                              # True      

        # 这部分直接将 h 和edge_attr 拼接进行mlp --->构建边特征                              
        self.edge_mlp = nn.Sequential(                      
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        # 这部分将
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),         
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.attention:                                                  # node_model和edge_model的区别 ----->   这里计算了注意力权重
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())


# 计算i <----- j  1 attentin后的massage(边特征) 2 未进行attention的massage
    def edge_model(self, target, source, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([target, source], dim=1)                
        else:
            out = torch.cat([target, source, edge_attr], dim=1)     # h[row], h[col]拼接这两个节点特征,和边特征
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)                                     # 计算注意力权重（将之前融合了的节点特征变为一个值）
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij


# 计算i <----- j 聚合所有j传递过来的massage更新i节点特征
    def node_model(self, h, edge_index, edge_attr, node_attr):
        row, col = edge_index
        
        if self.aggregation_method == 'sum':
             agg = scatter(edge_attr, row, dim=0, dim_size=h.size(0), reduce='sum')
             agg = agg / self.normalization_factor
        elif self.aggregation_method == 'mean':
             agg = scatter(edge_attr, row, dim=0, dim_size=h.size(0), reduce='mean')

        if node_attr is not None:                                           # None
            agg = torch.cat([h, agg, node_attr], dim=1)
        else:
            agg = torch.cat([h, agg], dim=1)                        # 应该是使用这个，没有节点特征吧，h就已经是节点特征了啊
        out = h + self.node_mlp(agg)
        
        return out, agg


# h edge_attr update 部分
    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)      # out计算注意力后的结果 mij是没有乘注意力
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)               # 这里和x拼接完居然算是h的内容
        if node_mask is not None:
            h = h * node_mask
        return h, mij

# x update部分
class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)              # 初始化权重
        self.coord_mlp = nn.Sequential(                                             # 走了两层mlp
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)        # 检查一下edge_attr是不是只有一个维度
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range       # coords_range相当于是缩放因子
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)                       # 使用这个 coord_diff表示方向（这里是用节点特征和边特征计算的mlp作为偏移量）
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = scatter(trans, row, dim=0, dim_size=coord.size(0), reduce=self.aggregation_method)
                                                                                    # 这里是将node特征和edge特征做mlp构建偏移再聚合到pos上

        coord = coord + agg
        return coord

    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None, node_mask=None, edge_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord



# 等变更新部分（先计算node和edge部分的多次更新，随后再计算pos的更新一次）


class EquivariantBlock(nn.Module):                                                  # 进行2次数GCL一次EquivariantUpdate
    def __init__(self, hidden_nf, edge_feat_nf=2, act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=30, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum'):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        #self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)                            # EGNN中控制坐标更新的尺度范围
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant                                         # EGNN中控制坐标更新的数值稳定性
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor                           # 信号尺度和稳定性控制
        self.aggregation_method = aggregation_method                               # sum

        for i in range(0, n_layers):                                # GCL是图卷积层   ---> 经过n_layers个GCL层在过一个等变更新
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              act_fn=act_fn, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method))
        
        
                                                                    # self.to(self.device)
# 这里是先从h ---> x 做了更新，后面又从h,x ---> x做了更新等变更新
    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)        # 这里是将距离和原始距离(边特征)作为新的边特征输入
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x




class EGNN(nn.Module):               # 更新3次
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, act_fn="silu", n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=30, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        # self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)            # 将整体坐标在模型的层数之间进行平均分配
        self.norm_diff = norm_diff                                          # True
        self.normalization_factor = normalization_factor                    # 100   归一化因子（控制数值尺度）
        self.aggregation_method = aggregation_method                        # sum

        if act_fn == "silu":
            act_fn = nn.SiLU()

        if sin_embedding:                                                   # False
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)              # hidden_nf=64
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=self.coords_range_layer, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method))
        # self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        distances, _ = coord2diff(x, edge_index)           # x 部分获取坐标差并进行embedding
        if self.sin_embedding is not None:                              # False
            distances = self.sin_embedding(distances)
        h = self.embedding(h)                                           # h 部分直接进行embedding     
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask, edge_attr=distances)

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x
