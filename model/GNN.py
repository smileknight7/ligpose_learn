

import numpy as np

import torch
import torch.nn.functional as F
from einops import rearrange

#将输入数据转换为向量
def make_embed(input_channel, output_channel):
    return torch.nn.Sequential(
        torch.nn.Linear(input_channel, output_channel),
        torch.nn.LeakyReLU())

#实现门残差链接的神经网络模块
#门控循环单元GRU（使得需要注意的部分————更新门，可以忘记的部分————重置门）
#GateResidue类经常应用于图神经网络，允许模型在多层次信息传递过程中保持重要的结构信息，避免过度平滑
class GateResidue(torch.nn.Module):
    def __init__(self, hidden, gate_flag=True):
        super(GateResidue, self).__init__()
        self.gate_flag = gate_flag
        if self.gate_flag:
            self.gate = torch.nn.Linear(hidden * 3, hidden)

    def forward(self, x, res):
        if self.gate_flag:
            g = self.gate(torch.cat((x, res, x - res), dim=-1)).sigmoid()
            return x * g + res  # res * (1 - g)
        else:
            return x + res
#实现特征的加权融合，此处x * g + res与 x * g + res * (1 - g)等价是因为没有实现对res进行加权
#此处multi虽然是1但是，此处添加这个参数可以保持变换-激活-正则化-变换这个处理流程，这里通过引入linear实现了非线性变换
#droupout可以提高泛化能力，防止过拟合（通过在训练过程中将一部分神经晕啊的输出随机设置为0）
class FeedForward(torch.nn.Module):
    def __init__(self, hidden, dropout, multi=1):
        super(FeedForward, self).__init__()
        self.FF = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden * multi),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden * multi, hidden)
        )

    def forward(self, x):
        return self.FF(x)

#GateNormFeedForward结合了前馈网络、门控残差连接和层归一化三个组件
class GateNormFeedForward(torch.nn.Module):
    def __init__(self, hidden, dropout):
        super(GateNormFeedForward, self).__init__()
        self.FF = FeedForward(hidden, dropout)
        self.gate = GateResidue(hidden)
        self.norm = torch.nn.LayerNorm(hidden)

    def forward(self, x):
        x_shortcut = x
        x = self.FF(x)
        x = self.gate(x, x_shortcut)
        x = self.norm(x)
        return x

#对坐标向量进行归一化处理，确保具有相同的长度
class CoorNorm(torch.nn.Module):
    def __init__(self):
        super(CoorNorm, self).__init__()
        # self.scale = torch.nn.Parameter(torch.ones(1)/ 1e+3)

    def forward(self, rel_coor):
        norm = rel_coor.norm(p=2, dim=-1, keepdim=True)#计算欧几里得范数
        norm = torch.where(norm == 0, norm + 1e+8, norm)  # for norm=0 (rel_coor with same atoms)两个原子位置相同是替换为一个很大数
        normed_rel_coor = rel_coor / norm.clamp(min=1e-8)#相对坐标除以范数来进行归一化得到单位向量
        return normed_rel_coor  # * self.scale

#这里实现了一个径向基函数（RBF）编码器，主要用于将距离信息转换为高维特征表示（是分子建模中的一种常见技术）
#其中要使用高斯核也称之为径向基数，是利用核技巧将隐式的 将数据转换到高维空间，而无需计算高维坐标（距离值相近核值接近1否则接近0）
class DistRBF(torch.nn.Module):
    def __init__(self, start=0., stop=2., num_gaussians=50):
        super(DistRBF, self).__init__()
        self.stop = stop
        self.offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (self.offset[1] - self.offset[0]).item() ** 2#这行代码是在计算系数β，完整公式f(x) = exp(-β * (x - μ)²)
        # self.scale = torch.nn.Parameter(torch.ones(1))
#β值增大，则和核函数越尖锐，模型对近距离样本更敏感，β数降低时，相反
#这里没有使用经验值，而是使用 -0.5 / (距离间隔)² 确保相邻高斯核有约63%的重叠作为参数
    def forward(self, dist):
        encode_dist = dist
        encode_dist = encode_dist.clamp_max(self.stop)
        encode_dist = encode_dist.unsqueeze(dim=-1) - self.offset.to(dist.device)
        encode_dist = torch.exp(self.coeff * torch.pow(encode_dist, 2))
        # encode_dist = torch.cat([dist.unsqueeze(dim=-1) * self.scale,
        #                          torch.exp(self.coeff * torch.pow(encode_dist, 2))],
        #                         dim=-1)
        return encode_dist

#SE3EquivariantAttention是实现等变注意力机制的模块
#等变性保持：使模型对3D空间中的旋转和平移变换具有不变性，保证分子坐标旋转后预测结果一致
#多头注意力准备：设置n_head注意头，每个头有head_hidden维特征
class SE3EquivariantAttention(torch.nn.Module):
    def __init__(self,
                 node_hidden,
                 edge_hidden,
                 n_head,
                 head_hidden,
                 dropout,
                 only_coor_out=False):
        super(SE3EquivariantAttention, self).__init__()
        self.only_coor_out = only_coor_out
        self.n_head = n_head
        self.head_hidden = head_hidden
        self.sqrt_head_hidden = np.sqrt(self.head_hidden)
#节点特征转换为q与k（线性输出分为四部分）
#双向消息传递左值到右值的方向和右值到左值都要进行考虑
#边缘特征与距离融合处理器，这里将边特征与距离编码直接相拼，进行线性变换随后使用LeakyReLU引入非线性（为什么要引入非线性？）
        self.lin_qk = torch.nn.Linear(node_hidden, n_head * head_hidden * 4)
        RBF_num_gaussians = edge_hidden
        self.dist_scale = DistRBF(num_gaussians=RBF_num_gaussians)
        self.edge_coor_to_att = torch.nn.Sequential(
            torch.nn.Linear(edge_hidden + RBF_num_gaussians, n_head * head_hidden * 2),
            torch.nn.LeakyReLU())
#条件性节点特征和边特征变换，head_hidden * 2是为了处理双向信息传递。
        if only_coor_out == False:
            self.lin_node_out = torch.nn.Linear(n_head * head_hidden * 2, node_hidden)
            self.lin_edge_out = torch.nn.Linear(n_head * head_hidden, edge_hidden)
#这里定义了一个坐标投影网络用于将注意力特征转换为坐标更新的权重？？？？？这个还要再看看
        self.coor_out = torch.nn.Sequential(torch.nn.Linear(n_head * head_hidden, n_head * head_hidden),
                                            torch.nn.LeakyReLU(),
                                            torch.nn.Linear(n_head * head_hidden, 1))
#辅助性组件
        self.coor_norm = CoorNorm()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, complex_graph):
        x = complex_graph.x
        coor = complex_graph.coor
        edge_attr = complex_graph.edge_attr
#等变注意力机制中的关键特征预处理
#相对坐标计算（使用重排和广播减法，计算所有原子对之间的相对位置向量）
#距离编码，将欧氏距离通过径向基函数转换为高维特征表示
        rel_coor = rearrange(coor, 'b i d -> b i () d') - rearrange(coor, 'b i d -> b () i d')
        rel_dist = self.dist_scale(rel_coor.norm(p=2, dim=-1))
#将线性变换后的节点特征分割为4部分，然后重新组织为多头格式，方便计算注意力
#边特征处理，将特征重新组织为多头形式，方便计算注意力
        l_m, r_m, l_v, r_v = self.lin_qk(x).chunk(4, dim=-1)
        l_m, r_m, l_v, r_v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.n_head), (l_m, r_m, l_v, r_v))
        edge_coor = torch.cat((edge_attr, rel_dist), dim=-1)
        e_m, e_v = self.edge_coor_to_att(edge_coor).chunk(2, dim=-1)
        e_m, e_v = map(lambda t: rearrange(t, 'b i j (h d) -> b i j h d', h=self.n_head), (e_m, e_v))

# einops 库中的重排模式表示法b - batch 批次维度
        # i - 源节点索引 (source nodes)
        # j - 目标节点索引 (target nodes)
        # h - 注意力头数 (attention heads)
        # d - 每个头的特征维度 (feature dimension per head)
        # () - 新增的空维度，用于后续广播计算

        # do attention
        l_m = rearrange(l_m, 'b i h d -> b i () h d')
        r_m = rearrange(r_m, 'b j h d -> b () j h d')
        att = l_m * r_m * e_m
        att_sum = att.sum(dim=-1) / self.sqrt_head_hidden
        att_mask = (rearrange(complex_graph.edge_mask_after_sampling, 'b i j -> b i j ()') == 0)
        att_sum.masked_fill_(att_mask, -torch.finfo(att_sum.dtype).max)
        att_softmax = att_sum.softmax(dim=-2)

        att_v = rearrange(l_v, 'b i h d -> b i () h d') * rearrange(r_v, 'b j h d -> b () j h d') * e_v
        att_v = att_softmax.unsqueeze(dim=-1) * att_v

        # for coor
        coor_att = self.coor_out(rearrange(att_v, 'b i j h d -> b i j (h d)')).squeeze(dim=-1)
        coor_att = coor_att * complex_graph.edge_mask_after_sampling
        rel_coor = self.coor_norm(rel_coor)  # to avoid coordinate exploding, use CoorNorm
        # rel_coor = self.dropout(rel_coor)
        coor_out = torch.einsum('b i j, b i j c -> b i c', coor_att, rel_coor) / (
                    complex_graph.edge_mask_after_sampling.sum(dim=-1, keepdims=True) + 1e-7)
        coor_out = coor_out * rearrange(complex_graph.flex_coor_mask_after_sampling,
                                        'b i -> b i ()')  # only update ligand coor
#只进行了配体的坐标更新而蛋白质结构坐标保持
#注意力机制的输出处理，生成节点特征，边特征与坐标更新
        if not self.only_coor_out:
            # for node
            node_att = att_softmax
            node_att = self.dropout(node_att)
            node_out = torch.einsum('b i j h, b j h d -> b i h d', node_att, torch.cat([l_v, r_v], dim=-1))
            node_out = rearrange(node_out, 'b n h d -> b n (h d)')
            node_out = self.lin_node_out(node_out)
            # for edge
            edge_att = att_softmax
            edge_att = self.dropout(edge_att)
            edge_out = self.lin_edge_out(rearrange(e_v * edge_att.unsqueeze(dim=-1), 'b i j h d -> b i j (h d)'))
        else:
            node_out, edge_out = None, None

        return node_out, edge_out, coor_out

#封装注意力机制的结果，为其他模块进行调用
class GateNormAttention(torch.nn.Module):
    def __init__(self,
                 node_hidden,
                 edge_hidden,
                 n_head,
                 head_hidden,
                 dropout,
                 only_coor_out=False,
                 ):
        super(GateNormAttention, self).__init__()
        self.only_coor_out = only_coor_out

        self.att_layer_i = SE3EquivariantAttention(
            node_hidden,
            edge_hidden,
            n_head,
            head_hidden,
            dropout,
            only_coor_out
        )

        if only_coor_out == False:
            self.gate_node_i = GateResidue(node_hidden)
            self.gate_edge_i = GateResidue(edge_hidden)
            self.norm_node_i = torch.nn.LayerNorm(node_hidden)
            self.norm_edge_i = torch.nn.LayerNorm(edge_hidden)

#前向传播用于配体化合物的坐标更新
    def forward(self, complex_graph):
        # att
        x_shortcut = complex_graph.x
        edge_attr_shortcut = complex_graph.edge_attr
        node_out_i, edge_out_i, coor_out_i = self.att_layer_i(complex_graph)
#使用注意力机制的输出更新节点特征和边特征
        if self.only_coor_out == False:
            complex_graph.x = self.gate_node_i(node_out_i, x_shortcut)
            complex_graph.edge_attr = self.gate_edge_i(edge_out_i, edge_attr_shortcut)
            complex_graph.x = self.norm_node_i(complex_graph.x)
            complex_graph.edge_attr = self.norm_edge_i(complex_graph.edge_attr)

        # coor
        complex_graph.coor = complex_graph.coor + coor_out_i

        return complex_graph