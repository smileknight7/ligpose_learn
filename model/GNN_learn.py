import numpy as np

import torch
import torch.nn.functional as F
from einops import rearrange

#embed

def make_embed(input_channel, output_channel):
    return torch.nn.Sequential(
        torch.nn.Linear(input_channel, output_channel),
        torch.nn.LeakyReLU())
#这里建立了embed函数，输入通道数和输出通道数，返回一个线性层和一个LeakyReLU激活函数的组合

#gate

class GateResidue(torch.nn.Module):
    def __init__(self, hidden, gate_flag=True):
        super(GateResidue, self).__init__()
        self.gate_flag = gate_flag
        if self.gate_flag:
            self.gate = torch.nn.Linear(hidden * 3, hidden)
#这里是进行了判断，gate_flag为True时，会构建一个全连接层，其输入维度为hidden * 3，输出维度为 hidden。
#会将其注册为模块的子模块，Pytorch会自动管理它的参数
#实例化GateResidue的时候可以来决定是否设置为True
        
    def forward(self, x, res):
        if self.gate_flag:
            g = self.gate(torch.cat((x, res, x - res), dim=-1)).sigmoid()
            return x * g + res  # res * (1 - g)
        else:
            return x + res
#此处是将torch.cat((x, res, x - res)作为输入利用门控生成一个动态的门控值最终对x进行加权    x * g + res 



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
#这段代码是结合了之前的FeedForward，并添加了门控和层归一化
#x_shortcut是为了实现残差链接，这里的self_cut其实是原本的X而到这一步的X是已经通过了self.ff的变量
    
#normalization
    
class CoorNorm(torch.nn.Module):
    def __init__(self):
        super(CoorNorm, self).__init__()
        # self.scale = torch.nn.Parameter(torch.ones(1)/ 1e+3)

    def forward(self, rel_coor):
        norm = rel_coor.norm(p=2, dim=-1, keepdim=True)
        norm = torch.where(norm == 0, norm + 1e+8, norm)  # for norm=0 (rel_coor with same atoms)
        normed_rel_coor = rel_coor / norm.clamp(min=1e-8)
        return normed_rel_coor  # * self.scale
#这个coornorm函数是用来对输入的相对坐标进行归一化的，
#归一化的作用，1消除尺度差异，将所有向量的长度标准化为1，2提高数值稳定性（较大的数值可能会导致溢出），3增强模型的泛化能力，使模型更专注于学习重要的模式
        #norm = rel_coor.norm(p=2, dim=-1, keepdim=True)是对已经获得的相对坐标计算L2范数（坐标距离差值）
        #p=2：指定计算 L2 范数。
        #dim=-1：沿最后一个维度计算范数，通常是坐标的分量（如 x、y、z）。
        #keepdim=True：保留计算后的维度，确保输出张量的形状与输入兼容。

        #norm = torch.where(norm == 0, norm + 1e+8, norm)是对坐标距离差值进行归一化，
        #这里要将norm + 1e+8设置为最大数值是因为后面要对其进行除法，被除掉之后，得到的数会是一个很小的数字