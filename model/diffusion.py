import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange

# 辅助函数：主要是在diff中的调度代码中，根据每个样本的时间步t从预先计算好的时间序列参数a中取值，并reshape为能够进行广播的形状
def extract(a, t, x_shape):
    device = t.device                                     #这里要先将t转换到gpu上面进行操作
    a = a.to(device)                                      #这里的a是一维张量，一般是长为T的时间序列参数，t是整数张量，形状为[B](batch size)，x_shape是目标加噪形状
    """
    Args:
        a: 形状[timesteps] 张量，主要表达该时间步噪音应该是什么样
        t: 形状[B] 张量，每个样本的时间步
        x_shape: 目标广播形状 [B, N, ...]
    Returns:
        [B, 1, 1, ...] 可广播张量
    """                                                                      #a是全局时间序列参数，长度T的张量
    out = a.gather(0, t).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))   #t的形状就是batch_size，提供了batch中每个样本的时间索引
    return out                                                               # 取出t个a中的元素放在第0维，然后构建了几个其他维度的元素方便进行广播
                                                                             #这里的计算方式是（1，）通过乘法复制len(x_shape) - 1次，并构建成元组的形式，在通过*解包传入作为reshape的参数
# 噪声调度器(timesteps, beta_start, beta_end, schedule_type)
class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule_type='linear'):
        self.timesteps = timesteps
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule_type == 'cosine':                                                             #余弦噪声调度，控制前向扩散时，每一步加多少噪声
            steps = timesteps + 1
            s = 0.008
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2        #((x / timesteps) + s) / (1 + s)，是对归一化时间做一个线性平移和缩放，使其位于[s/1+s，1]之间，(1 + s)是一个小偏移量，为了让cos曲线开头不全是1
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise NotImplementedError(f"Unknown schedule type: {schedule_type}")

        # 预计算常用值，这里就已经将t时间对应的noise其实是计算好了的，这些都是长度为timesteps的一维张量
        self.alphas = 1.0 - self.betas                                                  # 计算每一步的α值，α = 1 - β---->每一步保留信号的比例
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)                         # 计算累积的α值，α_t = α_0 * α_1 * ... * α_t-1---->原始信号保留的总比例
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)   # 这里是由于需要设置第一个是没有损失的信号，所以去掉最后一个，并在最开始的地方填充为1
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)                      # 累计√(α_t) sqrt_alphas_cumprod，用于加噪时的均值系数
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)      # 累计 √(1/1-α_t)，用于加噪时的方差系数（开方，采样时使用）
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)                          # 计算t步√(1 / α_t)，用于去噪时的系数均值系数
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)#去噪过程中需要的β的的计算   方差系数
        #alphas表示每一步保留信号的比例
        #sqrt_alphas_cumprod表示的是从t=0到t累计保留的信号比例
        #sqrt_alphas=sqrt(1-β)，cumprod就是累计乘积

#麻住了，这里这个先验后验证还有去噪过程还是要好好想想啊
    
    
    # 前向加噪获得x_t
    def q_sample(self, x_start, t, noise=None, mask=None):#x_start就是应该进行加噪的数据
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, x_start.shape).to(x_start.device)         #这里构建的维度是x_start.shape的个数
        sqrt_one_minus_alpha = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape).to(x_start.device)
        x_noised = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise                              #采样获得加噪数据
        if mask is not None:                                                                        #此处的mask是diff模型中的区域控制机制
            mask_expanded = mask.reshape(x_noised.shape[0], x_noised.shape[1], 1).expand_as(x_noised)#这里其实和x_noised.shape[2]一样，只是节省了内存
            x_noised = x_noised * mask_expanded + x_start * (~mask_expanded)                        #这里~是进行布尔取反操作，这样计算之后就会保留原本x_start数据，同时对mask中为true进行加噪处理
        return x_noised, noise

    # 反向去噪预测x0
    def predict_start_from_noise(self, x_t, t, noise):
        sqrt_recip = extract(self.sqrt_recip_alphas, t, x_t.shape)       
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_recip * x_t - sqrt_one_minus * noise / torch.sqrt(1.0 - self.alphas_cumprod[t])

    # 后验计算t-->t-1的分布
    def q_posterior(self, x_start, x_t, t):
        coef1 = extract(self.alphas_cumprod_prev / self.alphas_cumprod, t, x_t.shape)      
        coef2 = extract(self.betas / (1.0 - self.alphas_cumprod), t, x_t.shape)
        mean = coef1 * x_start + coef2 * x_t
        var = extract(self.posterior_variance, t, x_t.shape)
        return mean, var

#所以具体流程应该是反向预测x0，构建Xt-->Xt-1的后验分布，利用后验分布采样获得Xt-1

    # 单步采样p_sample
    @torch.no_grad()
    def p_sample(self, model, x_t, t, complex_graph, mask=None):
        complex_graph.coor = x_t
        complex_graph.timestep = t
        pred = model(complex_graph, return_graph=True)                  #这里是每次采样都调用这个NoiseScheduler函数的forward部分
        noise_pred = pred.noise_pred                                    #从模型中输出预测的噪声
        x0_pred = self.predict_start_from_noise(x_t, t, noise_pred)     #通过输出预测的噪声来预测x0
        mean, var = self.q_posterior(x0_pred, x_t, t)                   #输出后验分布的均值和方差，为x-1预测做准备
        if (t == 0).all():                                              #反向采样终止条件，如果已达到x0则直接返回mean
            return mean
        noise = torch.randn_like(x_t)
        x_prev = mean + torch.sqrt(var) * noise                         #Xt是作为条件信息，来参与对Xt-1的预测
        if mask is not None:
            mask_expanded = mask.reshape(x_t.shape[0], x_t.shape[1], 1).expand_as(x_prev)
            x_prev = x_prev * mask_expanded + x_t * (~mask_expanded)
        return x_prev

    # 逐步采样p_sample_loop
    @torch.no_grad()
    def p_sample_loop(self, model, complex_graph, timesteps=None, return_all=False, mask=None):
        device = next(model.parameters()).device
        batch_size = complex_graph.coor.shape[0]
        T = self.timesteps if timesteps is None else timesteps
        x = torch.randn_like(complex_graph.coor)  # 这个x不是complex_graph.x啊，那我之前设置的layers可能还是有问题的,我应该是取omplex_graph.coor这个的
        if mask is not None:
            mask_expanded = mask.reshape(x.shape[0], x.shape[1], 1).expand_as(x)
            x = x * mask_expanded + complex_graph.coor * (~mask_expanded)
        intermediates = [x]
        for t_idx in reversed(range(T)):
            t_tensor = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)#这里是构建了一系列长度为batch_size的张量，里面的值都是t_idx的张量
            x = self.p_sample(model, x, t_tensor, complex_graph, mask=mask)
            intermediates.append(x)
        return intermediates if return_all else x

class DiffusionWrapper(nn.Module):
    def __init__(self, base_model, noise_pred_net=None, timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule_type='linear'):
        super().__init__()
        self.base_model = base_model
        self.scheduler = NoiseScheduler(timesteps, beta_start, beta_end, schedule_type)
        
        hidden_dim = getattr(base_model, 'node_hidden', 768)
        if noise_pred_net is None:
            # 默认 MLP 预测噪声
            self.noise_pred_net = nn.Sequential(
                nn.LazyLinear(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3)
            )
        else:
            self.noise_pred_net = noise_pred_net

    def forward(self, complex_graph, t=None,return_graph=True,struct_mode=False):#return_graph=True按道理可以弃用的
        """
        前向调用基础模型 + 噪声预测
        Args:
            complex_graph: 复合图数据
            t: 时间步 [B]
            return_graph: 是否返回 complex_graph
        """
        if t is not None:
            complex_graph.timestep = t
        
        # 调用lipose_struct
        complex_graph = self.base_model(complex_graph, return_graph=True, time_step=t)   #这一步通过调用这个函数将complex_graph变成了tuple的形式了，所以丧失掉.protein_node_feature_init（已经不是pyg中的data形式了）
        

        
        cycle_i = complex_graph.cycle_i
        p_x_masked = rearrange(complex_graph.x, 'b n d -> (b n) d')[
            complex_graph.p_x_mask_bool_cycle[cycle_i].reshape(-1)]
        l_x_masked = rearrange(complex_graph.x, 'b n d -> (b n) d')[
            complex_graph.l_x_mask_bool_cycle[cycle_i].reshape(-1)]
        edge_masked = rearrange(complex_graph.edge_attr, 'b i j d -> (b i j) d')[
            complex_graph.edge_mask_bool_cycle[cycle_i].reshape(-1)]
    
        #将用于计算struct_loss的数据保存在Data对象中
        complex_graph.p_x_pred_1 = self.base_model.p_x_pretrain_1(p_x_masked)       #p_x_pred_1这说是对蛋白质节点的预训练输出？？？这个明天再考虑
        complex_graph.p_x_pred_2 = self.base_model.p_x_pretrain_2(p_x_masked)       #通过这种方式调用base_model中定义的函数
        complex_graph.l_x_pred   = self.base_model.l_x_pretrain(l_x_masked)
        complex_graph.edge_pred  = self.base_model.edge_pretrain(edge_masked)
        
        
        #这里没有什么意义，是开始识别不到complex的时候写的一点东西                                                                              
        # # 构建节点特征并预测噪声
        # from utils.data_utils import batch_index_select                             #也就是说构建节点的时候一定要有这个batch_index_select?,我在diffusion_layer中写了还不行吗？
        # if not hasattr(complex_graph, 'x') or complex_graph.x is None:
        #     print("complex_graph type:", type(complex_graph))
        #     print("complex_graph content:", complex_graph)
        #     embed_protein_node_feature_init = self.base_model.main_net.protein_embed(complex_graph.protein_node_feature_init)  #这里要加入一个base_model，因为调用的是其内部的一个函数，是不是有更好的解决方法呢？
        #     embed_ligand_node_feature_init = self.base_model.main_net.ligand_embed(complex_graph.ligand_node_feature_init)
        #     middle_pad_embed_node_feature_init = torch.cat(
        #         [embed_protein_node_feature_init, embed_ligand_node_feature_init], dim=-2)
        #     complex_graph.embed_node_feature_init = batch_index_select(
        #         middle_pad_embed_node_feature_init, complex_graph.idx_remove_middle_pad)
        #     complex_graph.x = complex_graph.embed_node_feature_init
        # if hasattr(complex_graph, 'x') and complex_graph.x is not None:             #实际走的是这条路，但是这里使用的是全部节点，这个complex_graph.x可能还是要好好再看一下
        #     complex_graph.noise_pred = self.noise_pred_net(complex_graph.x)
        # else:
        if hasattr(complex_graph, 'coor'):
            
            complex_graph.noise_pred = self.noise_pred_net(complex_graph.coor)
            
        else:       
            raise RuntimeError("complex_graph.node_mask_after_sampling 不存在，无法预测噪声")
        
        if struct_mode:
            return (complex_graph.coor_hidden, complex_graph.aff_pred,complex_graph.p_x_pred_1, complex_graph.p_x_pred_2,complex_graph.l_x_pred, complex_graph.edge_pred) #按道理走noise预测的时候应该是使用这个
        
        return complex_graph
        
        
        #包装器中返回的内容应该是用于计算noise的

            
    @torch.no_grad()
    def sample(self, complex_graph, timesteps=None, return_all=False, mask=None):
        """
        从纯噪声开始逐步反向采样
        """
        T = timesteps if timesteps is not None else self.scheduler.timesteps
        return self.scheduler.p_sample_loop(self, complex_graph, timesteps=T, return_all=return_all, mask=mask)#这里的self就是p_sample中需要传入的model
    
# 这里是一个Diffusion模型的包装器，用于封装现有的LigPose模型，增加噪声预测功能


