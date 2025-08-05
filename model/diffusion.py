import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#time embedding for diffusion models
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# 噪声调度器
class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule_type='linear'):
        """
        噪声调度器，负责管理前向过程（加噪）和反向过程（去噪）中的噪声级别
        
        Args:
            timesteps: 总时间步数
            beta_start: 初始β值
            beta_end: 最终β值
            schedule_type: 调度类型，可选['linear', 'cosine']
        """
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        
        # 生成噪声调度
        if schedule_type == 'linear':
        #生成等间序列的噪声强度信号
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule_type == 'cosine':
            # 使用余弦调度 - DDPM++论文中提出
            steps = timesteps + 1     #一共加噪步数是1000，中间会生成1001个点
            s = 0.008                 #这个s采用的是DDPM中的平移参数，对时间归一化坐标进行微笑调整，改善噪声调度曲线的平滑性和稳定性
            x = torch.linspace(0, timesteps, steps)
                                      # cumprod 构建扩散模型中的噪声衰减因子，
                                      # 这里x是一个1-1001的数字,最后乘以一个Π/2将整个序列映射到0-90度中
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2   #这里取平方的目的是让曲线变平滑
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])      #这里是在进行反推获得到β数值#这个可能还是要再考虑一下好吧，为什么要在这个构建这个β后面再返现进行计算呢
            self.betas = torch.clamp(betas, 0.0001, 0.9999)             #这里是为了数值稳定性，防止训练或者采样过程中出现数值溢出，梯度爆炸的现象
        else:
            raise NotImplementedError(f"Unknown schedule type: {schedule_type}")
    
    
    #用于前向加噪
        # 预计算常用值
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
    
    #用于反向去噪    
        # 用于计算噪声的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    
        # 用于计算重参数化技巧的系数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

#
    def q_sample(self, x_start, t, noise=None):
        """
        前向过程：给定初始样本x_0和时间步t，计算噪声样本x_t
        
        Args:
            x_start: 原始坐标 [B, N, 3]
            t: 时间步 [B]
            noise: 可选，预生成的噪声
        
        Returns:
            x_t: 含噪声的坐标
            noise: 添加的噪声
        """
#加噪过程
        print(f"q_sample - x_start形状: {x_start.shape}, t形状: {t.shape}, t值: {t}")
        
        if noise is None:
            noise = torch.randn_like(x_start)#这里x_start是原始分子的构象，如果是半监督的话可能要给无监督样本生成x_start_pred伪构象
            print(f"生成随机噪声，形状: {noise.shape}")
            
        # 计算 x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1-alpha_cumprod_t) * ε
        try:
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            
            print(f"系数提取成功 - sqrt_alphas_cumprod_t形状: {sqrt_alphas_cumprod_t.shape}")
            print(f"系数提取成功 - sqrt_one_minus_alphas_cumprod_t形状: {sqrt_one_minus_alphas_cumprod_t.shape}")
            
            #这里是计算出了加噪的坐标和噪音是，x= 开方下的阿尔法x0 + 开方下的1-阿尔法 乘以塞塔    
            x_noised = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
            print(f"加噪后坐标形状: {x_noised.shape}, 最小值: {x_noised.min().item()}, 最大值: {x_noised.max().item()}")
            
            return x_noised, noise
        except Exception as e:
            print(f"q_sample过程发生错误: {str(e)}")
            # 返回原始输入和零噪声作为应急方案
            print("返回原始输入和零噪声作为应急方案")
            return x_start, torch.zeros_like(x_start)
    
#去噪过程
    def predict_start_from_noise(self, x_t, t, noise):
        """从噪声预测原始坐标"""
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        return sqrt_recip_alphas_t * x_t - sqrt_one_minus_alphas_cumprod_t * noise / torch.sqrt(1.0 - self.alphas_cumprod)
#进行后验计算，直接从x_t预测到x0的部分
    def q_posterior(self, x_start, x_t, t):
        """计算后验均值和方差"""
        posterior_mean_coef1 = extract(self.alphas_cumprod_prev / self.alphas_cumprod, t, x_t.shape)
        posterior_mean_coef2 = extract(self.betas / (1.0 - self.alphas_cumprod), t, x_t.shape)
        
        posterior_mean = posterior_mean_coef1 * x_start + posterior_mean_coef2 * x_t
        posterior_var = extract(self.posterior_variance, t, x_t.shape)
        
        return posterior_mean, posterior_var
    
    def p_sample(self, model, x_t, t, complex_graph):
        """
        反向过程：单步采样
        
        Args:
            model: 模型
            x_t: 含噪声的坐标 [B, N, 3]
            t: 时间步 [B]
            complex_graph: 复合图数据
        
        Returns:
            x_{t-1}: 降噪一步的坐标
        """
        # 计算均值
        with torch.no_grad():
            # 设置时间步和当前坐标
            complex_graph.coor = x_t
            complex_graph.timestep = t
            
            # 让模型预测噪声
            pred = model(complex_graph, return_graph=True)
            noise_pred = pred.noise_pred
            
            # 用预测的噪声重建x_0
            x_0_pred = self.predict_start_from_noise(x_t, t, noise_pred)
            
            # 计算后验均值和方差
            mean, var = self.q_posterior(x_0_pred, x_t, t)
            
            # 无噪声情况下直接返回均值
            if t[0] == 0:
                return mean
            
            # 采样
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(var) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, model, complex_graph, noise=None, return_all=False):
        """
        反向采样过程：从纯噪声开始，逐步降噪，生成样本
        
        Args:
            model: 模型
            complex_graph: 复合图数据
            noise: 可选，初始噪声
            return_all: 是否返回所有中间步骤
            
        Returns:
            生成的样本
        """
        device = next(model.parameters()).device
        batch_size = complex_graph.coor.shape[0]
        
        # 从纯噪声开始
        if noise is None:
            cycle_i = complex_graph.cycle_i
            ligand_mask = complex_graph.l_x_mask_bool_cycle[cycle_i].reshape(-1)
            x = torch.randn_like(complex_graph.coor)
            # 只对配体部分添加噪声
            if ligand_mask is not None:
                #这里应该是将ligand_mask的形状进行拉伸，第一个维度是batch_size,第二个维度是原子数目，第三个维度是1经过广播后应该是变成空间坐标的形式
                ligand_mask_expanded = ligand_mask.reshape(x.shape[0], x.shape[1], 1).expand_as(x)
            #这里是实现了对噪声的控制，让其可以够在配体部分进行采样，这个ligand_mask_expanded其实是一个布尔值类型。这一段是将配体部分进行了剥离，方便加噪
                x = x * ligand_mask_expanded + complex_graph.coor * (~ligand_mask_expanded)
            
        else:
            x = noise
        
        intermediate = [x]
        
        # 逐步降噪
        for t in reversed(range(self.timesteps)):
            time_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, time_tensor, complex_graph)
            intermediate.append(x)
            
        return intermediate if return_all else x


# 辅助函数：提取指定时间步的值
# def extract(a, t, shape):
#     """
#     从a中提取指定时间步t的值，并reshape到指定形状
#     """
#     batch_size = t.shape[0]
#     out = a.gather(-1, t.cpu())
#     return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(t.device)
def extract(a, t, shape):
    """
    从a中提取指定时间步t的值，并reshape到指定形状
    
    Args:
        a: 包含所有时间步值的张量 [timesteps]
        t: 要提取的时间步 [batch_size]
        shape: 输出张量的形状
        
    Returns:
        形状为 [batch_size, 1, 1, ...] 的张量，用于广播
    """
    try:
        batch_size = t.shape[0]
        print(f"extract - a形状: {a.shape}, t形状: {t.shape}, 目标shape: {shape}, batch_size: {batch_size}")
        
        # 确保 t 和 a 在同一个设备上
        t = t.to(a.device)
        
        # 确保 t 的值在有效范围内
        t_clamped = torch.clamp(t, 0, a.shape[0] - 1)
        if not torch.equal(t, t_clamped):
            print(f"警告: t值超出范围，已被截断。原始值: {t}, 截断值: {t_clamped}")
            t = t_clamped
            
        # 提取对应时间步的值
        out = a.gather(-1, t)
        
        # 重塑张量以便于广播
        reshaped = out.reshape(batch_size, *((1,) * (len(shape) - 1)))
        print(f"提取成功 - 输出形状: {reshaped.shape}")
        
        return reshaped
    except Exception as e:
        print(f"extract函数发生错误: {str(e)}")
        # 返回一个安全的默认值
        try:
            local_batch_size = t.shape[0] if hasattr(t, 'shape') else 1
        except:
            local_batch_size = 1
            
        print(f"返回默认值1.0，batch_size={local_batch_size}")
        default_value = torch.ones((local_batch_size, *((1,) * (len(shape) - 1))), device=a.device)
        return default_value


# Diffusion模型包装器
class DiffusionWrapper(nn.Module):
    def __init__(self, base_model, noise_pred_net=None, timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule_type='linear'):
        """
        Diffusion模型包装器，用于封装现有模型，增加噪声预测功能
        
        Args:
            base_model: 基础模型 (LigPoseBase/LigPoseStruct/LigPoseScr)
            noise_pred_net: 噪声预测网络，如果为None则使用默认MLP
            timesteps: 扩散步数
            beta_start: 初始β值
            beta_end: 最终β值
            schedule_type: 噪声调度类型
        """
        super().__init__()
        self.base_model = base_model
        self.scheduler = NoiseScheduler(
            timesteps=timesteps, 
            beta_start=beta_start,
            beta_end=beta_end,
            schedule_type=schedule_type
        )
        
        # 如果没有提供噪声预测网络，则使用默认MLP，这里我们先暂时使用MLP进行吧
        hidden_dim = getattr(base_model, 'node_hidden', 768)  # 默认值为768
        #如果这个噪声预测网络是none的话就根据这个hidden_dim来构建一个MLP网络
        if noise_pred_net is None:
            self.noise_pred_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3)  # 预测3D坐标噪声
            )
        else:
            self.noise_pred_net = noise_pred_net
    
    def forward(self, complex_graph, return_graph=False, t=None):
        """
        前向传播：调用基础模型，并增加噪声预测
        
        Args:
            complex_graph: 复合图数据
            return_graph: 是否返回图对象
            t: 时间步，训练时随机生成，推理时从T递减到0
        
        Returns:
            与基础模型相同的输出，但增加了噪声预测
        """
        # 设置时间步
#加噪过程
        if t is None and self.training:
            # 打印complex_graph的所有属性，帮助调试
            debug_attrs = [attr for attr in dir(complex_graph) if not attr.startswith('_')]
            print("Complex Graph属性:", debug_attrs)
            
            # 更健壮地获取批量大小
            batch_size = None
            device = None
            
            # 对于数据加载器返回的ComplexStructDataset数据
            if hasattr(complex_graph, 'coor_init') and complex_graph.coor_init is not None:
                if isinstance(complex_graph.coor_init, torch.Tensor):
                    print(f"从coor_init获取batch_size，形状: {complex_graph.coor_init.shape}")
                    # coor_init shape应该是[batch_size, max_len_complex_before_sampling, 3]
                    batch_size = complex_graph.coor_init.shape[0]
                    device = complex_graph.coor_init.device
            
            # 尝试从coor获取批量大小
            if batch_size is None and hasattr(complex_graph, 'coor') and complex_graph.coor is not None:
                if isinstance(complex_graph.coor, torch.Tensor):
                    print(f"从coor获取batch_size，形状: {complex_graph.coor.shape}")
                    batch_size = complex_graph.coor.shape[0]
                    device = complex_graph.coor.device
            
            # 尝试从protein_node_feature_init获取
            if batch_size is None and hasattr(complex_graph, 'protein_node_feature_init') and complex_graph.protein_node_feature_init is not None:
                if isinstance(complex_graph.protein_node_feature_init, torch.Tensor):
                    print(f"从protein_node_feature_init获取batch_size，形状: {complex_graph.protein_node_feature_init.shape}")
                    batch_size = complex_graph.protein_node_feature_init.shape[0]
                    device = complex_graph.protein_node_feature_init.device
                    
            # 尝试从ligand_node_feature_init获取
            if batch_size is None and hasattr(complex_graph, 'ligand_node_feature_init') and complex_graph.ligand_node_feature_init is not None:
                if isinstance(complex_graph.ligand_node_feature_init, torch.Tensor):
                    print(f"从ligand_node_feature_init获取batch_size，形状: {complex_graph.ligand_node_feature_init.shape}")
                    batch_size = complex_graph.ligand_node_feature_init.shape[0]
                    device = complex_graph.ligand_node_feature_init.device
            
            # 尝试从aff_true获取
            if batch_size is None and hasattr(complex_graph, 'aff_true') and complex_graph.aff_true is not None:
                if isinstance(complex_graph.aff_true, torch.Tensor):
                    print(f"从aff_true获取batch_size，形状: {complex_graph.aff_true.shape}")
                    batch_size = complex_graph.aff_true.shape[0]
                    device = complex_graph.aff_true.device
            
            # 如果上面方法失败，尝试从x获取
            if batch_size is None and hasattr(complex_graph, 'x') and complex_graph.x is not None:
                if isinstance(complex_graph.x, torch.Tensor):
                    print(f"从x获取batch_size，形状: {complex_graph.x.shape}")
                    batch_size = complex_graph.x.shape[0]
                    device = complex_graph.x.device
            
            # 如果仍然失败，尝试获取batch属性
            if batch_size is None and hasattr(complex_graph, 'batch') and complex_graph.batch is not None:
                if isinstance(complex_graph.batch, torch.Tensor):
                    print(f"从batch获取batch_size，形状: {complex_graph.batch.shape}")
                    batch_size = complex_graph.batch.max().item() + 1
                    device = complex_graph.batch.device
            
            # 尝试从node_sampling_loc获取
            if batch_size is None and hasattr(complex_graph, 'node_sampling_loc') and complex_graph.node_sampling_loc is not None:
                if isinstance(complex_graph.node_sampling_loc, torch.Tensor):
                    print(f"从node_sampling_loc获取batch_size，形状: {complex_graph.node_sampling_loc.shape}")
                    # node_sampling_loc形状应为[n_cycle, batch_size, n_loc]
                    if len(complex_graph.node_sampling_loc.shape) >= 2:
                        batch_size = complex_graph.node_sampling_loc.shape[1]
                        device = complex_graph.node_sampling_loc.device
            
            # 如果所有方法都失败，尝试检查complex_graph是否是类字典
            if batch_size is None and hasattr(complex_graph, 'keys'):
                print("Complex Graph是类字典对象，键:", list(complex_graph.keys()))
                # 尝试一些常见的键
                for key in ['coor_init', 'coor', 'protein_node_feature_init', 'ligand_node_feature_init', 'aff_true', 'x', 'node_feature', 'edge_index']:
                    if key in complex_graph and isinstance(complex_graph[key], torch.Tensor):
                        print(f"从字典键'{key}'获取batch_size，形状: {complex_graph[key].shape}")
                        batch_size = complex_graph[key].shape[0]
                        device = complex_graph[key].device
                        break
            
            # 仍然无法确定batch_size，尝试回退到默认值1
            if batch_size is None:
                print("警告：无法确定batch_size，使用默认值1")
                batch_size = 1
                # 尝试从任何tensor属性获取设备
                device = None
                for attr_name in dir(complex_graph):
                    if attr_name.startswith('_'):
                        continue
                    attr = getattr(complex_graph, attr_name)
                    if isinstance(attr, torch.Tensor):
                        device = attr.device
                        print(f"从属性'{attr_name}'获取设备: {device}")
                        break
                        break
                if device is None:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    print(f"使用默认设备: {device}")
            
            # 生成时间步
            print(f"生成时间步，batch_size={batch_size}, device={device}")
            batch_size_int = int(batch_size)  # 确保batch_size是整数
            t = torch.randint(0, self.scheduler.timesteps, size=(batch_size_int,), device=device)
            complex_graph.timestep = t
            
            # 添加噪声到坐标
            # 安全地获取cycle_i
            if hasattr(complex_graph, 'cycle_i'):
                cycle_i = complex_graph.cycle_i
                print(f"当前cycle_i: {cycle_i}")
                
                # 安全地获取ligand_mask
                if hasattr(complex_graph, 'l_x_mask_bool_cycle') and cycle_i < len(complex_graph.l_x_mask_bool_cycle):
                    ligand_mask = complex_graph.l_x_mask_bool_cycle[cycle_i].reshape(-1)
                    print(f"获取到ligand_mask，形状: {ligand_mask.shape}, 非零元素: {ligand_mask.sum().item()}")
                    
                    # 添加噪声
                    if hasattr(complex_graph, 'coor') and complex_graph.coor is not None:
                        # 获取原始坐标
                        coor_orig = complex_graph.coor.clone()
                        print(f"原始坐标形状: {coor_orig.shape}")
                        
                        # 计算噪声坐标
                        noised_coor, target_noise = self.scheduler.q_sample(coor_orig, t)
                        print(f"添加噪声后坐标形状: {noised_coor.shape}, 目标噪声形状: {target_noise.shape}")
                        
                        # 保存目标噪声以便计算loss
                        complex_graph.target_noise = target_noise
                        
                        # 更新坐标
                        complex_graph.coor = noised_coor
                    elif hasattr(complex_graph, 'coor_init') and complex_graph.coor_init is not None:
                        # 尝试使用coor_init代替
                        coor_orig = complex_graph.coor_init.clone()
                        print(f"使用coor_init作为原始坐标，形状: {coor_orig.shape}")
                        
                        # 计算噪声坐标
                        noised_coor, target_noise = self.scheduler.q_sample(coor_orig, t)
                        print(f"添加噪声后坐标形状: {noised_coor.shape}, 目标噪声形状: {target_noise.shape}")
                        
                        # 保存目标噪声以便计算loss
                        complex_graph.target_noise = target_noise
                        
                        # 更新坐标或创建新属性
                        complex_graph.coor = noised_coor
                    else:
                        print("警告：无法找到坐标数据(coor或coor_init不存在或为None)，跳过噪声添加")
                else:
                    print("警告：无法获取ligand_mask，检查l_x_mask_bool_cycle属性")
                    if hasattr(complex_graph, 'l_x_mask_bool_cycle'):
                        print(f"l_x_mask_bool_cycle存在，长度: {len(complex_graph.l_x_mask_bool_cycle)}")
                    else:
                        print("l_x_mask_bool_cycle不存在")
            else:
                print("警告：complex_graph.cycle_i不存在，查找其他可用的掩码")
                # 尝试使用其他可能的掩码
                if hasattr(complex_graph, 'ligand_mask_after_sampling'):
                    print(f"使用ligand_mask_after_sampling，形状: {complex_graph.ligand_mask_after_sampling.shape}")
                    
                    # 添加噪声，使用任一可用的坐标源
                    if hasattr(complex_graph, 'coor') and complex_graph.coor is not None:
                        coor_orig = complex_graph.coor.clone()
                    elif hasattr(complex_graph, 'coor_init') and complex_graph.coor_init is not None:
                        coor_orig = complex_graph.coor_init.clone()
                    else:
                        print("警告：无法找到坐标数据，跳过噪声添加")
                        coor_orig = None
                    
                    if coor_orig is not None:
                        print(f"使用替代掩码添加噪声，坐标形状: {coor_orig.shape}")
                        noised_coor, target_noise = self.scheduler.q_sample(coor_orig, t)
                        complex_graph.target_noise = target_noise
                        complex_graph.coor = noised_coor
        
        # 首先调用基础模型
        if return_graph:
            complex_graph = self.base_model(complex_graph, timestep=t if self.training else None, return_graph=True)
            
            # 添加噪声预测，使用节点特征进行预测
            if hasattr(complex_graph, 'x') and complex_graph.x is not None:
                noise_pred = self.noise_pred_net(complex_graph.x)
                complex_graph.noise_pred = noise_pred
                print(f"噪声预测完成，形状: {noise_pred.shape}")
            else:
                print("警告：无法进行噪声预测，complex_graph.x不存在或为None")
            
            return complex_graph
        else:
            base_output = self.base_model(complex_graph, return_graph=False)
            
            # 将噪声预测添加到输出中
            # 注意：这需要根据具体的基础模型输出结构进行调整
            # 假设基础模型返回的是元组，我们在最后添加噪声预测
            noise_pred = self.noise_pred_net(complex_graph.x)
            return (*base_output, noise_pred)
#去噪过程
    @torch.no_grad()
    def sample(self, complex_graph, timesteps=None, return_all=False):
        """
        采样过程：从纯噪声开始，逐步降噪，生成样本
        
        Args:
            complex_graph: 复合图数据
            timesteps: 采样步数，默认使用训练时的步数
            return_all: 是否返回所有中间步骤
        
        Returns:
            生成的样本
        """
        if timesteps is not None:
            # 如果指定了采样步数，创建新的调度器
            scheduler = NoiseScheduler(timesteps=timesteps, 
                                     beta_start=self.scheduler.beta_start,
                                     beta_end=self.scheduler.beta_end,
                                     schedule_type=self.scheduler.schedule_type)
        else:
            scheduler = self.scheduler
            
        return scheduler.p_sample_loop(self, complex_graph, return_all=return_all)


#diffusion的训练过程是将不同噪音步骤的数据恢复到原本数据，而推理过程是将噪音逐步去噪到原本的数据中