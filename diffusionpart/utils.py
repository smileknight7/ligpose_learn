import torch
import math
import numpy as np
import torch.nn.functional as F
# utils function


# for EGNN
def remove_mean_with_mask(x, node_mask, fix_size):
    masked_max_abs_value = (x * node_mask).abs().max()
    assert masked_max_abs_value < 1e-5, f"Error{masked_max_abs_value} is too high"            # 校验无效节点
    
    
    if fix_size is None:                                                                      # 有效节点去均值
        fix_size = x.shape[1]
    N = node_mask[:, :fix_size].sum(dim=1, keepdim=True)                # N: [B, 1]
    
    mean = torch.sum(x[:, :fix_size], dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x 

def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x



# random noise
def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    assert len(size) ==3
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    x_projected = remove_mean_with_mask(x_masked, node_mask)                 # x.pos做了整体质心归零，噪声这里也要这样做一下
    return x_projected
def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    return x_masked

# assert for masked
def assert_correctly_masked(variable, node_mask):                                  # 检验无效节点是不是0（兜底函数）
    assert (variable * ~node_mask).abs().max().item() < 1e-4, \
        'Variables not masked properly.'
def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):                           # 兜底函数，检验质心是不是0
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f'Mean is not zero, relative_error {rel_error}'

# batch sum（聚合到B）
def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)

# 获得高斯分布的CDF(用于计算高斯分布左侧的概率)
def cdf_standard_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))


# noise_utils

def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)

class PositiveLinear(torch.nn.Module):                              # weight始终未正数的线性层(主要用物理化学可解释的模型)
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):                             # 这里的bias默认是True
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(                                   # 这里默认设置了weight是需要训练的，但是bias可以选择训练或者是不训练
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)                         # 初始化weight要加上一个偏置（此时的no_grad是为了不将这个操作记录 .init.uniform_内部也有这个函数）
                                                                                    # 加上这个偏移是为了让初始化的weight尽可能小
        if self.bias is not None:                                                   # 初始化bias，但是torch中一般是会默认进行初始化的
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = softplus(self.weight)                                     # 通过这种设计将weight限制为正值
        return F.linear(input, positive_weight, self.bias)



def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)
    return alphas_cumprod
def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2






# noise_schedule
# stationary noise schedule 
class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """
    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':                          
            alphas2 = cosine_beta_schedule(timesteps)
        
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]

# VDM（Variational Diffusion Model）可学习的噪音调度器   连续时间扩散模型（continuous-time diffusion models）” 的典型做法
class GammaNetwork(torch.nn.Module):                                                                    # 这里这个网络就是用来学习一个任意形状但是必须是单调递增的γt
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""
    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = torch.nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = torch.nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))        # 这里最终实现的是一个线性增长函数L1加上了L3（激活函数L2(映射到潜空间)映射回1个数字）

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)                     # 先用单调网络得到一个原始曲线γt
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (      # 通过归一化将γt映射到[0,1]区间
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]         ----> 将时间T映射到log-SNR 信噪比的对数上去       大的正数log-SNR------>信号占主导，几乎无噪音
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma           #    小的负数log-SNR------>噪音占主导，几乎无信号   

        return gamma
    


# KL utils
def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    return sum_except_batch(
            (
                torch.log(p_sigma / q_sigma)
                + 0.5 * (q_sigma**2 + (q_mu - p_mu)**2) / (p_sigma**2)
                - 0.5
            ) * node_mask
        )


def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    mu_norm2 = sum_except_batch((q_mu - p_mu)**2)
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1
    return d * torch.log(p_sigma / q_sigma) + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2) - 0.5 * d
