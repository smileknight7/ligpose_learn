# diffusion part
import torch
import numpy as np
from torch import nn
from diffusionpart.utils import *

class DiffusionLoss(nn.Module):
    def __init__(self, args):
        super().__init__()           # super(DiffusionLoss, self).__init__()
        self.args = args
        #self.pocket =args.pocket
        self.node_coarse = args.node_coarse
        self.in_node_nf = args.in_node_nf         # 8
        self.n_dims = args.n_dims                 # 3 

        self.loss_type = args.loss_type
        self.norm_values = args.norm_values
        self.norm_bias = args.norm_bias


        self.T = args.timesteps
        self.int_nf = args.int_nf
        self.cont_nf = args.cont_nf


        if args.noise_schedule == 'learned':
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(schedule_type=args.noise_schedule, T=self.T)

            
        self.protein_embed = nn.Embedding(21, self.in_node_nf)       # 这里写的有点简单了，用embedding做成了fragment特征的大小



# normalize and denormalize
    def subspace_dimensionality(self, node_mask):                           # "有效的平移不变子空间的维度"
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def normalize(self, x, h, node_mask):
        
        delta_log_px = -self.subspace_dimensionality(node_mask)*np.log(self.norm_values[0])
        
        x = x / self.norm_values[0]
        h = (h - self.norm_bias[1]) / self.norm_values[1]*node_mask
        return x, h, delta_log_px
    
# array inflation
    def inflate_batch_array(self, array, target):

        target_shape = (array.size(0),) + (1,) * (len(target.size()) -1)
        return array.view(target_shape)
    
# diffusion process functions 这里构建的是log-SNR形式的噪音调度
    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)
    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)
    def SNR(self, gamma):
        return torch.exp(-gamma)






    def sample_combined_position_feature_noise(self, n_samplexs, n_nodes, node_mask):

        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samplexs, n_nodes, self.n_dims),
            device=node_mask.device,
            node_mask=node_mask)
        z_h = sample_gaussian_with_mask(
            size=(n_samplexs, n_nodes, self.in_node_nf),
            device=node_mask.device,
            node_mask=node_mask)
        z = torch.cat([z_x, z_h], dim=2)
        return z


# 这个模型针对离散变量和连续变量加噪方式是一样的(只是在最后采样过程中将采样回到了离散变量)
# 计算logP(x,h|z0)的非常数部分 ---->整个分子
    def log_pxh_given_z0_without_constants(
        self, x, h, z_t, gamma_0, eps, net_out, node_mask, epsilon=1e-10):

        # discrete h
        z_h_int = z_t[:, :, self.n_dims:self.n_dims + self.int_nf]

        eps_x = eps[:, :, :self.n_dims]
        net_x = net_out[:, :, :self.n_dims]
        
        # continues h
        eps_h = eps[:, :, self.n_dims + self.int_nf : self.n_dims + self.int_nf + self.cont_nf]
        net_h = net_out[:, :, self.n_dims + self.int_nf : self.n_dims + self.int_nf + self.cont_nf]


        sigma_0 = self.sigma(gamma_0, target_tensor=x)
        sigma_0_int = sigma_0*self.norm_values[2]
        # continues h logP(x,h|z)        这个L2损失既可以当作loss，又可以作为logP来计算概率 
        log_p_x_given_z_without_constants = - 0.5 * self.compute_error(net_x, gamma_0, eps_x) 
        log_p_h_given_z_without_constants = - 0.5 * self.compute_error(net_h, gamma_0, eps_h)
        
        # discrete h loss
        h_integer = torch.round(h[:, :, :self.int_nf] * self.norm_values[2] + self.norm_biases[2]).long()
        estimated_h_integer = z_h_int * self.norm_values[2] + self.norm_biases[2]
        assert h_integer.size() == estimated_h_integer.size()     
        h_integer_centered = h_integer - estimated_h_integer
                                                                            # 对于类别变量的概率是一个范围所以这里要这样使用
        log_ph_integer = torch.log(
        cdf_standard_gaussian((h_integer_centered + 0.5) / sigma_0_int)
        - cdf_standard_gaussian((h_integer_centered - 0.5) / sigma_0_int)
        + epsilon)
        log_ph_integer = sum_except_batch(log_ph_integer * node_mask)  


        log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z_without_constants + log_ph_integer

        return log_p_xh_given_z


# L2 loss
    def compute_error(self, net_out, gamma_t, eps):
        eps_t = net_out                                                    # 这里的输出是预测噪声！！！！
        if self.training and self.loss_type == 'l2':
            denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1]       # eps_t.shape[1] 节点数量
            error = sum_except_batch((eps - eps_t) ** 2) / denom
        else:
            error = sum_except_batch((eps - eps_t) ** 2)                 #    
        return error

# logP常数部分
    def log_constants_p_x_given_z0(self, x, node_mask):
        batch_size = x.size(0)
        n_nodes = node_mask.squeeze(2).sum(dim=1)                           # [B]
        assert n_nodes.size() == (batch_size,)
        degree_of_freedom_x = (n_nodes -1) * self.n_dims

        zeros = torch.zeros((x.size(0),1), device=x.device)
        gamma_0 = self.gamma(zeros)
        log_sigma_0 = 0.5*gamma_0.view(batch_size)
        return degree_of_freedom_x*(- log_sigma_0 - 0.5*np.log(2*np.pi))
    def log_constants_p_h_given_z0(self, h, node_mask):
        batch_size = h.size(0)
        n_nodes = node_mask.squeeze(2).sum(dim=1)                           # [B]
        assert n_nodes.size() == (batch_size,)
        degree_of_freedom_h = n_nodes * self.in_node_nf

        zeros = torch.zeros((h.size(0),1), device=h.device)
        gamma_0 = self.gamma(zeros)
        log_sigma_0 = 0.5*gamma_0.view(batch_size)
        return degree_of_freedom_h*(- log_sigma_0 - 0.5*np.log(2*np.pi))

# 对最终加噪状态和标准高斯分布计算KL散度
    def kl_prior(self, xh, node_mask):
        
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((xh.size(0), 1), device=xh.device)
        gamma_T = self.gamma(ones)                                          # 最终加噪状态
        alpha_T = self.alpha(gamma_T, xh)
        # Compute means.
        mu_T = alpha_T * xh
        mu_T_x, mu_T_h = mu_T[:, :, :self.n_dims], mu_T[:, :, self.n_dims:]
        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.sigma(gamma_T, mu_T_x).squeeze()  # Remove inflate, only keep batch dimension for x-part.
        sigma_T_h = self.sigma(gamma_T, mu_T_h)
        # Compute KL for h-part.
        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        kl_distance_h = gaussian_KL(mu_T_h, sigma_T_h, zeros, ones, node_mask)
        # Compute KL for x-part.
        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        subspace_d = self.subspace_dimensionality(node_mask)
        kl_distance_x = gaussian_KL_for_dimension(mu_T_x, sigma_T_x, zeros, ones, d=subspace_d)
                                                                        # 计算forward diffusion 理论分布和标准高斯分布之间的KL散度
        return kl_distance_x + kl_distance_h





    def compute_loss(self, model, x, h, node_mask, edge_mask, t0_always, mol_shape):
        if t0_always:
            lowest_t = 1
        
        else:                       # 计算L2损失的时候不单独计算T0所以允许采到T0
            lowest_t = 0

        t_int = torch.randint(
            lowest_t, self.T + 1, size=(x.size(0),), device=x.device).float()          # t_int: [B] 输入t形状
        s_int = t_int -1
        t_is_zero = (t_int ==0).float()
        s = s_int / self.T
        t = t_int / self.T

# depart fragment and protein
        x, x_fix = x[:, :mol_shape], x[:, mol_shape:]                                           # x_fix: protein部分
        h, h_fix = h[:, :mol_shape], h[:, mol_shape:]
        node_mask, node_mask_fix = node_mask[:, :mol_shape], node_mask[:, mol_shape:]

        
# add noise
        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        eps = self.sample_combined_position_feature_noise(                                  # 感觉这里的加噪好像没有问题，针对当前ligand节点进行加噪的
            n_samplexs=x.size(0), n_nodes=x.size(1), node_mask=node_mask)

        xh = torch.cat([x, h], dim=-1)


        assert_mean_zero_with_mask(x, node_mask)
        z_t = alpha_t * xh + sigma_t * eps

        xh_fix = torch.cat([x_fix, h_fix], dim=-1)
        assert_correctly_masked(xh_fix, node_mask_fix)
        
# join fragment and protein
        z_t = torch.cat([z_t, xh_fix], dim=1)         
        node_mask = torch.cat([node_mask, node_mask_fix], dim=1)

# denoise
        net_out = model(z_t, t, node_mask, edge_mask, mol_shape=mol_shape)           

# loss compute L2
        net_out = net_out[:, :mol_shape]                                                    # 只取ligand部分          
        error = self.compute_error(net_out, gamma_t, eps)

        if self.training and self.loss_type == 'l2':
            SNR_weight = torch.ones_like(error)

# loss compute VLB                                                                          # 这个VLB损失有时间可能还是要看看
        else:
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        assert error.size() == SNR_weight.size()
        loss_t_larger_than_zero = 0.5*SNR_weight*error

# 常数项部分
        neg_log_constants = -self.log_constants_p_x_given_z0(x, node_mask[:, :mol_shape])            
        neg_log_constants += -self.log_constants_p_h_given_z0(h, node_mask[:, :mol_shape])


        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)
        
        kl_prior = self.kl_prior(xh, node_mask[:, :mol_shape])




        if t0_always:    # 这里的t0_always是对扩散模型t=0时间做一个显式log-likelihood计算
            loss_t = loss_t_larger_than_zero
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t                   # 这里是从loss_t出发对整个time_step计算了估计

            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.sigma(gamma_0, x)

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            eps_0 = self.sample_combined_position_feature_noise(
                n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask[:, :mol_shape])
            z_0 = alpha_0 * xh + sigma_0 * eps_0            
            z_0 = torch.cat([z_0, xh_fix], dim=1)

            net_out = model(z_0, t_zeros, node_mask, edge_mask, mol_shape=mol_shape)
            net_out = net_out[:, :mol_shape]                                    
            node_mask = node_mask[:, :mol_shape]

            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_0[:, :mol_shape], gamma_0, eps_0, net_out, node_mask)

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == loss_term_0.size()

            loss = kl_prior + neg_log_constants + estimator_loss_terms + loss_term_0

        else:                                                                   
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_t[:, :mol_shape], gamma_t, eps, net_out, node_mask[:, :mol_shape])

            t_is_not_zero = 1 - t_is_zero

            loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

            # Only upweigh estimator if using the vlb objective.
            if self.training and self.loss_type == 'l2':
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1  # Includes t = 0.
                estimator_loss_terms = num_terms * loss_t

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()

            loss = kl_prior + neg_log_constants + estimator_loss_terms
 
        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        return loss, {'t': t_int.squeeze(), 'loss_t': loss.squeeze(),
                    'error': error.squeeze()}

# 总结一下：t0_always:                                  计算VLB损失，此时要计算t=0时刻的logP(x,h|z0), 并且计算∑ₜ L_t，所以乘 T
#         self.training and self.loss_type == 'l2':    仅计算l2损失，常数项会设为0(neg_log_constants)，t=0时使用logP(x,h|z0)，非零时间步使用L2
# 训练时间一般只用l2损失


    def nll(self, model, x, h, node_mask, edge_mask, mol_shape=None):

        x, h, delta_log_px = self.normalize(x, h, node_mask)

        if self.training and self.loss_type == 'l2':                                              # self.training --> torch中默认的参数
            delta_log_px = torch.zeros_like(delta_log_px)                                   # 一般是使用ls损失代替KL散度，t0 always=False  
            loss, loss_dict = self.compute_loss(model ,x, h, node_mask, edge_mask, t0_always=False, mol_shape=mol_shape)
        
        else:
            loss, loss_dict = self.compute_loss(model, x, h, node_mask, edge_mask, t0_always=True, mol_shape=mol_shape)      
                                                                                                  # eval模式下使用KL散度，t0 always=True
        neg_log_pxh = loss
        assert neg_log_pxh.size() == delta_log_px.size()
        neg_log_pxh = neg_log_pxh - delta_log_px
        return neg_log_pxh
    


    def forward(self, model, batch):                                                   # 这里区分了一下蛋白条件生成和非条件生成
        x = batch['fragment_pos']           # 这里这个mol_shape可能不对，需要再考虑一下
        #mol_shape = None
        #if self.pocket:
        mol_shape = x.shape[1]
        x = torch.cat([x, batch['protein_pos']],dim=1)
        
        #node_mask = batch['atom_mask']
        fragment_mask = batch['fragment_mask']  
        protein_mask = batch['protein_mask']
        node_mask = torch.cat([fragment_mask, protein_mask],dim=1)
        
        #if self.pocket:                                                             # 构建边的全连接图
        fragment_edge_mask = batch['fragment_edge_mask']
        protein_edge_mask = batch['protein_edge_mask']
        edge_mask_shape = batch['fragment_edge_mask'].shape[1]+ batch['protein_edge_mask'].shape[1]
        edge_mask_concat = torch.zeros(fragment_edge_mask.shape[0], edge_mask_shape, edge_mask_shape)  
        edge_mask_concat[:, :mol_shape, :mol_shape] = fragment_edge_mask
        edge_mask_concat[:, mol_shape:, mol_shape:] = protein_edge_mask
        edge_mask = edge_mask_concat

        # if self.pocket:
        h = batch['fragment_feature']
        protein_feat = self.protein_embed(batch['protein_feature'])                 # self.protein_embed这个要写一下
        h = torch.cat([h, protein_feat],dim=1)


        x = remove_mean_with_mask(x, mask=node_mask.unsqueeze(-1))

        bs, n_nodes, n_dims = x.shape

        edge_mask = edge_mask.view(bs, n_nodes* n_nodes)                            # edge_mask: [B, N*N] 这里需要拉平一下
        assert_correctly_masked(x, node_mask)

        neg_log_pxh = self.nll(model, x, h, node_mask, edge_mask, mol_shape=mol_shape)  # 无pocket时是none
        nll = neg_log_pxh
        nll = nll.mean(0)

        return {"loss":nll}                                                         # 这个nll是不是能再换个名字呢？    
    

# sample process
    @torch.no_grad()
    def sample(self, num_samples, device, context=None, pocket_cond=None):
        sample_n = self.nodes_dist.sample(num_samples)          
        node_mask = torch.zeros([num_samples, max(sample_n), 1])
        edge_mask = torch.zeros(num_samples, max(sample_n), max(sample_n))

        for i in range(len(sample_n)):
            node_mask[i, :sample_n[i]] = 1
            edge_mask[i, :sample_n[i], :sample_n[i]] = 1 - torch.eye(sample_n[i])
        node_mask = node_mask.to(device).bool()
        edge_mask = edge_mask.to(device).bool()

        z = self.sample_combined_position_feature_noise(num_samples, max(sample_n), node_mask).to(device)
        if pocket_cond is not None:
            pocket_feat = self.pocket_embed(pocket_cond[0].type_as(z).long())
            pocket_pos = pocket_cond[1].type_as(z)
            pocket_node_mask = pocket_cond[2].type_as(z).bool()
            pocket_edge_mask = pocket_cond[3].type_as(z).bool()

            node_mask_concat = torch.cat([node_mask, pocket_node_mask], dim=1)
            edge_mask_concat = torch.zeros(num_samples, max(sample_n) + pocket_pos.size(1), max(sample_n) + pocket_pos.size(1)).type_as(edge_mask)
            edge_mask_concat[:, :max(sample_n), :max(sample_n)] = edge_mask
            edge_mask_concat[:, max(sample_n):, max(sample_n):] = pocket_edge_mask

        mol_shape = max(sample_n)          

        for s in reversed(range(0, self.T)):
            s_array = torch.full((num_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            if pocket_cond is not None:
                z = self.sample_p_zs_given_zt(s_array, t_array, torch.cat([z, torch.cat([pocket_pos, pocket_feat], dim=-1)], dim=1), node_mask_concat, edge_mask_concat, context, mol_shape=mol_shape)
            else:
                z = self.sample_p_zs_given_zt(s_array, t_array, z, node_mask, edge_mask, context, mol_shape=mol_shape)
        
        z = z[:, :mol_shape]
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context)
        x = [x[i, :sample_n[i]].cpu() for i in range(len(sample_n))]
        h = [h[i, :sample_n[i]].cpu() for i in range(len(sample_n))]
        if context is not None:
            context_out = [context[i, :sample_n[i]].cpu() for i in range(len(sample_n))]
            results = [{'x': x[i], 'h': h[i], 'context': context_out[i]} for i in range(len(sample_n))]
        else:
            results = [{'x': x[i], 'h': h[i]} for i in range(len(sample_n))]
        return results
    