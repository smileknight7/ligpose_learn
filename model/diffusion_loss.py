import torch
import torch.nn.functional as F
from torch_scatter import scatter_min, scatter_mean, scatter_add
from einops import rearrange, repeat
from model.loss import StructLoss

class DiffusionLoss(StructLoss):
    def __init__(self, args):
        super(DiffusionLoss, self).__init__(args)
        # Diffusion特有的损失权重
        self.noise_loss_weight = getattr(args, 'noise_loss_weight', 1.0)
        
    def forward(self, tup_pred, complex_graph, epoch=1e+5):
        # 先调用父类的forward方法处理常规损失
        grad_loss, eval_loss = super().forward(tup_pred, complex_graph, epoch)
        
        # 处理Diffusion特有的噪声预测损失
        noise_loss = torch.tensor(0.0, device=grad_loss.device)
        
        if hasattr(complex_graph, 'target_noise') and complex_graph.target_noise is not None:
            if hasattr(complex_graph, 'noise_pred') and complex_graph.noise_pred is not None:
                # 获取目标噪声和预测噪声
                target_noise = complex_graph.target_noise
                noise_pred = complex_graph.noise_pred
                
                print(f"噪声损失计算 - 目标噪声形状: {target_noise.shape}, 预测噪声形状: {noise_pred.shape}")
                
                # 只对配体部分计算噪声损失
                if hasattr(complex_graph, 'cycle_i') and hasattr(complex_graph, 'l_x_mask_bool_cycle'):
                    cycle_i = complex_graph.cycle_i
                    if cycle_i < len(complex_graph.l_x_mask_bool_cycle):
                        ligand_mask = complex_graph.l_x_mask_bool_cycle[cycle_i]
                        print(f"使用cycle_i={cycle_i}的配体掩码，形状: {ligand_mask.shape}, 非零元素: {ligand_mask.sum().item()}")
                        
                        # 确保掩码维度和张量匹配
                        if ligand_mask.dim() < target_noise.dim():
                            # 将掩码扩展到与目标噪声相同的维度
                            for _ in range(target_noise.dim() - ligand_mask.dim()):
                                ligand_mask = ligand_mask.unsqueeze(-1)
                            # 扩展掩码到坐标维度
                            ligand_mask = ligand_mask.expand_as(target_noise)
                        
                        # 应用掩码，计算MSE损失
                        masked_target = target_noise[ligand_mask]
                        masked_pred = noise_pred[ligand_mask]
                        
                        if masked_target.numel() > 0 and masked_pred.numel() > 0:
                            noise_loss = F.mse_loss(masked_pred, masked_target)
                            print(f"配体噪声MSE损失: {noise_loss.item()}")
                        else:
                            print("警告: 掩码后的张量为空，无法计算MSE损失")
                    else:
                        print(f"警告: cycle_i ({cycle_i}) 超出 l_x_mask_bool_cycle 长度 ({len(complex_graph.l_x_mask_bool_cycle)})")
                        # 尝试使用其他掩码
                        if hasattr(complex_graph, 'ligand_mask_after_sampling'):
                            ligand_mask = complex_graph.ligand_mask_after_sampling
                            print(f"使用替代掩码 ligand_mask_after_sampling，形状: {ligand_mask.shape}")
                            
                            # 确保掩码维度和张量匹配
                            if ligand_mask.dim() < target_noise.dim():
                                # 将掩码扩展到与目标噪声相同的维度
                                for _ in range(target_noise.dim() - ligand_mask.dim()):
                                    ligand_mask = ligand_mask.unsqueeze(-1)
                                # 扩展掩码到坐标维度
                                ligand_mask = ligand_mask.expand_as(target_noise)
                            
                            masked_target = target_noise[ligand_mask.bool()]
                            masked_pred = noise_pred[ligand_mask.bool()]
                            
                            if masked_target.numel() > 0 and masked_pred.numel() > 0:
                                noise_loss = F.mse_loss(masked_pred, masked_target)
                                print(f"使用替代掩码的噪声MSE损失: {noise_loss.item()}")
                            else:
                                print("警告: 替代掩码后的张量为空，计算全局MSE损失")
                                noise_loss = F.mse_loss(noise_pred, target_noise)
                        else:
                            print("警告: 无法找到合适的掩码，计算全局MSE损失")
                            noise_loss = F.mse_loss(noise_pred, target_noise)
                elif hasattr(complex_graph, 'ligand_mask_after_sampling'):
                    # 直接使用ligand_mask_after_sampling
                    ligand_mask = complex_graph.ligand_mask_after_sampling
                    print(f"使用ligand_mask_after_sampling，形状: {ligand_mask.shape}")
                    
                    # 确保掩码维度和张量匹配
                    if ligand_mask.dim() < target_noise.dim():
                        # 将掩码扩展到与目标噪声相同的维度
                        for _ in range(target_noise.dim() - ligand_mask.dim()):
                            ligand_mask = ligand_mask.unsqueeze(-1)
                        # 扩展掩码到坐标维度
                        ligand_mask = ligand_mask.expand_as(target_noise)
                    
                    masked_target = target_noise[ligand_mask.bool()]
                    masked_pred = noise_pred[ligand_mask.bool()]
                    
                    if masked_target.numel() > 0 and masked_pred.numel() > 0:
                        noise_loss = F.mse_loss(masked_pred, masked_target)
                        print(f"使用ligand_mask_after_sampling的噪声MSE损失: {noise_loss.item()}")
                    else:
                        print("警告: 掩码后的张量为空，计算全局MSE损失")
                        noise_loss = F.mse_loss(noise_pred, target_noise)
                else:
                    # 如果没有掩码信息，计算全局MSE损失
                    print("未找到任何掩码，计算全局MSE损失")
                    noise_loss = F.mse_loss(noise_pred, target_noise)
                    print(f"全局噪声MSE损失: {noise_loss.item()}")
            else:
                print("警告: complex_graph.noise_pred不存在或为None，跳过噪声损失计算")
        else:
            print("警告: complex_graph.target_noise不存在或为None，跳过噪声损失计算")
        
        # 更新总损失
        weighted_noise_loss = self.noise_loss_weight * noise_loss
        grad_loss = grad_loss + weighted_noise_loss
        
        # 添加到评估损失字典
        eval_loss['diffusion_noise_loss'] = noise_loss.detach().cpu().numpy()
        eval_loss['weighted_diffusion_noise_loss'] = weighted_noise_loss.detach().cpu().numpy()
        
        return grad_loss, eval_loss
