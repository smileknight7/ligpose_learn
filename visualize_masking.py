#!/usr/bin/env python3
"""
可视化半监督掩蔽过程的演示脚本
"""

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
import matplotlib.pyplot as plt
import seaborn as sns

class MaskingVisualization:
    def __init__(self, mask_rate=0.15):
        self.mask_rate = mask_rate
        
    def gen_mask_index(self, feat_label_list, mask_rate=0.15):
        """掩蔽位置生成算法的可视化版本"""
        allow_mask_pos = torch.cat([
            feat_label.sum(dim=-1, keepdim=True) == 1 for feat_label in feat_label_list
        ], dim=-1).prod(dim=-1)
        
        origin_shape = allow_mask_pos.shape
        origin_shape_flat = origin_shape.numel()
        n_mask = max(int(mask_rate * origin_shape_flat), 1)
        mask_index = torch.randperm(origin_shape_flat)[:n_mask]
        mask = torch.zeros(origin_shape_flat).index_fill_(-1, mask_index, 1)
        mask = mask.reshape(origin_shape).bool()
        
        print(f"总节点数: {origin_shape_flat}")
        print(f"允许掩蔽的节点数: {allow_mask_pos.sum().item()}")
        print(f"实际掩蔽的节点数: {n_mask}")
        print(f"掩蔽率: {n_mask/origin_shape_flat:.3f}")
        
        return mask, allow_mask_pos
    
    def demo_protein_masking(self):
        """演示蛋白质节点掩蔽"""
        print("=" * 60)
        print("蛋白质节点掩蔽演示")
        print("=" * 60)
        
        # 模拟蛋白质节点特征
        n_protein_nodes = 100
        
        # 原子类型特征 (37维，one-hot编码)
        p_atom_label = torch.zeros(n_protein_nodes, 37)
        atom_types = torch.randint(0, 37, (n_protein_nodes,))
        p_atom_label[torch.arange(n_protein_nodes), atom_types] = 1
        
        # 残基类型特征 (20维，one-hot编码)  
        p_res_label = torch.zeros(n_protein_nodes, 20)
        res_types = torch.randint(0, 20, (n_protein_nodes,))
        p_res_label[torch.arange(n_protein_nodes), res_types] = 1
        
        # 生成掩蔽
        mask, allow_mask = self.gen_mask_index([p_atom_label, p_res_label], self.mask_rate)
        
        # 模拟掩蔽过程
        protein_features = torch.cat([torch.randn(n_protein_nodes, 50), p_res_label, p_atom_label], dim=-1)
        print(f"原始特征维度: {protein_features.shape}")
        
        # 添加掩蔽标记维度
        protein_features_masked = F.pad(protein_features, (0, 1), 'constant', 0)
        protein_features_masked[mask] = 0  # 掩蔽位置置零
        protein_features_masked[mask, -1] = 1  # 添加掩蔽标记
        
        print(f"掩蔽后特征维度: {protein_features_masked.shape}")
        print(f"掩蔽标记数量: {protein_features_masked[:, -1].sum().item()}")
        
        return mask, p_atom_label, p_res_label
    
    def demo_ligand_masking(self):
        """演示配体节点掩蔽"""
        print("\n" + "=" * 60)
        print("配体节点掩蔽演示")
        print("=" * 60)
        
        # 模拟配体节点特征
        n_ligand_nodes = 30
        
        # 原子类型特征 (10维，one-hot编码)
        l_atom_label = torch.zeros(n_ligand_nodes, 10)
        atom_types = torch.randint(0, 10, (n_ligand_nodes,))
        l_atom_label[torch.arange(n_ligand_nodes), atom_types] = 1
        
        # 生成掩蔽
        mask, allow_mask = self.gen_mask_index([l_atom_label], self.mask_rate)
        
        # 模拟掩蔽过程
        ligand_features = torch.cat([l_atom_label, torch.randn(n_ligand_nodes, 20)], dim=-1)
        print(f"原始特征维度: {ligand_features.shape}")
        
        # 添加掩蔽标记维度
        ligand_features_masked = F.pad(ligand_features, (0, 1), 'constant', 0)
        ligand_features_masked[mask] = 0  # 掩蔽位置置零
        ligand_features_masked[mask, -1] = 1  # 添加掩蔽标记
        
        print(f"掩蔽后特征维度: {ligand_features_masked.shape}")
        print(f"掩蔽标记数量: {ligand_features_masked[:, -1].sum().item()}")
        
        return mask, l_atom_label
    
    def demo_edge_masking(self):
        """演示边特征掩蔽"""
        print("\n" + "=" * 60)
        print("边特征掩蔽演示")
        print("=" * 60)
        
        # 模拟边特征矩阵
        n_nodes = 50
        edge_dim = 8
        
        # 生成对称的边特征矩阵
        edge_features = torch.randn(n_nodes, n_nodes, edge_dim)
        edge_features = (edge_features + edge_features.transpose(0, 1)) / 2  # 保证对称性
        
        # 边类型标签 (假设每条边有一个类型标签)
        edge_label = torch.randint(0, edge_dim, (n_nodes, n_nodes))
        edge_label = (edge_label + edge_label.T) / 2  # 保证对称性
        edge_label_onehot = F.one_hot(edge_label.long(), edge_dim).float()
        
        # 生成掩蔽（只掩蔽上三角，然后镜像到下三角）
        mask_upper = self.gen_mask_index([edge_label_onehot], self.mask_rate)[0].triu()
        edge_mask = mask_upper + mask_upper.T
        
        print(f"边特征矩阵形状: {edge_features.shape}")
        print(f"掩蔽的边数: {edge_mask.sum().item() // 2}")  # 除以2因为对称
        
        # 模拟掩蔽过程
        edge_features_masked = F.pad(edge_features, (0, 1), 'constant', 0)
        edge_features_masked[edge_mask] = 0  # 掩蔽位置置零
        edge_features_masked[edge_mask, -1] = 1  # 添加掩蔽标记
        
        return edge_mask, edge_features
    
    def demo_coordinate_noise(self):
        """演示坐标噪声添加"""
        print("\n" + "=" * 60)
        print("坐标噪声演示")
        print("=" * 60)
        
        # 模拟蛋白质坐标
        n_protein_atoms = 100
        protein_coords = torch.randn(n_protein_atoms, 3) * 10  # 模拟真实坐标范围
        
        # 模拟残基标签用于噪声掩蔽
        p_res_label = torch.zeros(n_protein_atoms, 20)
        res_types = torch.randint(0, 20, (n_protein_atoms,))
        p_res_label[torch.arange(n_protein_atoms), res_types] = 1
        
        # 生成噪声掩蔽
        noise_mask, _ = self.gen_mask_index([p_res_label], self.mask_rate)
        
        # 添加噪声
        noise_distance = 2.0
        coor_scale = 1.0
        
        coords_with_noise = protein_coords.clone()
        coords_with_noise[noise_mask] += torch.randn(coords_with_noise[noise_mask].shape) * noise_distance / coor_scale
        
        # 计算噪声前后的差异
        coord_diff = torch.norm(coords_with_noise - protein_coords, dim=-1)
        
        print(f"添加噪声的原子数: {noise_mask.sum().item()}")
        print(f"平均坐标偏移: {coord_diff[noise_mask].mean().item():.3f}")
        print(f"最大坐标偏移: {coord_diff[noise_mask].max().item():.3f}")
        
        return noise_mask, protein_coords, coords_with_noise
    
    def visualize_masking_pattern(self, mask, title="掩蔽模式"):
        """可视化掩蔽模式"""
        plt.figure(figsize=(12, 4))
        
        if len(mask.shape) == 1:
            # 1D掩蔽（节点）
            plt.subplot(1, 2, 1)
            mask_array = mask.float().numpy()
            plt.plot(mask_array, 'o-', markersize=3)
            plt.title(f"{title} - 节点掩蔽")
            plt.xlabel("节点索引")
            plt.ylabel("是否掩蔽")
            
            plt.subplot(1, 2, 2)
            plt.hist(np.where(mask_array)[0], bins=20, alpha=0.7)
            plt.title("掩蔽位置分布")
            plt.xlabel("节点索引")
            plt.ylabel("频率")
            
        elif len(mask.shape) == 2:
            # 2D掩蔽（边）
            plt.subplot(1, 2, 1)
            sns.heatmap(mask.float().numpy(), cmap='RdYlBu_r', cbar=True)
            plt.title(f"{title} - 边掩蔽")
            
            plt.subplot(1, 2, 2)
            mask_density = mask.float().sum(dim=0).numpy()
            plt.plot(mask_density)
            plt.title("每个节点的掩蔽边数")
            plt.xlabel("节点索引")
            plt.ylabel("掩蔽边数")
        
        plt.tight_layout()
        plt.show()

def main():
    """主演示函数"""
    print("PDBbind半监督掩蔽机制演示")
    print("=" * 80)
    
    visualizer = MaskingVisualization(mask_rate=0.15)
    
    # 演示各种掩蔽
    p_mask, p_atom, p_res = visualizer.demo_protein_masking()
    l_mask, l_atom = visualizer.demo_ligand_masking()
    e_mask, e_feat = visualizer.demo_edge_masking()
    n_mask, orig_coords, noisy_coords = visualizer.demo_coordinate_noise()
    
    print("\n" + "=" * 80)
    print("半监督学习的关键思想:")
    print("1. 通过掩蔽部分特征，强制模型学习分子的内在表示")
    print("2. 无标签数据可以用于自监督的特征重建任务")
    print("3. 多层次掩蔽（节点、边、坐标）提供丰富的学习信号")
    print("4. 对称性保持确保物理合理性")
    print("5. 渐进式学习平衡监督和自监督任务")
    
    # 如果需要可视化（需要matplotlib）
    try:
        import matplotlib.pyplot as plt
        print("\n生成可视化图表...")
        visualizer.visualize_masking_pattern(p_mask, "蛋白质节点掩蔽")
        visualizer.visualize_masking_pattern(l_mask, "配体节点掩蔽")
        visualizer.visualize_masking_pattern(e_mask, "边掩蔽")
    except ImportError:
        print("matplotlib未安装，跳过可视化")

if __name__ == "__main__":
    main()
