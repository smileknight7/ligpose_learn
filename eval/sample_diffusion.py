"""
使用训练好的Diffusion模型进行分子构象采样
"""

import os
import sys
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))

import argparse
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 导入现有模块
from model.layers import LigPoseStruct, LigPoseScr
from model.diffusion import DiffusionWrapper
from utils.pdbbind_utils import ComplexStructDataset
from utils.common import *

def plot_3d_coordinates(coordinates, title, save_path=None):
    """
    绘制3D坐标点云
    
    Args:
        coordinates: 坐标数组，形状为[N, 3]
        title: 图表标题
        save_path: 保存路径，如果为None则显示图表
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点云
    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c='b', marker='o', s=20)
    
    # 设置标题和轴标签
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 保存或显示
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def sample_from_diffusion(args):
    """
    从训练好的Diffusion模型采样生成分子构象
    
    Args:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() and args.gpu_id >= 0 else 'cpu')
    
    # 加载模型
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model_args = checkpoint['args']
    
    # 根据模型类型创建基础模型
    if args.model_type == 'struct':
        base_model = LigPoseStruct(model_args).to(device)
    elif args.model_type == 'scr':
        base_model = LigPoseScr(model_args).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # 创建Diffusion包装器
    model = DiffusionWrapper(
        base_model=base_model,
        timesteps=args.diffusion_timesteps,
        beta_start=args.diffusion_beta_start,
        beta_end=args.diffusion_beta_end,
        schedule_type=args.diffusion_schedule_type
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载测试数据
    print(f"Loading test data from {args.data_path}...")
    test_dataset = ComplexStructDataset('test', model_args, [args.data_path])
    test_data = test_dataset[0].to(device)
    
    # 采样生成分子构象
    print("Sampling molecular conformations...")
    with torch.no_grad():
        # 使用不同的采样步数
        for sample_steps in args.sample_steps:
            print(f"Sampling with {sample_steps} steps...")
            
            # 从模型采样
            if args.return_all_steps:
                # 返回所有中间步骤
                samples = model.sample(test_data, timesteps=sample_steps, return_all=True)
                
                # 创建保存目录
                save_dir = Path(args.output_dir) / f"sample_{sample_steps}_steps"
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存所有中间步骤
                for i, sample in enumerate(samples):
                    # 提取配体坐标
                    ligand_coords = sample[test_data.ligand_mask].cpu().numpy()
                    
                    # 绘制并保存
                    step_title = f"Step {i}/{sample_steps}"
                    save_path = save_dir / f"step_{i:04d}.png"
                    plot_3d_coordinates(ligand_coords, step_title, save_path)
                    
                    # 保存坐标
                    np.save(save_dir / f"coords_{i:04d}.npy", ligand_coords)
            else:
                # 只返回最终结果
                sample = model.sample(test_data, timesteps=sample_steps, return_all=False)
                
                # 提取配体坐标
                ligand_coords = sample[test_data.ligand_mask].cpu().numpy()
                
                # 创建保存目录
                save_dir = Path(args.output_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # 绘制并保存
                title = f"Final Sample ({sample_steps} steps)"
                save_path = save_dir / f"final_sample_{sample_steps}_steps.png"
                plot_3d_coordinates(ligand_coords, title, save_path)
                
                # 保存坐标
                np.save(save_dir / f"final_coords_{sample_steps}_steps.npy", ligand_coords)
    
    print(f"Sampling complete. Results saved to {args.output_dir}")


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained diffusion model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the test data')
    parser.add_argument('--output_dir', type=str, default='./diffusion_samples', help='Directory to save the samples')
    parser.add_argument('--model_type', type=str, default='struct', choices=['struct', 'scr'], help='Model type: struct or scr')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--diffusion_timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--diffusion_beta_start', type=float, default=1e-4, help='Initial beta value for noise schedule')
    parser.add_argument('--diffusion_beta_end', type=float, default=0.02, help='Final beta value for noise schedule')
    parser.add_argument('--diffusion_schedule_type', type=str, default='linear', choices=['linear', 'cosine'], help='Type of noise schedule')
    parser.add_argument('--sample_steps', type=int, nargs='+', default=[50, 100, 200, 1000], help='Number of steps to use for sampling')
    parser.add_argument('--return_all_steps', type=bool, default=False, help='Whether to return all intermediate steps')
    
    args = parser.parse_args()
    
    # 开始采样
    sample_from_diffusion(args)
