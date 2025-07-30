#!/usr/bin/env python3
"""
专门用于读取单个pkl文件并转换为txt格式的脚本
"""

import pickle
import argparse
import os
import numpy as np
from collections import defaultdict


def analyze_pkl_structure(data, indent=0):
    """
    递归分析pkl文件的数据结构
    
    Args:
        data: 要分析的数据
        indent: 缩进级别
    
    Returns:
        str: 格式化的结构描述
    """
    prefix = "  " * indent
    
    if isinstance(data, dict):
        result = f"{prefix}字典 (包含 {len(data)} 个键值对):\n"
        for key, value in data.items():
            result += f"{prefix}  键: {key} ({type(key).__name__})\n"
            if isinstance(value, (list, tuple, np.ndarray)):
                if hasattr(value, 'shape'):
                    result += f"{prefix}    值: {type(value).__name__} 形状: {value.shape}\n"
                else:
                    result += f"{prefix}    值: {type(value).__name__} 长度: {len(value)}\n"
                # 显示前几个元素
                if len(value) > 0:
                    if isinstance(value[0], (int, float, str)):
                        preview = str(value[:3]) + "..." if len(value) > 3 else str(value)
                        result += f"{prefix}    前几个元素: {preview}\n"
            elif isinstance(value, dict):
                result += analyze_pkl_structure(value, indent + 2)
            else:
                result += f"{prefix}    值: {value} ({type(value).__name__})\n"
    elif isinstance(data, (list, tuple)):
        result = f"{prefix}{type(data).__name__} (长度: {len(data)})\n"
        if len(data) > 0:
            result += f"{prefix}  第一个元素类型: {type(data[0]).__name__}\n"
            if len(data) <= 5:
                for i, item in enumerate(data):
                    result += f"{prefix}  [{i}]: {item}\n"
            else:
                result += f"{prefix}  前3个元素: {data[:3]}\n"
    elif isinstance(data, np.ndarray):
        result = f"{prefix}NumPy数组 形状: {data.shape}, 数据类型: {data.dtype}\n"
        if data.size <= 10:
            result += f"{prefix}  内容: {data}\n"
        else:
            result += f"{prefix}  部分内容: {data.flatten()[:5]}...\n"
    else:
        result = f"{prefix}{type(data).__name__}: {data}\n"
    
    return result


def pkl_to_detailed_txt(pkl_file_path, output_txt_path):
    """
    将pkl文件转换为详细的txt格式
    
    Args:
        pkl_file_path: pkl文件路径
        output_txt_path: 输出txt文件路径
    """
    try:
        print(f"正在读取文件: {pkl_file_path}")
        
        # 读取pkl文件
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"成功读取pkl文件")
        print(f"数据类型: {type(data)}")
        
        # 写入详细的txt文件
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"PKL文件详细分析报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"源文件: {pkl_file_path}\n")
            f.write(f"根数据类型: {type(data)}\n")
            f.write(f"生成时间: {os.path.getctime(pkl_file_path)}\n")
            f.write("=" * 60 + "\n\n")
            
            # 数据结构分析
            f.write("数据结构分析:\n")
            f.write("-" * 40 + "\n")
            structure_info = analyze_pkl_structure(data)
            f.write(structure_info)
            f.write("\n")
            
            # 详细内容
            f.write("详细内容:\n")
            f.write("-" * 40 + "\n")
            
            if isinstance(data, dict):
                for key, value in data.items():
                    f.write(f"\n[键] {key}:\n")
                    f.write(f"  类型: {type(value)}\n")
                    
                    if isinstance(value, np.ndarray):
                        f.write(f"  形状: {value.shape}\n")
                        f.write(f"  数据类型: {value.dtype}\n")
                        if value.size <= 20:
                            f.write(f"  内容:\n{value}\n")
                        else:
                            f.write(f"  部分内容 (前5个元素):\n{value.flatten()[:5]}\n")
                    
                    elif isinstance(value, (list, tuple)):
                        f.write(f"  长度: {len(value)}\n")
                        if len(value) <= 10:
                            f.write(f"  内容: {value}\n")
                        else:
                            f.write(f"  前5个元素: {value[:5]}\n")
                    
                    elif isinstance(value, (int, float, str)):
                        f.write(f"  值: {value}\n")
                    
                    else:
                        f.write(f"  值: {str(value)[:200]}{'...' if len(str(value)) > 200 else ''}\n")
            
            elif isinstance(data, (list, tuple)):
                f.write(f"这是一个{type(data).__name__}，包含 {len(data)} 个元素\n")
                for i, item in enumerate(data[:10]):  # 只显示前10个
                    f.write(f"  [{i}]: {type(item).__name__} = {str(item)[:100]}\n")
                if len(data) > 10:
                    f.write(f"  ... 还有 {len(data) - 10} 个元素\n")
            
            else:
                f.write(f"数据内容:\n{str(data)}\n")
        
        print(f"成功保存为txt文件: {output_txt_path}")
        
        # 打印基本统计信息
        if isinstance(data, dict):
            print(f"\n基本信息:")
            print(f"- 字典包含 {len(data)} 个键值对")
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    print(f"- {key}: NumPy数组 {value.shape}")
                elif isinstance(value, (list, tuple)):
                    print(f"- {key}: {type(value).__name__} 长度 {len(value)}")
                else:
                    print(f"- {key}: {type(value).__name__}")
        
        return True
        
    except Exception as e:
        print(f"转换失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='将pkl文件转换为详细的txt格式')
    parser.add_argument('--pkl_file', type=str, 
                       default='~/learn/ligpose_data/test/1a4k.pkl',
                       help='输入的pkl文件路径')
    parser.add_argument('--output', type=str,
                       help='输出txt文件路径（如果不指定，会自动生成）')
    
    args = parser.parse_args()
    
    # 展开用户路径
    pkl_file = os.path.expanduser(args.pkl_file)
    
    # 检查文件是否存在
    if not os.path.exists(pkl_file):
        print(f"错误: 文件不存在 {pkl_file}")
        return
    
    # 生成输出文件名
    if args.output:
        output_file = args.output
    else:
        base_name = os.path.splitext(os.path.basename(pkl_file))[0]
        output_dir = os.path.dirname(pkl_file)
        output_file = os.path.join(output_dir, f"{base_name}_detailed.txt")
    
    print("=" * 60)
    print("PKL文件详细分析工具")
    print("=" * 60)
    print(f"输入文件: {pkl_file}")
    print(f"输出文件: {output_file}")
    print("-" * 60)
    
    # 执行转换
    success = pkl_to_detailed_txt(pkl_file, output_file)
    
    if success:
        print("\n" + "=" * 60)
        print("转换完成！")
        print(f"详细分析结果已保存到: {output_file}")
        print("=" * 60)
    else:
        print("转换失败！")


if __name__ == '__main__':
    main()
