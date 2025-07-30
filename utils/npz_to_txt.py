#!/usr/bin/env python3
"""
读取npz文件并转换为txt格式的脚本
"""

import numpy as np
import argparse
import os


def npz_to_txt(npz_file_path, output_txt_path):
    """
    将npz文件转换为txt格式
    
    Args:
        npz_file_path: npz文件路径
        output_txt_path: 输出txt文件路径
    """
    try:
        # 读取npz文件
        print(f"正在读取npz文件: {npz_file_path}")
        data = np.load(npz_file_path, allow_pickle=True)
        
        print(f"成功读取npz文件，包含 {len(data.files)} 个数组")
        
        # 写入txt文件
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write("=== NPZ文件内容分析 ===\n")
            f.write(f"源文件: {npz_file_path}\n")
            f.write(f"包含数组数量: {len(data.files)}\n")
            f.write("=" * 50 + "\n\n")
            
            # 遍历所有数组
            for i, key in enumerate(data.files, 1):
                array = data[key]
                f.write(f"[{i}] 数组名称: {key}\n")
                f.write(f"    数据类型: {array.dtype}\n")
                f.write(f"    形状: {array.shape}\n")
                f.write(f"    维度: {array.ndim}\n")
                
                if array.size > 0:
                    if array.ndim == 0:  # 标量
                        f.write(f"    值: {array.item()}\n")
                    elif array.ndim == 1:  # 一维数组
                        if array.size <= 20:  # 小数组全部显示
                            f.write(f"    内容: {array.tolist()}\n")
                        else:  # 大数组显示部分
                            f.write(f"    前10个元素: {array[:10].tolist()}\n")
                            f.write(f"    后10个元素: {array[-10:].tolist()}\n")
                    elif array.ndim == 2:  # 二维数组
                        f.write(f"    形状: {array.shape[0]} x {array.shape[1]}\n")
                        if array.size <= 100:  # 小矩阵全部显示
                            f.write("    内容:\n")
                            for row in array:
                                f.write(f"      {row.tolist()}\n")
                        else:  # 大矩阵显示部分
                            f.write("    前3行:\n")
                            for row in array[:3]:
                                if len(row) <= 10:
                                    f.write(f"      {row.tolist()}\n")
                                else:
                                    f.write(f"      前10列: {row[:10].tolist()}\n")
                            if array.shape[0] > 6:
                                f.write("    ...\n")
                                f.write("    后3行:\n")
                                for row in array[-3:]:
                                    if len(row) <= 10:
                                        f.write(f"      {row.tolist()}\n")
                                    else:
                                        f.write(f"      前10列: {row[:10].tolist()}\n")
                    else:  # 高维数组
                        f.write(f"    高维数组，形状: {array.shape}\n")
                        if array.size <= 50:
                            f.write(f"    扁平化前50个元素: {array.flatten()[:50].tolist()}\n")
                else:
                    f.write("    空数组\n")
                
                f.write("-" * 40 + "\n")
            
            # 统计信息
            f.write("\n=== 统计信息 ===\n")
            total_size = sum(data[key].size for key in data.files)
            f.write(f"总元素数: {total_size}\n")
            
            # 按数据类型分类
            dtype_stats = {}
            for key in data.files:
                dtype = str(data[key].dtype)
                if dtype in dtype_stats:
                    dtype_stats[dtype] += 1
                else:
                    dtype_stats[dtype] = 1
            
            f.write("数据类型分布:\n")
            for dtype, count in dtype_stats.items():
                f.write(f"  {dtype}: {count} 个数组\n")
        
        print(f"成功转换为txt文件: {output_txt_path}")
        return True
        
    except Exception as e:
        print(f"转换失败: {e}")
        return False


def npz_to_csv(npz_file_path, output_csv_path):
    """
    将npz文件中的数值数据转换为CSV格式
    
    Args:
        npz_file_path: npz文件路径
        output_csv_path: 输出CSV文件路径
    """
    try:
        import csv
        
        data = np.load(npz_file_path, allow_pickle=True)
        
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入头部信息
            writer.writerow(['array_name', 'shape', 'dtype', 'data'])
            
            for key in data.files:
                array = data[key]
                
                # 对于小的数组，将数据写入CSV
                if array.size <= 1000:  # 限制大小避免CSV过大
                    if array.ndim <= 2:
                        shape_str = 'x'.join(map(str, array.shape))
                        if array.ndim == 0:
                            data_str = str(array.item())
                        elif array.ndim == 1:
                            data_str = ';'.join(map(str, array.tolist()))
                        else:  # 2D
                            data_str = '|'.join([';'.join(map(str, row)) for row in array])
                        
                        writer.writerow([key, shape_str, str(array.dtype), data_str])
                    else:
                        writer.writerow([key, 'x'.join(map(str, array.shape)), str(array.dtype), 'high_dimensional_array'])
                else:
                    writer.writerow([key, 'x'.join(map(str, array.shape)), str(array.dtype), 'large_array'])
        
        print(f"成功保存为CSV文件: {output_csv_path}")
        return True
        
    except Exception as e:
        print(f"CSV转换失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='将npz文件转换为txt或CSV格式')
    parser.add_argument('--npz_file', type=str, 
                       default='/home/smileknight/learn/work_file/complex/1a1e-1a1e.npz',
                       help='输入的npz文件路径')
    parser.add_argument('--output_dir', type=str,
                       default='/home/smileknight/learn/work_file/output/',
                       help='输出目录')
    parser.add_argument('--format', choices=['txt', 'csv', 'both'], default='both',
                       help='输出格式: txt, csv, 或 both')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.npz_file):
        print(f"错误: NPZ文件不存在: {args.npz_file}")
        return
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成输出文件路径
    base_name = os.path.splitext(os.path.basename(args.npz_file))[0]
    txt_output = os.path.join(args.output_dir, f"{base_name}_content.txt")
    csv_output = os.path.join(args.output_dir, f"{base_name}_content.csv")
    
    print(f"输入文件: {args.npz_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"输出格式: {args.format}")
    print("-" * 50)
    
    # 执行转换
    success = True
    if args.format in ['txt', 'both']:
        success &= npz_to_txt(args.npz_file, txt_output)
    
    if args.format in ['csv', 'both']:
        success &= npz_to_csv(args.npz_file, csv_output)
    
    if success:
        print("\n" + "=" * 50)
        print("转换完成！")
        if args.format in ['txt', 'both']:
            print(f"TXT文件: {txt_output}")
        if args.format in ['csv', 'both']:
            print(f"CSV文件: {csv_output}")
        print("=" * 50)
    else:
        print("转换过程中出现错误！")


if __name__ == '__main__':
    main()
