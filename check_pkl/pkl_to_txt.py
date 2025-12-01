#!/usr/bin/env python3
"""
将 pdbbind_screening_allow_dict.pkl 文件转换为 txt 格式的脚本
"""

import pickle
import argparse
import os


def pkl_to_txt(pkl_file_path, output_txt_path):
    """
    将 pkl 文件中的字典数据转换为 txt 格式
    
    Args:
        pkl_file_path: pkl 文件路径
        output_txt_path: 输出的 txt 文件路径
    """
    try:
        # 读取 pkl 文件
        with open(pkl_file_path, 'rb') as f:
            allow_dict = pickle.load(f)
        
        print(f"成功读取 pkl 文件: {pkl_file_path}")
        print(f"字典包含 {len(allow_dict)} 个键值对")
        
        # 写入 txt 文件
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write("# PDBbind Screening Allow Dictionary\n")
            f.write("# Format: ligand_pdb_id -> [protein_pdb_ids]\n")
            f.write("# Total entries: {}\n".format(len(allow_dict)))
            f.write("=" * 50 + "\n\n")
            
            # 按照键排序输出
            for ligand_id in sorted(allow_dict.keys()):
                protein_ids = allow_dict[ligand_id]
                f.write(f"Ligand: {ligand_id}\n")
                f.write(f"Allowed proteins ({len(protein_ids)}): {', '.join(protein_ids)}\n")
                f.write("-" * 40 + "\n")
        
        print(f"成功保存为 txt 文件: {output_txt_path}")
        
        # 打印一些统计信息
        total_pairs = sum(len(proteins) for proteins in allow_dict.values())
        print(f"\n统计信息:")
        print(f"- 配体数量: {len(allow_dict)}")
        print(f"- 总的配体-蛋白质对数: {total_pairs}")
        print(f"- 平均每个配体对应的蛋白质数: {total_pairs / len(allow_dict):.2f}")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {pkl_file_path}")
    except Exception as e:
        print(f"错误: {e}")


def pkl_to_csv(pkl_file_path, output_csv_path):
    """
    将 pkl 文件中的字典数据转换为 CSV 格式
    
    Args:
        pkl_file_path: pkl 文件路径
        output_csv_path: 输出的 CSV 文件路径
    """
    try:
        # 读取 pkl 文件
        with open(pkl_file_path, 'rb') as f:
            allow_dict = pickle.load(f)
        
        # 写入 CSV 文件
        with open(output_csv_path, 'w', encoding='utf-8') as f:
            f.write("ligand_pdb_id,protein_pdb_id\n")
            
            for ligand_id, protein_ids in allow_dict.items():
                for protein_id in protein_ids:
                    f.write(f"{ligand_id},{protein_id}\n")
        
        print(f"成功保存为 CSV 文件: {output_csv_path}")
        
    except Exception as e:
        print(f"错误: {e}")


def main():
    parser = argparse.ArgumentParser(description='将 pkl 文件转换为 txt 或 CSV 格式')
    parser.add_argument('--pkl_file', type=str, 
                       default='./suppl/filtered_pdbbind_screening_allow_dict.pkl',
                       help='输入的 pkl 文件路径')
    parser.add_argument('--output_dir', type=str,
                       default='./suppl/',
                       help='输出目录')
    parser.add_argument('--format', choices=['txt', 'csv', 'both'], default='txt',
                       help='输出格式: txt, csv, 或 both')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成输出文件路径
    base_name = os.path.splitext(os.path.basename(args.pkl_file))[0]
    txt_output = os.path.join(args.output_dir, f"{base_name}.txt")
    csv_output = os.path.join(args.output_dir, f"{base_name}.csv")
    
    print(f"输入文件: {args.pkl_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"输出格式: {args.format}")
    print("-" * 50)
    
    # 执行转换
    if args.format in ['txt', 'both']:
        pkl_to_txt(args.pkl_file, txt_output)
    
    if args.format in ['csv', 'both']:
        pkl_to_csv(args.pkl_file, csv_output)


if __name__ == '__main__':
    main()
