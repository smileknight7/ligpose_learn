#!/usr/bin/env python3
"""
根据.npz文件列表筛选pdbbind_screening_allow_dict.pkl文件
"""

import os
import pickle
import argparse
import csv
from collections import defaultdict


def scan_npz_files(npz_directory):
    """
    扫描目录中的所有.npz文件
    
    Args:
        npz_directory: 包含.npz文件的目录
    
    Returns:
        tuple: (pdb_ids, npz_files) 或 ([], []) 如果出错
    """
    if not os.path.exists(npz_directory):
        print(f"错误: 目录不存在 {npz_directory}")
        return [], []
    
    npz_files = []
    pdb_ids = []
    
    print(f"扫描目录: {npz_directory}")
    
    for file in os.listdir(npz_directory):
        if file.endswith('.npz'):
            npz_files.append(file)
            # 提取PDB ID（文件名去掉.npz后缀）
            pdb_id = file.replace('.npz', '')
            pdb_ids.append(pdb_id)
    
    print(f"找到 {len(npz_files)} 个.npz文件")
    return pdb_ids, npz_files


def save_file_list(pdb_ids, npz_files, output_dir):
    """
    保存文件列表到txt和csv
    
    Args:
        pdb_ids: PDB ID列表
        npz_files: npz文件名列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为txt格式
    txt_file = os.path.join(output_dir, "npz_file_list.txt")
    with open(txt_file, 'w') as f:
        f.write("# NPZ文件列表\n")
        f.write(f"# 总数: {len(npz_files)}\n")
        f.write("# 格式: PDB_ID\n")
        f.write("=" * 40 + "\n\n")
        
        for pdb_id in sorted(pdb_ids):
            f.write(f"{pdb_id}\n")
    
    # 保存为csv格式
    csv_file = os.path.join(output_dir, "npz_file_list.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pdb_id', 'npz_filename'])
        
        for pdb_id, npz_file in zip(sorted(pdb_ids), sorted(npz_files)):
            writer.writerow([pdb_id, npz_file])
    
    print(f"文件列表已保存:")
    print(f"  TXT: {txt_file}")
    print(f"  CSV: {csv_file}")


def filter_pkl_by_npz_list(original_pkl_path, pdb_ids, output_path):
    """
    根据npz文件的PDB ID列表筛选原始pkl文件
    
    Args:
        original_pkl_path: 原始pkl文件路径
        pdb_ids: 要保留的PDB ID列表
        output_path: 输出的筛选后pkl文件路径
    """
    try:
        # 读取原始pkl文件
        print(f"读取原始pkl文件: {original_pkl_path}")
        with open(original_pkl_path, 'rb') as f:
            original_dict = pickle.load(f)
        
        print(f"原始字典包含 {len(original_dict)} 个条目")
        
        # 将pdb_ids转换为set以提高查找效率
        pdb_id_set = set(pdb_ids)
        
        # 筛选字典
        filtered_dict = {}
        
        for ligand_id, protein_list in original_dict.items():
            # 如果配体ID在npz文件列表中，保留这个条目
            if ligand_id in pdb_id_set:
                # 同时筛选蛋白质列表，只保留在npz文件列表中的蛋白质
                filtered_proteins = [p for p in protein_list if p in pdb_id_set]
                if filtered_proteins:  # 如果还有蛋白质剩余
                    filtered_dict[ligand_id] = filtered_proteins
        
        print(f"筛选后字典包含 {len(filtered_dict)} 个条目")
        
        # 保存筛选后的字典
        with open(output_path, 'wb') as f:
            pickle.dump(filtered_dict, f)
        
        print(f"筛选后的pkl文件已保存: {output_path}")
        
        # 生成统计信息
        stats_file = output_path.replace('.pkl', '_stats.txt')
        total_pairs = sum(len(proteins) for proteins in filtered_dict.values())
        avg_proteins = total_pairs / len(filtered_dict) if filtered_dict else 0
        
        with open(stats_file, 'w') as f:
            f.write("筛选后的映射关系统计信息\n")
            f.write("=" * 30 + "\n")
            f.write(f"原始条目数: {len(original_dict)}\n")
            f.write(f"筛选后条目数: {len(filtered_dict)}\n")
            f.write(f"可用的npz文件数: {len(pdb_ids)}\n")
            f.write(f"总的配体-蛋白质对数: {total_pairs}\n")
            f.write(f"平均每个配体对应的蛋白质数: {avg_proteins:.2f}\n")
        
        print(f"统计信息已保存: {stats_file}")
        
        # 生成筛选后的txt格式文件
        txt_output = output_path.replace('.pkl', '.txt')
        with open(txt_output, 'w') as f:
            f.write("# 筛选后的Ligand-Protein映射关系\n")
            f.write("# 基于可用的npz文件筛选\n")
            f.write(f"# 总条目数: {len(filtered_dict)}\n")
            f.write("=" * 50 + "\n\n")
            
            for ligand_id in sorted(filtered_dict.keys()):
                protein_ids = filtered_dict[ligand_id]
                f.write(f"Ligand: {ligand_id}\n")
                f.write(f"Allowed proteins ({len(protein_ids)}): {', '.join(protein_ids)}\n")
                f.write("-" * 40 + "\n")
        
        print(f"筛选后的txt文件已保存: {txt_output}")
        return True
        
    except Exception as e:
        print(f"筛选过程中出错: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='根据npz文件筛选pdbbind_screening_allow_dict.pkl')
    parser.add_argument('--npz_dir', type=str, 
                       default='/data/lpw/ligpose/data/work_file/tmp',
                       help='包含.npz文件的目录')
    parser.add_argument('--original_pkl', type=str,
                       default='/home/smileknight/learn/LigPose_demo_linux/suppl/pdbbind_screening_allow_dict.pkl',
                       help='原始pkl文件路径')
    parser.add_argument('--output_dir', type=str,
                       default='/home/smileknight/learn/LigPose_demo_linux/suppl/',
                       help='输出目录')
    parser.add_argument('--mode', choices=['scan', 'filter', 'both'], default='both',
                       help='运行模式: scan=只扫描npz文件, filter=只筛选pkl, both=两者都做')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NPZ文件筛选和PKL文件重建工具")
    print("=" * 60)
    print(f"NPZ目录: {args.npz_dir}")
    print(f"原始PKL: {args.original_pkl}")
    print(f"输出目录: {args.output_dir}")
    print(f"运行模式: {args.mode}")
    print("-" * 60)
    
    if args.mode in ['scan', 'both']:
        # 扫描npz文件
        pdb_ids, npz_files = scan_npz_files(args.npz_dir)
        
        if not pdb_ids:
            print("未找到任何npz文件，程序退出")
            return
        
        # 保存文件列表
        save_file_list(pdb_ids, npz_files, args.output_dir)
    
    if args.mode in ['filter', 'both']:
        # 如果模式是filter且之前没有扫描，需要先扫描
        if args.mode == 'filter':
            pdb_ids, _ = scan_npz_files(args.npz_dir)
            if not pdb_ids:
                print("未找到任何npz文件，程序退出")
                return
        
        # 筛选pkl文件
        output_pkl = os.path.join(args.output_dir, "filtered_pdbbind_screening_allow_dict.pkl")
        success = filter_pkl_by_npz_list(args.original_pkl, pdb_ids, output_pkl)
        
        if success:
            print("\n" + "=" * 60)
            print("筛选完成！")
            print("现在你可以使用筛选后的pkl文件来确保与npz文件一一对应。")
            print("=" * 60)
        else:
            print("筛选失败！")


if __name__ == '__main__':
    main()
