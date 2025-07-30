#!/usr/bin/env python3
"""
PDBbind数据处理脚本 - 支持生成配体-蛋白质映射关系
"""

import os
import re
import sys
import shutil
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import Pool
import argparse
from collections import defaultdict
# from tqdm import tqdm  # 如果需要进度条，取消注释


info_path = r'/home/smileknight/learn/ligpose_data/INDEX_refined_set.txt'


def get_aff(info_path):
    with open(info_path, 'r') as f:
        lines = f.readlines()
    dic_aff = {}
    unit_conversion = {
        'nM': 1,
        'uM': 1e3,
        'mM': 1e6,
        'pM': 1e-3,
    }
    # 建立字典来方便对单位进行换算

    for line in lines:
        if not line.startswith('#'):
            pdb_id = line[:4]
            # 使用正则表达式提取数值和单位
            match = re.search(r'(K[di]=)([\d\.]+)([a-zA-Z]+)', line)
            if match:
                _, value_str, unit = match.groups()
                # 这里是进行了解包操作使用_, 放弃对第一个元素解包，try:可以用来捕捉异常数据
                try:
                    value = float(value_str)  # 转换为浮点数
                    if unit in unit_conversion:
                        value *= unit_conversion[unit]  # 转换为 nM 单位
                        dic_aff[pdb_id] = value
                    else:
                        print(f"未知单位: {unit}，跳过该行")
                except ValueError:
                    print(f"无法解析数值: {value_str}，跳过该行")
            else:
                print(f"未找到数值和单位，跳过该行: {line.strip()}")
    return dic_aff


def read_mol_from_pdbbind(data_path, pdb_id):
    ligand_mol2_path = f'{data_path}/{pdb_id}/{pdb_id}_ligand.mol2'

    # 检查文件是否存在
    if not os.path.exists(ligand_mol2_path):
        print(f"文件 {ligand_mol2_path} 不存在")
        return None

    ligand_mol = Chem.MolFromMol2File(ligand_mol2_path)

    if ligand_mol is not None:
        ligand_mol_addHs = Chem.AddHs(ligand_mol)
        print(f"成功读取并添加氢原子到 {pdb_id} 的分子")
        return ligand_mol_addHs  # 返回添加氢原子的分子
    else:
        print(f"读取 {pdb_id} 的分子失败")
        return None


def generate_ligand_protein_mapping(data_path, suppl_path, output_path, limit=None):
    """
    生成配体-蛋白质映射关系
    
    Args:
        data_path: PDB数据目录
        suppl_path: 亲和力数据文件路径
        output_path: 输出目录
    """
    print("开始生成配体-蛋白质映射关系...")
    
    # 读取亲和力数据
    try:
        dic_aff = get_aff(suppl_path)
        print(f"成功读取 {len(dic_aff)} 个PDB的亲和力数据")
    except Exception as e:
        print(f"读取亲和力数据失败: {e}")
        return False
    
    # 创建映射字典
    ligand_protein_map = defaultdict(list)
    ligand_info = {}  # 存储配体信息
    
    # 获取所有可用的PDB条目
    if not os.path.exists(data_path):
        print(f"数据路径不存在: {data_path}")
        return False
    
    pdb_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    print(f"找到 {len(pdb_dirs)} 个PDB目录")
    
    # 根据limit参数决定是否限制处理数量
    if limit is not None:
        pdb_dirs = pdb_dirs[:limit]
        print(f"限制处理前 {limit} 个条目")
    else:
        print("处理所有PDB条目")
    
    processed_count = 0
    for pdb_id in pdb_dirs:
        try:
            # 检查是否有配体文件
            ligand_file = os.path.join(data_path, pdb_id, f"{pdb_id}_ligand.mol2")
            protein_file = os.path.join(data_path, pdb_id, f"{pdb_id}_protein.pdb")
            
            if not os.path.exists(ligand_file) or not os.path.exists(protein_file):
                continue
            
            # 读取配体分子
            ligand_mol = Chem.MolFromMol2File(ligand_file)
            if ligand_mol is None:
                continue
            
            # 获取配体的SMILES作为标识符
            ligand_smiles = Chem.MolToSmiles(ligand_mol)
            
            # 基于SMILES分组相似的配体
            # 这里我们使用一个简化的方法：使用配体的分子式作为分组依据
            mol_formula = Chem.rdMolDescriptors.CalcMolFormula(ligand_mol)
            
            # 将蛋白质PDB ID添加到对应配体分组中
            ligand_protein_map[pdb_id].append(pdb_id)  # 自己对自己肯定是允许的
            
            # 查找具有相似配体的其他PDB
            for other_pdb_id in pdb_dirs:
                if other_pdb_id == pdb_id:
                    continue
                    
                other_ligand_file = os.path.join(data_path, other_pdb_id, f"{other_pdb_id}_ligand.mol2")
                if not os.path.exists(other_ligand_file):
                    continue
                
                other_ligand_mol = Chem.MolFromMol2File(other_ligand_file)
                if other_ligand_mol is None:
                    continue
                
                other_mol_formula = Chem.rdMolDescriptors.CalcMolFormula(other_ligand_mol)
                
                # 如果分子式相同，认为是相似的配体
                if mol_formula == other_mol_formula:
                    if other_pdb_id not in ligand_protein_map[pdb_id]:
                        ligand_protein_map[pdb_id].append(other_pdb_id)
            
            ligand_info[pdb_id] = {
                'smiles': ligand_smiles,
                'formula': mol_formula,
                'affinity': dic_aff.get(pdb_id, None)
            }
            
            processed_count += 1
            print(f"已处理 {processed_count} 个PDB条目: {pdb_id}")
                
        except Exception as e:
            print(f"处理 {pdb_id} 时出错: {e}")
            continue
    
    print(f"完成处理，共处理 {processed_count} 个PDB条目")
    print(f"生成了 {len(ligand_protein_map)} 个配体-蛋白质映射关系")
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 保存映射关系为pkl文件
    mapping_file = os.path.join(output_path, "ligand_protein_mapping.pkl")
    with open(mapping_file, 'wb') as f:
        pickle.dump(dict(ligand_protein_map), f)
    
    # 保存配体信息
    info_file = os.path.join(output_path, "ligand_info.pkl")
    with open(info_file, 'wb') as f:
        pickle.dump(ligand_info, f)
    
    print(f"映射关系已保存到: {mapping_file}")
    print(f"配体信息已保存到: {info_file}")
    
    # 生成统计信息和txt格式
    total_pairs = sum(len(proteins) for proteins in ligand_protein_map.values())
    avg_proteins_per_ligand = total_pairs / len(ligand_protein_map) if ligand_protein_map else 0
    
    # 生成类似于原来格式的txt文件
    txt_file = os.path.join(output_path, "ligand_protein_mapping.txt")
    with open(txt_file, 'w') as f:
        f.write("# Ligand-Protein Mapping Dictionary\n")
        f.write("# Format: ligand_pdb_id -> [protein_pdb_ids]\n")
        f.write("# Total entries: {}\n".format(len(ligand_protein_map)))
        f.write("=" * 50 + "\n\n")
        
        # 按照键排序输出
        for ligand_id in sorted(ligand_protein_map.keys()):
            protein_ids = ligand_protein_map[ligand_id]
            f.write(f"Ligand: {ligand_id}\n")
            f.write(f"Allowed proteins ({len(protein_ids)}): {', '.join(protein_ids)}\n")
            f.write("-" * 40 + "\n")
    
    stats_file = os.path.join(output_path, "mapping_stats.txt")
    with open(stats_file, 'w') as f:
        f.write("配体-蛋白质映射统计信息\n")
        f.write("=" * 30 + "\n")
        f.write(f"配体数量: {len(ligand_protein_map)}\n")
        f.write(f"总的配体-蛋白质对数: {total_pairs}\n")
        f.write(f"平均每个配体对应的蛋白质数: {avg_proteins_per_ligand:.2f}\n")
        f.write(f"处理的PDB条目数: {processed_count}\n")
    
    print(f"TXT格式映射文件已保存到: {txt_file}")
    print(f"统计信息已保存到: {stats_file}")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PDBbind数据处理脚本')
    parser.add_argument('--data_path', type=str,
                         default='/home/smileknight/learn/ligpose_data/refined-set', help='数据路径')
    parser.add_argument('--data_suppl_path', type=str,
                         default='/home/smileknight/learn/ligpose_data/INDEX_refined_set.txt', help='补充数据路径')
    parser.add_argument('--output_path', type=str,
                         default='/home/smileknight/learn/ligpose_data/', help='输出路径')
    parser.add_argument('--cache', type=str,
                         default='/home/smileknight/learn/ligpose_data/cache', help='缓存路径')
    
    # 添加模式选择参数
    parser.add_argument('--mode', choices=['mapping'], default='mapping',
                       help='运行模式: mapping=生成映射关系')
    parser.add_argument('--limit', type=int, default=None,
                       help='限制处理的PDB条目数量（用于测试），默认处理所有')

    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_path, exist_ok=True)
    
    # 生成映射关系模式
    print("运行配体-蛋白质映射生成模式...")
    success = generate_ligand_protein_mapping(args.data_path, args.data_suppl_path, args.output_path, args.limit)
    if success:
        print("映射关系生成完成!")
    else:
        print("映射关系生成失败!")


if __name__ == '__main__':
    main()
