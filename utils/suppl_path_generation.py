#问题1原本数据集中是否考虑单位问题




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
#建立字典来方便对单位进行换算

    for line in lines:
        if not line.startswith('#'):
            pdb_id = line[:4]
            # 使用正则表达式提取数值和单位
            match = re.search(r'(K[di]=)([\d\.]+)([a-zA-Z]+)', line)
            if match:
                _, value_str, unit = match.groups()
                #这里是进行了解包操作使用_, 放弃对第一个元素解包，try:可以用来捕捉异常数据
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

# 调用函数并打印结果
# dic_aff = get_aff(info_path)
# print(dic_aff)


def process(pdb_id, data_path, data_suppl_path, output_path, cache):
    """处理单个PDB条目的函数 - 生成pkl和npz文件"""
    try:
        print(f"Processing {pdb_id}...")
        
        # 读取分子数据
        ligand_mol = read_mol_from_pdbbind(data_path, pdb_id)
        if ligand_mol is None:
            print(f"无法读取 {pdb_id} 的分子数据")
            return False
        
        # 读取亲和力数据
        try:
            dic_aff = get_aff(data_suppl_path)
            if pdb_id not in dic_aff:
                print(f"警告: {pdb_id} 没有亲和力数据")
                aff = None
            else:
                aff = dic_aff[pdb_id]
        except Exception as e:
            print(f"读取亲和力数据失败: {e}")
            aff = None
        
        # 构建数据字典
        data_dict = {
            'pdb_id': pdb_id,
            'ligand_mol': ligand_mol,
            'affinity': aff,
            'processing_info': f"Processed by suppl_path_generation.py"
        }
        
        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)
        
        # 保存为 pkl 文件
        pkl_file = os.path.join(output_path, f"{pdb_id}.pkl")
        with open(pkl_file, 'wb') as f:
            pickle.dump(data_dict, f)
        
        # 也可以保存为 npz 文件（如果需要的话）
        npz_file = os.path.join(output_path, f"{pdb_id}_basic.npz")
        np.savez_compressed(npz_file, 
                           pdb_id=pdb_id,
                           affinity=aff if aff is not None else np.nan)
        
        print(f"成功处理 {pdb_id}: 保存到 {pkl_file}")
        return True
        
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return False


def try_prepare_pdbbind(task):
    """尝试准备PDBbind数据的包装函数"""
    try:
        func, args = task
        return func(*args)
    except Exception as e:
        print(f"Error in try_prepare_pdbbind: {e}")
        return False






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



#这里是对文件中的pdb分子文件进行了遍历，这个数据集中不可以有除了mol2文件以外的其他文件

def process_all_pdbs_in_folder(data_path):
    pdb_ids = [name for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]
    mols = {}
    for pdb_id in pdb_ids:
        ligand_mol = read_mol_from_pdbbind(data_path, pdb_id)
        if ligand_mol is not None:
            mols[pdb_id] = ligand_mol
        else:
            print(f"无法读取 {pdb_id} 的分子数据")
    return mols

data_path = r'/home/smileknight/learn/ligpose_data/refined-set'
# mols = process_all_pdbs_in_folder(data_path)
# print(f"mols 字典包含 {len(mols)} 个分子")
# #打印结果
# print(f"mols 字典包含 {len(mols)} 个分子")
# for pdb_id, mol in mols.items():
#     print(f"{pdb_id}: {mol}")


def generate_ligand_protein_mapping(data_path, suppl_path, output_path):
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
    
    processed_count = 0
    for pdb_id in pdb_dirs[:50]:  # 先处理前50个作为测试
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
            
            # 查找具有相似配体的其他PDB（在前50个中查找）
            for other_pdb_id in pdb_dirs[:50]:
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
    
    stats_file = os.path.join(output_path, "mapping_stats.txt")
    with open(stats_file, 'w') as f:
        f.write("配体-蛋白质映射统计信息\n")
        f.write("=" * 30 + "\n")
        f.write(f"配体数量: {len(ligand_protein_map)}\n")
        f.write(f"总的配体-蛋白质对数: {total_pairs}\n")
        f.write(f"平均每个配体对应的蛋白质数: {avg_proteins_per_ligand:.2f}\n")
        f.write(f"处理的PDB条目数: {processed_count}\n")
        
        f.write("\n映射关系详细信息:\n")
        f.write("-" * 30 + "\n")
        for ligand_id, protein_ids in ligand_protein_map.items():
            f.write(f"Ligand: {ligand_id}\n")
            f.write(f"Allowed proteins ({len(protein_ids)}): {', '.join(protein_ids)}\n")
            f.write("-" * 40 + "\n")
    
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
    parser.add_argument('--mode', choices=['process', 'mapping', 'convert'], default='process',
                       help='运行模式: process=处理数据, mapping=生成映射关系, convert=转换pkl文件')
    parser.add_argument('--convert_pkl', type=str, help='要转换的pkl文件路径')
    parser.add_argument('--txt_output', type=str, help='输出的txt文件路径')

    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    
    # 根据模式执行不同的功能
    if args.mode == 'mapping':
        # 生成映射关系模式
        print("运行配体-蛋白质映射生成模式...")
        success = generate_ligand_protein_mapping(args.data_path, args.data_suppl_path, args.output_path)
        if success:
            print("映射关系生成完成!")
        else:
            print("映射关系生成失败!")
        return
    
    elif args.mode == 'convert':
        if not args.convert_pkl:
            print("错误: 转换模式需要指定 --convert_pkl 参数")
            return
        
        if not os.path.exists(args.convert_pkl):
            print(f"错误: PKL文件不存在: {args.convert_pkl}")
            return
        
        # 自动生成输出文件名
        if not args.txt_output:
            base_name = os.path.splitext(os.path.basename(args.convert_pkl))[0]
            args.txt_output = f"{base_name}_content.txt"
        
        print(f"开始转换PKL文件: {args.convert_pkl}")
        success = convert_pkl_to_txt(args.convert_pkl, args.txt_output)
        
        if success:
            print(f"转换完成! 输出文件: {args.txt_output}")
        else:
            print("转换失败!")
        
        return
    
    # 数据处理模式（原有功能）
    print("运行数据处理模式...")
    
    # 测试功能 - 取消注释以运行
    # dic_aff = get_aff(info_path)
    # print(dic_aff)
    
    # mols = process_all_pdbs_in_folder(data_path)
    # print(f"mols 字典包含 {len(mols)} 个分子")
    
    print('Preparing tasks...')
    tasks = []
    for c in os.listdir(args.data_path):
        tasks.append((process, (c, args.data_path, args.data_suppl_path, args.output_path, args.cache)))
    print(f'Task num: {len(tasks)}')

    print(f'Begin...')
    # for p, task in tqdm(tasks):
    #     p(task)
    # sys.exit()
    pool = Pool()
    fail = 0
    for r in pool.map(try_prepare_pdbbind, tasks):
        if not r:
            fail += 1
    pool.close()
    pool.join()
    print(f'Success: {len(tasks) - fail}/{len(tasks)}, {(len(tasks) - fail) / len(tasks) * 100:.2f}%')

    shutil.rmtree(args.cache)
    print('='*20 + 'DONE' + '='*20)


if __name__ == '__main__':
    main()


def convert_pkl_to_txt(pkl_file_path, output_txt_path):
    """
    将pkl文件转换为txt格式
    
    Args:
        pkl_file_path (str): pkl文件的路径
        output_txt_path (str): 输出txt文件的路径
    """
    try:
        # 读取pkl文件
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 写入txt文件
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write("=== PKL文件内容转换 ===\n")
            f.write(f"源文件: {pkl_file_path}\n")
            f.write(f"数据类型: {type(data)}\n")
            f.write("=" * 50 + "\n\n")
            
            if isinstance(data, dict):
                f.write(f"字典包含 {len(data)} 个键值对:\n\n")
                for i, (key, value) in enumerate(data.items(), 1):
                    f.write(f"[{i}] 键: {key}\n")
                    if isinstance(value, list):
                        f.write(f"    值类型: 列表，长度: {len(value)}\n")
                        f.write(f"    内容: {value[:10]}{'...' if len(value) > 10 else ''}\n")
                    elif isinstance(value, dict):
                        f.write(f"    值类型: 字典，包含 {len(value)} 个键值对\n")
                    else:
                        f.write(f"    值: {value}\n")
                    f.write("-" * 30 + "\n")
            
            elif isinstance(data, list):
                f.write(f"列表包含 {len(data)} 个元素:\n\n")
                for i, item in enumerate(data[:100], 1):  # 只显示前100个
                    f.write(f"[{i}] {item}\n")
                if len(data) > 100:
                    f.write(f"... 还有 {len(data) - 100} 个元素\n")
            
            else:
                f.write(f"数据内容:\n{data}\n")
        
        print(f"成功将 {pkl_file_path} 转换为 {output_txt_path}")
        return True
        
    except Exception as e:
        print(f"转换失败: {e}")
        return False


if __name__ == '__main__':
    main()
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
            if processed_count % 100 == 0:
                print(f"已处理 {processed_count} 个PDB条目...")
                
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
    
    # 生成统计信息
    total_pairs = sum(len(proteins) for proteins in ligand_protein_map.values())
    avg_proteins_per_ligand = total_pairs / len(ligand_protein_map) if ligand_protein_map else 0
    
    stats_file = os.path.join(output_path, "mapping_stats.txt")
    with open(stats_file, 'w') as f:
        f.write("配体-蛋白质映射统计信息\n")
        f.write("=" * 30 + "\n")
        f.write(f"配体数量: {len(ligand_protein_map)}\n")
        f.write(f"总的配体-蛋白质对数: {total_pairs}\n")
        f.write(f"平均每个配体对应的蛋白质数: {avg_proteins_per_ligand:.2f}\n")
        f.write(f"处理的PDB条目数: {processed_count}\n")
        
        f.write("\n前10个映射关系示例:\n")
        f.write("-" * 30 + "\n")
        for i, (ligand_id, protein_ids) in enumerate(list(ligand_protein_map.items())[:10]):
            f.write(f"Ligand: {ligand_id}\n")
            f.write(f"Allowed proteins ({len(protein_ids)}): {', '.join(protein_ids)}\n")
            f.write("-" * 20 + "\n")
    
    print(f"统计信息已保存到: {stats_file}")
    return True