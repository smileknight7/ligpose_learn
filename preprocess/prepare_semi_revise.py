import pickle
import os
import shutil
import sys
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
import argparse
import random

import gc

import numpy as np
from tqdm import tqdm, trange
from ray.util.multiprocessing import Pool

from utils.common import delmkdir, print_args
from utils.pdbbind_preprocess import process_semi_pocket, process_semi_ligand

def generate_combinations_from_disk(protein_dir, ligand_dir):
    protein_ids = {f.split('.')[0] for f in os.listdir(protein_dir) if f.endswith('.npz')}
    ligand_ids = {f.split('.')[0] for f in os.listdir(ligand_dir) if f.endswith('.npz')}
    valid_ids = protein_ids & ligand_ids  # 两者都存在的才处理

    combinations = [(pdb_id, pdb_id) for pdb_id in sorted(valid_ids)]
    return combinations

def process(task):
    """
    处理单个任务的函数
    """
    pdb_id, data_path, protein_output_path, ligand_output_path, complex_output_path, cache_path = task
    
    try:
        # 处理蛋白质口袋
        dic_pocket = process_semi_pocket(pdb_id, data_path, cache_path)
        if dic_pocket is None:
            return False
            
        # 处理配体
        dic_ligand = process_semi_ligand(pdb_id, data_path)
        if dic_ligand is None:
            return False
        
        # 保存蛋白质数据
        np.savez_compressed(f'{protein_output_path}/{pdb_id}.npz', **dic_pocket)
        
        # 保存配体数据
        np.savez_compressed(f'{ligand_output_path}/{pdb_id}.npz', **dic_ligand)
        
        # 保存复合物索引（1:1对应）
        complex_name = f'{pdb_id}-{pdb_id}'  # 自己对自己
        complex_info = {
            'protein_id': pdb_id,
            'ligand_id': pdb_id,
        }
        np.savez_compressed(f'{complex_output_path}/{complex_name}.npz', **complex_info)
        shutil.rmtree(os.path.join(cache_path, pdb_id), ignore_errors=True)
        return True
        
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return False

def process_batch(task):
    """
    批量处理函数 - 先处理所有数据，再生成组合
    """
    pdb_list, data_path, protein_output_path, ligand_output_path, complex_output_path, cache_path = task
    
    # 确保输出目录存在
    os.makedirs(protein_output_path, exist_ok=True)
    os.makedirs(ligand_output_path, exist_ok=True)
    os.makedirs(complex_output_path, exist_ok=True)
    
   
    failed_ids = []
    
    print(f"Processing {len(pdb_list)} complexes...")
    
    # 1. 处理所有蛋白质和配体数据
    for idx, pdb_id in enumerate(tqdm(pdb_list, desc="Processing proteins and ligands")):
        try:
            # 处理蛋白质口袋
            dic_pocket = process_semi_pocket(pdb_id, data_path, cache_path)
            if dic_pocket is None:
                failed_ids.append(pdb_id)
                continue

            # 处理配体
            dic_ligand = process_semi_ligand(pdb_id, data_path)
            if dic_ligand is None:
                failed_ids.append(pdb_id)
                continue

            # 保存数据
            np.savez_compressed(f'{protein_output_path}/{pdb_id}.npz', **dic_pocket)
            np.savez_compressed(f'{ligand_output_path}/{pdb_id}.npz', **dic_ligand)

            
            del dic_pocket
            del dic_ligand

            # 每处理10个，主动释放一次内存
            if (idx + 1) % 10 == 0:
                gc.collect()

            shutil.rmtree(os.path.join(cache_path, pdb_id), ignore_errors=True)

        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")
            failed_ids.append(pdb_id)
            continue
    successful_ids = [pdb_id for pdb_id in pdb_list if pdb_id not in failed_ids]
    print(f"Successfully processed: {len(successful_ids)} / {len(pdb_list)} complexes")
    
    # 2. 生成组合策略（只使用成功处理的数据）

    combinations = generate_combinations_from_disk(args.protein_output_path, args.ligand_output_path)
    
    for protein_id, ligand_id in combinations:
        complex_path = os.path.join(complex_output_path, f'{protein_id}-{ligand_id}.npz')
        np.savez_compressed(complex_path, protein_id=protein_id, ligand_id=ligand_id)

def try_prepare_pdbbind(task):
    """
    包装函数，用于错误处理
    """
    try:
        func, args = task
        return func(args)
    except Exception as e:
        print(f"Task failed: {e}")
        return False

if __name__ == '__main__':
    # main args
    parser = argparse.ArgumentParser()
    
    # data source
    parser.add_argument('--data_path', type=str,
                        default='/home/smileknight/learn/ligpose_data/v2020-other-PL', help='data path')
    
    # output paths
    parser.add_argument('--protein_output_path', type=str,
                        default='/home/smileknight/learn/work_file/proteins', help='protein output path')
    parser.add_argument('--ligand_output_path', type=str,
                        default='/home/smileknight/learn/work_file/ligands', help='ligand output path')
    parser.add_argument('--complex_output_path', type=str,
                        default='/home/smileknight/learn/work_file/complex', help='complex output path')
    parser.add_argument('--cache_path', type=str,
                        default='/home/smileknight/learn/work_file/cache', help='tmp path')
    
    # processing options
    parser.add_argument('--batch_mode', action='store_true', 
                        help='Use batch processing mode (recommended)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers')

    args = parser.parse_args()
    print_args(args)

    # prepare file paths
    delmkdir(args.protein_output_path)
    delmkdir(args.ligand_output_path) 
    delmkdir(args.complex_output_path)
    delmkdir(args.cache_path)

    print('Preparing tasks...')
    
    # 获取所有PDB ID
    pdb_list = [c for c in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, c))]
    print(f'Found {len(pdb_list)} PDB complexes')
    
    if args.batch_mode:
        # 批量处理模式（推荐）
        print("Using batch processing mode...")
        
        # 将PDB列表分割成多个批次
        batch_size = max(1, len(pdb_list) // args.num_workers)
        batches = [pdb_list[i:i + batch_size] for i in range(0, len(pdb_list), batch_size)]
        
        tasks = []
        for batch in batches:
            tasks.append((process_batch, (batch, args.data_path, args.protein_output_path, 
                         args.ligand_output_path, args.complex_output_path, args.cache_path)))
        
        print(f'Created {len(tasks)} batch tasks')
        
    else:
        # 单个处理模式
        print("Using individual processing mode...")
        tasks = []
        for pdb_id in pdb_list:
            tasks.append((process, (pdb_id, args.data_path, args.protein_output_path,
                         args.ligand_output_path, args.complex_output_path, args.cache_path)))
        
        print(f'Created {len(tasks)} individual tasks')

    print(f'Begin processing...')
    
    # 构建多任务进程池来处理任务列表并统计失败的任务数量
    pool = Pool(args.num_workers)
    fail = 0
    
    try:
        results = pool.map(try_prepare_pdbbind, tasks)
        for r in results:
            if not r:
                fail += 1
                
        success_count = len(tasks) - fail
        success_rate = success_count / len(tasks) * 100
        
        print(f'Success: {success_count}/{len(tasks)}, {success_rate:.2f}%')
        
    except Exception as e:
        print(f"Processing failed: {e}")
    finally:
        pool.close()
        pool.join()
    
    # 清理缓存文件
    if os.path.exists(args.cache_path):
        shutil.rmtree(args.cache_path)
    
    print('='*20 + ' DONE ' + '='*20)
