import pickle
import os
import shutil
import sys
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
import argparse
#这一段是依照目前的目录来按照顺序找到上上级目录，并进行提取
#abspath(__file__).split('/')[:-2]))。。。'/'.join是将路径重新拼接起来

import numpy as np
from tqdm import tqdm, trange
from ray.util.multiprocessing import Pool
#这里的ray.util.multiprocessing.Pool是基于CPU进行的运算

from utils.common import delmkdir, print_args
from utils.pdbbind_preprocess import process_semi_pocket, process_semi_ligand

#定义task作为process的传入变量
#将task任务进行拆包并提取内部的元素
def process(task):
    
    pdb_id, data_path, suppl_path, output_path, cache_path = task

    

    dic_pocket = process_semi_pocket(pdb_id, data_path, suppl_path, cache_path)
    # print(f"{pdb_id}: {dic_data is not None}")检验是否能够构建dic_data字典
    pickle.dump(dic_pocket, open(f'{output_path}/{pdb_id}.pkl', 'wb'))
    np.savez_compressed(f'{output_path}/{pdb_id}.npz', **dic_pocket)

    dic_ligand = process_semi_ligand(pdb_id, data_path, suppl_path, cache_path)
    # print(f"{pdb_id}: {dic_data is not None}")检验是否能够构建dic_data字典
    pickle.dump(dic_ligand, open(f'{output_path}/{pdb_id}.pkl', 'wb'))
    np.savez_compressed(f'{output_path}/{pdb_id}.npz', **dic_ligand)


def process(task):
    pdb_list, data_path, protein_output_path, ligand_output_path, complex_output_path, cache_path = task
    
    """
    生成半监督数据的组合策略
    """
    # 1. 先处理所有蛋白质和配体数据
    protein_data = {}
    ligand_data = {}
    
    for pdb_id in tqdm(pdb_list, desc="Processing proteins and ligands"):
        try:
            # 处理蛋白质口袋
            dic_pocket = process_semi_pocket(pdb_id, data_path, cache_path)
            np.savez_compressed(f'{protein_output_path}/{pdb_id}.npz', **dic_pocket)
            protein_data[pdb_id] = dic_pocket
            
            # 处理配体
            dic_ligand = process_semi_ligand(pdb_id, data_path, cache_path)
            np.savez_compressed(f'{ligand_output_path}/{pdb_id}.npz', **dic_ligand)
            ligand_data[pdb_id] = dic_ligand
            
        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")
            continue
    
    # 2. 生成组合策略
    combinations = generate_combinations(protein_data, ligand_data)
    
    # 3. 保存复合物索引
    for protein_id, ligand_id in combinations:
        complex_name = f'{protein_id}-{ligand_id}'
        complex_info = {
            'protein_id': protein_id,
            'ligand_id': ligand_id,
        }
        np.savez_compressed(f'{complex_output_path}/{complex_name}.npz', **complex_info)
    
    return True

def generate_combinations(protein_data, ligand_data):
    """
    生成蛋白质-配体组合策略
    """
    combinations = []
    
    # 策略1：每个蛋白质与多个随机配体组合
    for protein_id in protein_data.keys():
        # 自己和自己的组合（保留原始对应）
        combinations.append((protein_id, protein_id))
        

        #这个方法因该不行啊，这个可能会产生很多负样本信息
        # 随机选择其他配体进行组合
        # other_ligands = [lid for lid in ligand_data.keys() if lid != protein_id]
        # if len(other_ligands) > 0:
        #     # 每个蛋白质随机选择2-5个其他配体
        #     n_combinations = min(5, len(other_ligands))
        #     selected_ligands = random.sample(other_ligands, n_combinations)
        #     for ligand_id in selected_ligands:
        #         combinations.append((protein_id, ligand_id))
    
    return combinations









    
    return True


if __name__ == '__main__':
    # main args
    parser = argparse.ArgumentParser()
#argparse是python内置的命令行参数解析模块
    # data source
    parser.add_argument('--data_path', type=str,
                        default= '/data/lpw/ligpose/data/refined-set', help='data path')


    # output
    parser.add_argument('--protein_output_path', type=str,
                        default= '/data/lpw/ligpose/data/work_file/tmp', help='prepared path')
    parser.add_argument('--ligand_output_path', type=str,
                        default= '/data/lpw/ligpose/data/work_file/tmp', help='prepared path')
    parser.add_argument('--cache', type=str,
                        default= '/data/lpw/ligpose/data/cache', help='tmp path')

    args = parser.parse_args()
    #这行代码是在解析命令行参数，上面的add_argument(）模块是传递命令行参数进去
    print_args(args)

    # prepare file path
    delmkdir(args.output_path)
    delmkdir(args.cache)


#构建tasks列表，将使用的函数，传入的参数交给多进程池

    print('Preparing tasks...')
    tasks = []
    for c in os.listdir(args.data_path):
        tasks.append((process, (c, args.data_path, args.data_suppl_path, args.output_path, args.cache)))
    print(f'Task num: {len(tasks)}')

    print(f'Begin...')
    # for p, task in tqdm(tasks):
    #     p(task)
    # sys.exit()

    #构建多任务进程池来处理任务列表并统计失败的任务数量
    #fail是用来初始化计数器，以统计失败任务 pool.map（）会将任务分配到进程池中，让多个任务同时运行
    pool = Pool()
    fail = 0
    for r in pool.map(try_prepare_pdbbind, tasks):
        if not r:
            fail += 1
    print(f'Success: {len(tasks) - fail}/{len(tasks)}, {(len(tasks) - fail) / len(tasks) * 100:.2f}%')
    #扫除缓存文件
    shutil.rmtree(args.cache)
    print('='*20 + 'DONE' + '='*20)
    #:.2f表示保留两位小数




def process(task):
    pdb_id, data_path, p_output_path, l_output_path, c_output_path, cache_path, all_pdb_ids = task

    try:
        # 处理蛋白质口袋数据
        dic_pocket = process_semi_pocket(pdb_id, data_path, cache_path)
        np.savez_compressed(f'{p_output_path}/{pdb_id}.npz', **dic_pocket)
        
        # 处理配体数据  
        dic_ligand = process_semi_ligand(pdb_id, data_path, cache_path)
        np.savez_compressed(f'{l_output_path}/{pdb_id}.npz', **dic_ligand)
        
        # 生成多个复合物组合
        combinations = []
        
        # 1. 自组合（原始对应）
        combinations.append((pdb_id, pdb_id))
        
        # 2. 随机组合（增加数据多样性）
        other_pdb_ids = [pid for pid in all_pdb_ids if pid != pdb_id]
        if len(other_pdb_ids) > 0:
            n_combinations = min(3, len(other_pdb_ids))  # 每个蛋白质最多3个随机配体
            selected_ligands = random.sample(other_pdb_ids, n_combinations)
            for ligand_id in selected_ligands:
                combinations.append((pdb_id, ligand_id))
        
        # 保存所有组合
        for protein_id, ligand_id in combinations:
            complex_name = f'{protein_id}-{ligand_id}'
            complex_info = {
                'protein_id': protein_id,
                'ligand_id': ligand_id,
            }
            np.savez_compressed(f'{c_output_path}/{complex_name}.npz', **complex_info)
        
        return True
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return False

if __name__ == '__main__':
    # ...existing argument parsing...
    
    print('Preparing tasks...')
    pdb_list = os.listdir(args.data_path)
    tasks = []
    for pdb_id in pdb_list:
        tasks.append((pdb_id, args.data_path, args.protein_output_path, 
                     args.ligand_output_path, args.complex_output_path, 
                     args.cache, pdb_list))  # 传入完整的pdb_list用于组合
    print(f'Task num: {len(tasks)}')
    
    # ...existing processing logic...

##############################基于相似性组合###############################

def process_semi_data_with_combinations(pdb_list, data_path, protein_output_path, 
                                      ligand_output_path, complex_output_path, cache_path):
    """
    生成半监督数据的组合策略
    """
    # 1. 先处理所有蛋白质和配体数据
    protein_data = {}
    ligand_data = {}
    
    for pdb_id in tqdm(pdb_list, desc="Processing proteins and ligands"):
        try:
            # 处理蛋白质口袋
            dic_pocket = process_semi_pocket(pdb_id, data_path, cache_path)
            np.savez_compressed(f'{protein_output_path}/{pdb_id}.npz', **dic_pocket)
            protein_data[pdb_id] = dic_pocket
            
            # 处理配体
            dic_ligand = process_semi_ligand(pdb_id, data_path, cache_path)
            np.savez_compressed(f'{ligand_output_path}/{pdb_id}.npz', **dic_ligand)
            ligand_data[pdb_id] = dic_ligand
            
        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")
            continue
    
    # 2. 生成组合策略
    combinations = generate_combinations(protein_data, ligand_data)
    
    # 3. 保存复合物索引
    for protein_id, ligand_id in combinations:
        complex_name = f'{protein_id}-{ligand_id}'
        complex_info = {
            'protein_id': protein_id,
            'ligand_id': ligand_id,
        }
        np.savez_compressed(f'{complex_output_path}/{complex_name}.npz', **complex_info)
    
    return combinations

def generate_combinations(protein_data, ligand_data):
    """
    生成蛋白质-配体组合策略
    """
    combinations = []
    
    # 策略1：每个蛋白质与多个随机配体组合
    for protein_id in protein_data.keys():
        # 自己和自己的组合（保留原始对应）
        combinations.append((protein_id, protein_id))
        
        # 随机选择其他配体进行组合
        other_ligands = [lid for lid in ligand_data.keys() if lid != protein_id]
        if len(other_ligands) > 0:
            # 每个蛋白质随机选择2-5个其他配体
            n_combinations = min(5, len(other_ligands))
            selected_ligands = random.sample(other_ligands, n_combinations)
            for ligand_id in selected_ligands:
                combinations.append((protein_id, ligand_id))
    
    return combinations