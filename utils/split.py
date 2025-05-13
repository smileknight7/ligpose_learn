import os
import shutil
import argparse
import numpy as np
from ray.util.multiprocessing import Pool
from utils.common import delmkdir, print_args
from utils.pdbbind_preprocess import process_pdbbind, try_prepare_pdbbind
import random

def process(task, labeled_ratio=0.2):
    pdb_id, data_path, suppl_path, output_path, cache_path = task
    dic_data = process_pdbbind(pdb_id, data_path, suppl_path, cache_path)

    # 随机划分为有标签和无标签数据
    if random.random() < labeled_ratio:
        np.savez_compressed(f'{output_path}/labeled/{pdb_id}.npz', **dic_data)
    else:
        np.savez_compressed(f'{output_path}/unlabeled/{pdb_id}.npz', **dic_data)
    return True

if __name__ == '__main__':
    main()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'D:\Neral network\pythonProject\LigPose_demo\data\refined_set', help='data path')
    parser.add_argument('--data_suppl_path', type=str, default=r'D:\Neral network\pythonProject\LigPose_demo\INDEX_refined_set.txt', help='suppl path')
    parser.add_argument('--output_path', type=str, default=r'D:\Neral network\pythonProject\LigPose_demo\data\tmp', help='prepared path')
    parser.add_argument('--cache', type=str, default=r'D:\Neral network\pythonProject\LigPose_demo\data\cache', help='tmp path')
    parser.add_argument('--labeled_ratio', type=float, default=0.2, help='ratio of labeled data')
    args = parser.parse_args()

    print_args(args)

    # 检查路径
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path {args.data_path} does not exist.")
    if not os.path.exists(args.data_suppl_path):
        raise FileNotFoundError(f"Supplementary path {args.data_suppl_path} does not exist.")

    # 创建输出文件夹
    delmkdir(args.output_path)
    os.makedirs(f'{args.output_path}/labeled', exist_ok=True)
    os.makedirs(f'{args.output_path}/unlabeled', exist_ok=True)
    delmkdir(args.cache)

    # 准备任务列表
    print('Preparing tasks...')
    tasks = [
        (process, (c, args.data_path, args.data_suppl_path, args.output_path, args.cache, args.labeled_ratio))
        for c in os.listdir(args.data_path)
    ]
    print(f'Task num: {len(tasks)}')

    # 多进程处理任务
    print('Processing tasks...')
    pool = Pool()
    fail = 0
    for r in pool.map(try_prepare_pdbbind, tasks):
        if not r:
            fail += 1
    print(f'Success: {len(tasks) - fail}/{len(tasks)}, {(len(tasks) - fail) / len(tasks) * 100:.2f}%')

    # 清理缓存
    shutil.rmtree(args.cache)
    print('=' * 20 + 'DONE' + '=' * 20)

