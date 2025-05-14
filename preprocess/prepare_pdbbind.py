import os
import shutil
import sys
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
import argparse

import numpy as np
from tqdm import tqdm, trange
from ray.util.multiprocessing import Pool
#这里的ray.util.multiprocessing.Pool是基于CPU进行的运算

from utils.common import delmkdir, print_args
from utils.pdbbind_preprocess import process_pdbbind, try_prepare_pdbbind


def process(task):
    pdb_id, data_path, suppl_path, output_path, cache_path = task
    dic_data = process_pdbbind(pdb_id, data_path, suppl_path, cache_path)
    # pickle.dump(dic_data, open(f'{output_path}/{pdb_id}.pkl', 'wb'))
    np.savez_compressed(f'{output_path}/{pdb_id}.npz', **dic_data)
    return True


if __name__ == '__main__':
    # main args
    parser = argparse.ArgumentParser()
#argparse是python内置的命令行参数解析模块
    # data source
    parser.add_argument('--data_path', type=str,
                        default= '/home/smileknight/learn/ligpose_data/general_set_10', help='data path')
    parser.add_argument('--data_suppl_path', type=str,
                        default= '/home/smileknight/learn/ligpose_data/INDEX_general_PL.txt', help='suppl path')

    # output
    parser.add_argument('--output_path', type=str,
                        default= '/home/smileknight/learn/work_file/tmp', help='prepared path')
    parser.add_argument('--cache', type=str,
                        default= '/home/smileknight/learn/work_file/cache', help='tmp path')

    args = parser.parse_args()
    #这行代码是在解析命令行参数，上面的add_argument(）模块是传递命令行参数进去
    print_args(args)

    # prepare file path
    delmkdir(args.output_path)
    delmkdir(args.cache)


#利用task列表将pbd数据依次传入

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
    pool = Pool()
    fail = 0
    for r in pool.map(try_prepare_pdbbind, tasks):
        if not r:
            fail += 1
    print(f'Success: {len(tasks) - fail}/{len(tasks)}, {(len(tasks) - fail) / len(tasks) * 100:.2f}%')
    #扫除缓存文件
    shutil.rmtree(args.cache)
    print('='*20 + 'DONE' + '='*20)


















