

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
from multiprocessing import Pool

                                                                            # 这里的ray.util.multiprocessing.Pool是基于CPU进行的运算

from utils.common import delmkdir, print_args
from utils.pdbbind_preprocess import process_pdbbind, try_prepare_pdbbind

                                                                            # 定义task作为process的传入变量
                                                                            # 将task任务进行拆包并提取内部的元素
def process(task):
    
    pdb_id, data_path, suppl_path, output_path, cache_path = task
    dic_data = process_pdbbind(pdb_id, data_path, suppl_path, cache_path)                             
    np.savez_compressed(f'{output_path}/{pdb_id}.npz', **dic_data)              
    return True


if __name__ == '__main__':
    # main args
    parser = argparse.ArgumentParser()
                                                                    # argparse是python内置的命令行参数解析模块
    # data source
    parser.add_argument('--data_path', type=str,
                        default= '/data/lpw/ligpose/data/refined-set', help='data path')
    parser.add_argument('--data_suppl_path', type=str,
                        default= '/data/lpw/ligpose/data/INDEX_refined_set.txt', help='suppl path')

    # output
    parser.add_argument('--output_path', type=str,
                        default= '/data/lpw/ligpose/data/work_file/tmp', help='prepared path')
    parser.add_argument('--cache', type=str,
                        default= '/data/lpw/ligpose/data/cache', help='tmp path')

    args = parser.parse_args()
                                                                    # 这行代码是在解析命令行参数，上面的add_argument(）模块是传递命令行参数进去
    print_args(args)

    # prepare file path
    delmkdir(args.output_path)
    delmkdir(args.cache)


                                                                    # 构建tasks列表，将使用的函数，传入的参数交给多进程池

    print('Preparing tasks...')
    tasks = []
    for c in os.listdir(args.data_path):
        tasks.append((process, (c, args.data_path, args.data_suppl_path, args.output_path, args.cache)))
    print(f'Task num: {len(tasks)}')

    print(f'Begin...')
    pool = Pool()
    fail = 0
    for r in pool.map(try_prepare_pdbbind, tasks):
        if not r:
            fail += 1
    print(f'Success: {len(tasks) - fail}/{len(tasks)}, {(len(tasks) - fail) / len(tasks) * 100:.2f}%')

    shutil.rmtree(args.cache)
    print('='*20 + 'DONE' + '='*20)
    

















