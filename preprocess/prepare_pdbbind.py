

import pickle
import tqdm
import os
import shutil
import sys
import lmdb
from pathlib import Path
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]))
import argparse
#这一段是依照目前的目录来按照顺序找到上上级目录，并进行提取
#abspath(__file__).split('/')[:-2]))。。。'/'.join是将路径重新拼接起来
import io
import numpy as np
from tqdm import tqdm, trange
from multiprocessing import Pool                                            # # 这里的ray.util.multiprocessing.Pool是基于CPU进行的运算的
from utils.common import delmkdir, print_args
from utils.semi_preprocess import preprocess_ligand_geom, preprocess_protein_pdb, GeomUnpickler
from utils.pdbbind_preprocess import process_pdbbind, try_prepare_pdbbind
import logging



                                                                            # 定义task作为process的传入变量
                                                                            # 将task任务进行拆包并提取内部的元素
def process(task):
    
    if args.mode =='pdbbind':
        pdb_id, data_path, suppl_path, output_path, cache_path = task
        dic_data = process_pdbbind(pdb_id, data_path, suppl_path, cache_path)                             
        np.savez_compressed(file=f'{output_path}/{pdb_id}.npz', **dic_data)              
        return True

    elif args.mode == 'semi_ligand':
        i, key, lmdb_path, save_dir = task
        env = lmdb.open(
        lmdb_path,
        readonly=True,
        lock=False,
        subdir=False,
        )
        with env.begin() as txn:
            value = txn.get(key)
            if value is None:
                return False
            item = GeomUnpickler(io.BytesIO(value)).load()  # 这里使用lmdb取item的时候也要注意，一个一个的取
        dic_data = preprocess_ligand_geom(i, item, save_dir)   # 将保存文件的部分写在函数内部可以节省内存
        if dic_data is None:
            bad_file = os.path.join(save_dir, "bad_ligand.txt")
            with open(bad_file, "a") as f:
                f.write(f"{i}\t{key.decode() if isinstance(key, bytes) else key}\n")
            return False
        return True
    
    elif args.mode == 'semi_protein':
        gz_path, save_dir = task
        dic_data = preprocess_protein_pdb(gz_path, save_dir)
        return True
    
    else:
        raise NotImplementedError


if __name__ == '__main__':
    # main args
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data preprocessing')


    # data source
    parser.add_argument('--mode', type=str,
                        default= 'pdbbind', help='pdbbind / semi_ligand / semi_protein')

    parser.add_argument('--data_path', type=str,
                        default= '/data/lpw/ligpose/data/refined-set', help='data path')
    parser.add_argument('--data_suppl_path', type=str,
                        default= '/data/lpw/ligpose/data/INDEX_refined_set.txt', help='suppl path')
    # output
    parser.add_argument('--output_path', type=str,
                        default= '/data/lpw/ligpose/data/work_file/tmp', help='prepared path')
    parser.add_argument('--cache', type=str,
                        default= '/data/lpw/ligpose/data/cache', help='tmp path')




    # semi data source
    parser.add_argument('--semi_ligand', type=str,
                        default= '/home/smileknight/data/workfile/ligpose/geom_drug/processed.lmdb', help='lmdb_path')
    parser.add_argument('--semi_protein', type=str,
                        default= '/home/smileknight/data/pdb_data/pdb_divided', help='semi data path')
    # semi output
    parser.add_argument('--l_npz_path', type=str,
                        default= '/home/smileknight/data/workfile/ligpose/l_p_pretrain/ligand', help='semi prepared path')
    parser.add_argument('--p_npz_path', type=str,
                        default= '/home/smileknight/data/workfile/ligpose/l_p_pretrain/protein', help='semi prepared path')





    args = parser.parse_args()
                                                                    # 这行代码是在解析命令行参数，上面的add_argument(）模块是传递命令行参数进去
    print_args(args)

    # prepare file path



                                                                    # 构建tasks列表，将使用的函数，传入的参数交给多进程池

    print('Preparing tasks...')
    tasks = []
    if args.mode == 'pdbbind':
        for c in os.listdir(args.data_path):                        # 这个for循环要写在外面
            tasks.append((process, (c, args.data_path, args.data_suppl_path, args.output_path, args.cache)))
    
    
    elif args.mode == 'semi_ligand':
        env = lmdb.open(args.semi_ligand, 
            readonly=True,
                lock=False,
                subdir=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            n_entries = txn.stat()['entries']
            for i, (key, _) in tqdm(enumerate(cursor), total=n_entries):
                tasks.append((process, (i, key, args.semi_ligand, args.l_npz_path)))

    
    elif args.mode == 'semi_protein':
        protein_path = Path(args.semi_protein)
        for gz_path in  protein_path.glob("*/*.ent.gz"):
            tasks.append((process, (gz_path, args.p_npz_path)))




    print(f'Task num: {len(tasks)}')

    print(f'Begin...')
    pool = Pool(processes=args.num_workers)
    fail = 0
    for r in tqdm(pool.map(try_prepare_pdbbind, tasks), total=len(tasks)):
        if not r:
            fail += 1
    print(f'Success_rate: {len(tasks) - fail}/{len(tasks)}, {(len(tasks) - fail) / len(tasks) * 100:.2f}%')
    print(f'Success_num: {len(tasks) - fail}, Fail num: {fail}')
    print(f'Fail_num: {fail}')
    
    shutil.rmtree(args.cache)
    print('='*20 + 'DONE' + '='*20)
    

















