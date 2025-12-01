import os 
import torch
import argparse
import numpy as np
from rdkit import Chem
from multiprocessing import Pool
from utils.pdbbind_preprocess import try_prepare_pdbbind
from utils.common import delmkdir, print_args
from diffusionpart.preprocess import process_fragment


def to_numpy(val):
    # 单个 tensor：直接转 numpy
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()

    # 列表且非空，且元素是 tensor：先 stack 再转
    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], torch.Tensor):
        return torch.stack(val).detach().cpu().numpy()  # 形状比如 (n_frag, feat_dim)

    # 其他情况交给 numpy 自己处理
    return np.array(val)





def process(task):                     # 没有默认值的参数必须在有默认值参数的前面
    
    bad_ligands = []
    pdb_id, data_path, radius, vocab_path, moltree, output_path = task
    result = process_fragment(pdb_id, data_path, radius, vocab_path, moltree)                      # 如果保存moltree对象这里要改成pkl格式数据
    if result is None:
        bad_ligands.append(pdb_id)
        print("bad ligand:", pdb_id)
    else:
        dic_data = result
        print(result,'result')
        dic_np = {k: to_numpy(v) for k, v in dic_data.items()}
        print(dic_np,'dic_np')
        np.savez_compressed(f'{output_path}/{pdb_id}.npz', **dic_np)
    return True                             # 用于task机制返回




if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/lpw/ligpose/data/work_file/test",
        help="Path to row_data"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="/data/lpw/ligpose/data/work_file/output_test",
        help="Directory to save processed data."
    )


    parser.add_argument(
        "--vocab_path",
        type=str,
        required=False,
        default="/home/lpw/ligpose_learn/data/vocab_blur_fps_updated.csv",
        help="Path to the vocabulary file."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for data processing."
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=6.0,
        help="Radius for protein pocket extraction."
    )

    parser.add_argument(
        "--moltree",
        action='store_true',
        help="Whether to use MolTree representation for ligands."
    )
    


    args = parser.parse_args()
    print_args(args)
    print('Preparing tasks...')

    tasks = []
    for pdb_id in os.listdir(args.data_path):
        task = (process, (pdb_id, args.data_path, args.radius, args.vocab_path, args.moltree, args.output_path))
        tasks.append(task)
    
    print(f'Task num: {len(tasks)}')
    print(f'Begin...')
    pool = Pool(processes=args.num_workers)
    fail = 0
    for r in pool.map(try_prepare_pdbbind, tasks):
        if not r:
            fail += 1
    print(f'Success: {len(tasks) - fail}/{len(tasks)}, {(len(tasks) - fail) / len(tasks) * 100:.2f}%')