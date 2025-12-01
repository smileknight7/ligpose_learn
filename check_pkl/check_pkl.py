import os
import numpy as np
import shutil

source_dir = '/data/lpw/ligpose/data/work_file/tmp'
target_dir = '/data/lpw/ligpose/data/corrupted_pkls'

os.makedirs(target_dir, exist_ok=True)

for fname in os.listdir(source_dir):
    if fname.endswith('.pkl'):
        fpath = os.path.join(source_dir, fname)
        try:
            _ = np.load(fpath, allow_pickle=True)
        except Exception as e:
            print(f"Corrupted: {fname} --> {e}")
            shutil.move(fpath, os.path.join(target_dir, fname))  # 或者 shutil.copy

#可以在这里添加一个显示进度条的功能