import os
import random
import torch

random.seed(42)

pdbbind_data_dir = '/home/smileknight/data/pdbbind_refine/refined-set'

def collect_pairs(root):

    all_pairs = []

    for dirpath, dirnames, filenames in os.walk(root):
        protein_files = [f for f in filenames if f.endswith('protein.pdb')]
        
        sdf_files = [f for f in filenames if f.endswith('.sdf')]
        mol2_files = [f for f in filenames if f.endswith('.mol2')]

        if sdf_files:
            ligand_file = sdf_files[0]          # 
        elif mol2_files:
            ligand_file = mol2_files[0]         # 
        else:
            continue 
        
        if not protein_files:
            continue

        for p_file in protein_files:                                         
                p_path = os.path.join(dirpath, p_file)
                l_path = os.path.join(dirpath, ligand_file)
                all_pairs.append((p_path, l_path))
    return all_pairs

all_pairs = collect_pairs(pdbbind_data_dir)
print(f'Total pairs collected: {len(all_pairs)}')

random.shuffle(all_pairs)
split_ratio = 0.5
split_index = int(len(all_pairs) * split_ratio)                   # int 会自动将小数部分去掉

tain_pairs = all_pairs[:split_index]
test_pairs = all_pairs[split_index:]

split_path = {
    'train': tain_pairs,
    'test': test_pairs
}

output_path = 'data/pdbbind_split_whole.pt'
torch.save(split_path, output_path)