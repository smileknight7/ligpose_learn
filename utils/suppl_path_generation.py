#问题1原本数据集中是否考虑单位问题




import os
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem




info_path = r'D:\Neral network\pythonProject\LigPose_demo\data\INDEX_refined_set.txt'

def get_aff(info_path):
    lines = open(info_path, 'r').readlines()
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
dic_aff = get_aff(info_path)
print(dic_aff)






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

data_path = r'D:\Neral network\pythonProject\LigPose_demo\data\refined_set'
mols = process_all_pdbs_in_folder(data_path)
print(f"mols 字典包含 {len(mols)} 个分子")
#打印结果
print(f"mols 字典包含 {len(mols)} 个分子")
for pdb_id, mol in mols.items():
    print(f"{pdb_id}: {mol}")




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
    print(f'Success: {len(tasks) - fail}/{len(tasks)}, {(len(tasks) - fail) / len(tasks) * 100:.2f}%')

    shutil.rmtree(args.cache)
    print('='*20 + 'DONE' + '='*20)
