import os
import subprocess

data_dir = 'your_pdbbind_dir'

for pdb_id in os.listdir(data_dir):
    mol2_path = os.path.join(data_dir, pdb_id, f'{pdb_id}_ligand.mol2')
    if not os.path.exists(mol2_path):
        continue

    smi_path = os.path.join(data_dir, pdb_id, f'{pdb_id}_ligand.smi')
    pdb_path = os.path.join(data_dir, pdb_id, f'{pdb_id}_ligand.pdb')

    # 如果文件已经存在，就跳过
    if os.path.exists(smi_path) and os.path.exists(pdb_path):
        continue

    try:
        subprocess.run(['obabel', mol2_path, '-O', smi_path])
        subprocess.run(['obabel', mol2_path, '-O', pdb_path])
    except Exception as e:
        print(f"{pdb_id} 转换失败: {e}")