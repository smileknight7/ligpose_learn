# debug_semi_dataset.py
import os
import random
import torch
from types import SimpleNamespace

from utils.pdbbind_utils import ComplexStructDataset, collate_struct, load_semi_list

def make_args():
    # 这里的路径你按自己真实路径改
    return SimpleNamespace(
        fragment=False,
        pdbbind_path="/home/smileknight/data/workfile/ligpose/pdbbind_other",   # 有标签 npz
        original_path="/home/dtj/work_site/prepare_data/complex/v2020-PL",

        # semi 用到的 npz（你自己设的）
        l_npz_path="/home/smileknight/data/workfile/ligpose/l_p_pretrain/ligand",
        p_npz_path="/home/smileknight/data/workfile/ligpose/l_p_pretrain/protein",
        c_npz_path="./",             # 暂时没用到的话随便给一个
        f_pkl_path="/your/fragment_dir",

        # model 超参（和主脚本保持一致就行）
        n_cycle=4,
        coor_scale=10,
        aff_scale=10,
        max_len_before_sampling=800,
        max_len_after_sampling=700,
        max_len_after_sampling_for_eval=700,
        max_len_ligand=150,
        max_ligand_atom_init_distance=10,
        max_ligand_atom_pretrain_distance=30,

        # pocket 采样
        sample_pocket_flag=True,
        select_pocket_type="CA",
        select_center_type="any_atom",

        # mask / semi
        semi_rate=1.0,        # ✅ 为了测试，直接全用 semi
        mask_rate_l=0.15,
        mask_rate_p=0.15,
        noise_distance=2.0,

        # 其他
        dropout=0.1,
    )

def main():
    args = make_args()

    # 1. 读自监督 semi 列表
    semi_list = load_semi_list("/home/smileknight/data/workfile/ligpose/l_p_pretrain/semi_list.txt")
    print(f"Loaded {len(semi_list)} semi pairs, example:", semi_list[0])

    # 2. 用 pdbbind 的 npz 文件名当有标签列表（随便取一小部分）
    all_npz = [f for f in os.listdir(args.pdbbind_path) if f.endswith(".npz")]
    all_npz.sort()
    train_list = all_npz[:10]     # 先拿 10 个测试

    print(f"Train_list size: {len(train_list)}, first one: {train_list[0]}")

    # 3. 构建 dataset（只测 train，mode='train'）
    dataset = ComplexStructDataset("train", args, train_list, semi_list)
    print("Dataset length:", len(dataset))

    # 4. 随机取几个样本看看
    for i in range(3):
        idx = random.randint(0, len(dataset)-1)
        sample = dataset[idx]
        if sample is None:
            print(f"[{i}] sample is None, skipped")
            continue

        print(f"[{i}] f_name(idx) =", sample["idx"])
        print("    protein_node_feature_init:", sample["protein_node_feature_init"].shape)
        print("    ligand_node_feature_init:", sample["ligand_node_feature_init"].shape)
        print("    edge_feature_init:", sample["edge_feature_init"].shape)
        print("    coor_init:", sample["coor_init"].shape)
        print("    len_ligand:", sample["len_ligand"])
        print("    len_complex_before_sampling:", sample["len_complex_before_sampling"])
        print("    len_complex_after_sampling:", sample["len_complex_after_sampling"])
        print("    data_type:", "semi" if "-" in sample["idx"] else "pdbbind")

    # 5. 再用 DataLoader + collate_struct 测一下 batch 维度
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_struct
    )

    batch = next(iter(loader))
    print("\nBatch shapes:")
    print("  protein_node_feature_init:", batch.protein_node_feature_init.shape)
    print("  ligand_node_feature_init:", batch.ligand_node_feature_init.shape)
    print("  edge_feature_init:", batch.edge_feature_init.shape)
    print("  coor_init:", batch.coor_init.shape)
    print("  node_sampling_loc:", batch.node_sampling_loc.shape)
    print("  ligand_node_loc_after_sampling_flat:", batch.ligand_node_loc_after_sampling_flat.shape)
    print("  ligand_match:", batch.ligand_match.shape)

if __name__ == "__main__":
    main()
