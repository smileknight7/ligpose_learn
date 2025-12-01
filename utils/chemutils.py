import rdkit
import rdkit.Chem as Chem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem.Descriptors import MolLogP, qed
from torch_geometric.data import Data, Batch
from random import sample
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
import numpy as np
from math import sqrt
import torch
from copy import deepcopy
MST_MAX_WEIGHT = 100
MAX_NCAND = 2000


def vina_score(mol):
    ligand_rdmol = Chem.AddHs(mol, addCoords=True)
    if use_uff:
        UFFOptimizeMolecule(ligand_rdmol)

def lipinski(mol):
    if qed(mol)<=5 and Chem.Lipinski.NumHDonors(mol)<=5 and Chem.Lipinski.NumHAcceptors(mol)<=10 and Chem.Descriptors.ExactMolWt(mol)<=500 and Chem.Lipinski.NumRotatableBonds(mol)<=5:
        return True
    else:
        return False


def list_filter(a,b):
    filter = []
    for i in a:
        if i in b:
            filter.append(i)
    return filter


def rand_rotate(dir, ref, pos, alpha=None):
    #dir = dir/torch.norm(dir)
    if alpha is None:
        alpha = torch.randn(1)
    n_pos = pos.shape[0]
    sin, cos = torch.sin(alpha), torch.cos(alpha)
    K = 1 - cos
    M = torch.dot(dir, ref)
    nx, ny, nz = dir[0], dir[1], dir[2]
    x0, y0, z0 = ref[0], ref[1], ref[2]
    T = torch.tensor([nx ** 2 * K + cos, nx * ny * K - nz * sin, nx * nz * K + ny * sin,
         (x0 - nx * M) * K + (nz * y0 - ny * z0) * sin,
         nx * ny * K + nz * sin, ny ** 2 * K + cos, ny * nz * K - nx * sin,
         (y0 - ny * M) * K + (nx * z0 - nz * x0) * sin,
         nx * nz * K - ny * sin, ny * nz * K + nx * sin, nz ** 2 * K + cos,
         (z0 - nz * M) * K + (ny * x0 - nx * y0) * sin,
         0, 0, 0, 1]).reshape(4, 4)
    pos = torch.cat([pos.t(), torch.ones(n_pos).unsqueeze(0)], dim=0)
    rotated_pos = torch.mm(T, pos)[:3]
    return rotated_pos.t()


def kabsch(A, B):
    # Input:
    #     Nominal  A Nx3 matrix of points
    #     Measured B Nx3 matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix (B to A)
    # t = 3x1 translation vector (B to A)
    assert len(A) == len(B)
    N = A.shape[0] # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    # center the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    H = np.transpose(BB) * AA
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T * U.T
    t = -R * centroid_B.T + centroid_A.T
    return R, t


def kabsch_torch(A, B, C):
    A=A.double()
    B=B.double()
    C=C.double()
    a_mean = A.mean(dim=0, keepdims=True)
    b_mean = B.mean(dim=0, keepdims=True)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = torch.matmul(A_c.transpose(0,1), B_c)  # [B, 3, 3]
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = torch.matmul(V, U.transpose(0,1))  # [B, 3, 3]
    # Translation vector
    t = b_mean - torch.matmul(R, a_mean.transpose(0,1)).transpose(0,1)
    C_aligned = torch.matmul(R, C.transpose(0,1)).transpose(0,1) + t
    return C_aligned, R, t


def eig_coord_from_dist(D):
    M = (D[:1, :] + D[:, :1] - D) / 2
    L, V = torch.linalg.eigh(M)
    L = torch.diag_embed(torch.sort(L, descending=True)[0])
    X = torch.matmul(V, L.clamp(min=0).sqrt())
    return X[:, :3].detach()


def self_square_dist(X):
    dX = X.unsqueeze(0) - X.unsqueeze(1)  # [1, N, 3] - [N, 1, 3]
    D = torch.sum(dX**2, dim=-1)
    return D


def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def decode_stereo(smiles2D):
    mol = Chem.MolFromSmiles(smiles2D)
    dec_isomers = list(EnumerateStereoisomers(mol))

    dec_isomers = [Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True)) for mol in dec_isomers]
    smiles3D = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in dec_isomers]

    chiralN = [atom.GetIdx() for atom in dec_isomers[0].GetAtoms() if
               int(atom.GetChiralTag()) > 0 and atom.GetSymbol() == "N"]
    if len(chiralN) > 0:
        for mol in dec_isomers:
            for idx in chiralN:
                mol.GetAtomWithIdx(idx).SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            smiles3D.append(Chem.MolToSmiles(mol, isomericSmiles=True))

    return smiles3D


def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol


def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol


def get_submol(mol, idxs, mark=[]):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    map = {}
    for atom in mol.GetAtoms():
        if atom.GetIdx() in idxs:
            new_atom = copy_atom(atom)
            if atom.GetIdx() in mark:
                new_atom.SetAtomMapNum(1)
            else:
                new_atom.SetAtomMapNum(0)
            map[atom.GetIdx()] = new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if a1 in idxs and a2 in idxs:
            bt = bond.GetBondType()
            new_mol.AddBond(map[a1], map[a2], bt)
    return new_mol.GetMol()


def get_clique_mol(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol)  # We assume this is not None
    for atom in new_mol.GetAtoms():
        print(atom.GetIdx(), atom.GetSymbol())
    
    return new_mol





# 这个是JAV-VAE树的简化版本 how to get AtomIdx of every clusters?
def tree_decomp(mol, reference_vocab=None):
 
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []                        # 这个取出来的是非环键    注意这里返回的是无向边
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            cliques.append([a1,a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]          # 取出mol中最小环的原子索引
    cliques.extend(ssr)

    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)                        # 返回一个原子对应所属环的列表 [[0,1], [1], [2]]这种形式，第一个原子属于1和2号环
    
    #Merge Rings with intersection > 2 atoms                这里涉及到了环合并，邻接映射，对称剪枝的操作
    for i in range(len(cliques)):
        if len(cliques[i]) <= 2: continue                   # 跳过非环键（之间构建的小列表）
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2: continue # 检查原子从属的除其之外其他环的大小，小于两个原子的话跳过
                inter = set(cliques[i]) & set(cliques[j])   # i >= j 这个是防止双向检查（因为这个边是无向边，所以边索引会被保存两次）
                if len(inter) > 2:                          # 上面的这个操作其实算是一种剪枝
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))      # 如果共享原子>2就将环进行合并
                    cliques[j] = []                         # 清空
    
    cliques = [c for c in cliques if len(c) > 0]            # 这里是清空 变成空集的环，随后重新建立映射列表  nei_list 还是表示原子对应的环归属
    nei_list = [[] for i in range(n_atoms)]                 # 为每个原子构建一个空列表
    for i in range(len(cliques)):                           # 外层 --->一个clique一个clique的做   这种嵌套循环就是从外层理解整体逻辑，从内层理解局部动作（外层循环一次，内层循环一遍）
        for atom in cliques[i]:                             #
            nei_list[atom].append(i)                        # 这里并不是依次添加而是按照atom索引添加cliques索引
    
    #Build edges and add singleton cliques       逻辑其实不是很难就是选定一个原子，知道到其所有的聚类团，依次比较聚类团中是否有重复的原子
    edges = defaultdict(int)
    for atom in range(n_atoms):
        if len(nei_list[atom]) <= 1:               # 如果当前原子只属于一个团簇就跳过
            continue
        cnei = nei_list[atom]                                   # 当前原子的团簇索引
        bonds = [c for c in cnei if len(cliques[c]) == 2]       # 构建bond索引表    某个原子的关联bond --->团簇元素数量=2
        rings = [c for c in cnei if len(cliques[c]) > 4]        # 构建ring索引表
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2): # 情况1>两个键 或 两个键加还有环        # In general, if len(cnei) >= 3, a singleton should be added, but 1 bond + 2 ring is currently not dealt with.
            cliques.append([atom])                                                                      # 对于复杂节点，避免出现三环结构 ---->直接取出这个原子索引（当作原子信息储存）
            c2 = len(cliques) - 1                                                                       # 将该原子作为clique添加cliques表中
            for c1 in cnei:
                edges[(c1,c2)] = 1                                                                      # 这里是构建了与这个原子相关的边 value给1
        elif len(rings) > 2: #Multiple (n>2) complex rings      # 情况2 原子共享三个环及以上
            cliques.append([atom])
            c2 = len(cliques) - 1                                                                       # 将该原子作为clique添加cliques表
            for c1 in cnei:
                edges[(c1,c2)] = MST_MAX_WEIGHT - 1
        else:                                                   # 情况3 不复杂的原子归属情况
            for i in range(len(cnei)):                                                                  # 这里的i和j相当于是重新建立关于cnei的局部索引
                for j in range(i + 1, len(cnei)):
                    c1,c2 = cnei[i],cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1,c2)] < len(inter):             # 使用defaultdict构建的字典默认值是0
                        edges[(c1,c2)] = len(inter) # cnei[i] < cnei[j] by construction   按照连接数目定义edge权重

    conformer = mol.GetConformer()
    positions = conformer.GetPositions()
    clique_centers = []

    for cluster in cliques:
        center = [0.0, 0.0, 0.0]                # Initialize center as a float array
        for atom in cluster:
            center += positions[atom]           # Convert positions[atom] to float
        center /= len(cluster)                  
        clique_centers.append(center)           # cluster中心坐标
    
    
    
    edges = [u + (MST_MAX_WEIGHT-v,) for u,v in edges.items()]   # MST_MAX_WEIGHT-v  这里这个是为了权重翻转
    if len(edges) == 0:                                          # 打包 index和权重
        return cliques, edges, clique_centers

    #Compute Maximum Spanning Tree                               # MST 最小生成树的计算 ---> 权重越小越重要 （但是这里的设计目标是想要权重大的更重要，所以进行了取反操作）
    row,col,data = zip(*edges)
    n_clique = len(cliques)
    clique_graph = csr_matrix( (data,(row,col)), shape=(n_clique,n_clique) )        # 这个是构建一种稀疏矩阵      在普通的GNN中使用scatter_add/index_select来进行消息聚合  构建节点bond的权重
    junc_tree = minimum_spanning_tree(clique_graph)                                 # 这里是真正的在构建树，先对节点bond进行排序   ( A-B:1 ), ( B-C:2 ), ( C-D:3 ), ( A-C:4 ), ( B-D:5 )    排除掉( A-C:4 ), ( B-D:5 )因为添加这两个节点会使得树木出现环形
    row,col = junc_tree.nonzero()                                                   # 返回非0元素所在索引
    edges = [(row[i],col[i]) for i in range(len(row))]                              # 这里相当于是构建了有向边
    return cliques, edges, clique_centers
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # edges = defaultdict(int)
    # n_atoms = mol.GetNumAtoms()
    # clusters = []
    # atom_indices = []
    # for bond in mol.GetBonds():
    #     a1 = bond.GetBeginAtom().GetIdx()
    #     a2 = bond.GetEndAtom().GetIdx()
    #     if not bond.IsInRing():
    #         clusters.append({a1, a2})                   # cluster定义：成键原子对，成环原子(有共用原子的环会重建成一个环)
    #         atom_indices.append([a1, a2])               # nei_list定义：每个原子构成的团簇（元素数量是num_atoms）
    # # extract rotatable bonds

    # ssr = [set(x) for x in Chem.GetSymmSSSR(mol)]
    # # remove too large circles
    # ssr = [x for x in ssr if len(x) <= 8]

    # # Merge Rings with intersection >= 2 atoms
    # # check the reference_vocab if it is not None
    # for i in range(len(ssr)-1):
    #     if len(ssr[i]) <= 2:
    #         continue
    #     for j in range(i+1, len(ssr)):          # 这行可以避免重复计算
    #         if len(ssr[j]) <= 2:
    #             continue
    #         inter = ssr[i] & ssr[j]
    #         if reference_vocab is not None:         
    #             if len(inter) >= 2:
    #                 merge = ssr[i] | ssr[j]
    #                 smile_merge = Chem.MolFragmentToSmiles(mol, merge, canonical=True, kekuleSmiles=True)
    #                 if reference_vocab[smile_merge] <= 100 and len(inter) == 2: # 依照参考表这个片段的频次看是否合并
    #                     continue
    #                 ssr[i] = merge
    #                 ssr[j] = set()
    #         else:                              
    #             if len(inter) > 2:
    #                 merge = ssr[i] | ssr[j]
    #                 ssr[i] = merge            # 应该是使用这个 
    #                 ssr[j] = set()
    # print(ssr)
    # ssr = [c for c in ssr if len(c) > 0]      # ssr = [{0, 1, 2, 3, 4, 5, 6, 7}, set(), {10, 11, 12, 13, 14},]这种样子
    # clusters.extend(ssr)                      # 去掉空集合，将大环加到原本的原子对中
    # print(clusters)
    # atom_indices.extend([list(c) for c in ssr])

    # nei_list = [[] for _ in range(n_atoms)]   # 为原子建立clusters归属
    # for i in range(len(clusters)):
    #     for atom in clusters[i]:
    #         nei_list[atom].append(i)

    # # Calculate center point of every cluster
    # conformer = mol.GetConformer()
    # positions = conformer.GetPositions()

    # cluster_centers = []

    # for cluster in clusters:
    #     center = [0.0, 0.0, 0.0]                # Initialize center as a float array
    #     for atom in cluster:
    #         center += positions[atom]           # Convert positions[atom] to float
    #     center /= len(cluster)                  
    #     cluster_centers.append(center)          # cluster中心坐标

    # # Build edges
    # for atom in range(n_atoms):
    #     if len(nei_list[atom]) <= 1:
    #         continue
    #     cnei = nei_list[atom]                                           # cnei：当前原子归属的团簇索引表
    #     bonds = [c for c in cnei if len(clusters[c] == 2)]
    #     rings = [c for c in cnei if len(clusters[c]) > 4]
    #     if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2):
    #         clusters.append([atom])                                    # 将该原子单独作为一个团簇
    #     elif len(rings) >= 2:
    #         clusters.append([atom])                                    # 将该原子单独作为一个团簇
        
        

# 下面这种构环方式可能有问题，vocab中总是查不到目标node表
        # for i in range(len(cnei)):
        #     for j in range(i + 1, len(cnei)):
        #         c1, c2 = cnei[i], cnei[j]
        #         inter = set(clusters[c1]) & set(clusters[c2])
        #         if edges[(c1, c2)] < len(inter):
        #             edges[(c1, c2)] = len(inter)  # cnei[i] < cnei[j] by construction

    # edges = [u + (MST_MAX_WEIGHT - v,) for u, v in edges.items()]       # u和v分别是索引和权重
    # if len(edges) == 0:
    #     return clusters, edges, cluster_centers, atom_indices

    # # Compute Maximum Spanning Tree
    # row, col, data = zip(*edges)
    # n_clique = len(clusters)
    # clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
    # junc_tree = minimum_spanning_tree(clique_graph)                     # 构建最小生成树状的时候会消歧（一个原子只进行一个团簇归属）
    # row, col = junc_tree.nonzero()
    # edges = [(row[i], col[i]) for i in range(len(row))]



    # return clusters, edges, cluster_centers, atom_indices






def atom_equal(a1, a2):
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()


# Bond type not considered because all aromatic (so SINGLE matches DOUBLE)
def ring_bond_equal(bond1, bond2, reverse=False):
    b1 = (bond1.GetBeginAtom(), bond1.GetEndAtom())
    if reverse:
        b2 = (bond2.GetEndAtom(), bond2.GetBeginAtom())
    else:
        b2 = (bond2.GetBeginAtom(), bond2.GetEndAtom())
    return atom_equal(b1[0], b2[0]) and atom_equal(b1[1], b2[1]) and bond1.GetBondType() == bond2.GetBondType()



def attach(ctr_mol, nei_mol, amap):
    ctr_mol = Chem.RWMol(ctr_mol)
    for atom in nei_mol.GetAtoms():
        if atom.GetIdx() not in amap:
            new_atom = copy_atom(atom)
            new_atom.SetAtomMapNum(2)
            amap[atom.GetIdx()] = ctr_mol.AddAtom(new_atom)

    for bond in nei_mol.GetBonds():
        a1 = amap[bond.GetBeginAtom().GetIdx()]
        a2 = amap[bond.GetEndAtom().GetIdx()]
        if ctr_mol.GetBondBetweenAtoms(a1, a2) is None:
            ctr_mol.AddBond(a1, a2, bond.GetBondType())

    return ctr_mol.GetMol(), amap


def attach_mols(ctr_mol, neighbors, prev_nodes, nei_amap):
    prev_nids = [node.nid for node in prev_nodes]
    for nei_node in prev_nodes + neighbors:
        nei_id, nei_mol = nei_node.nid, nei_node.mol
        amap = nei_amap[nei_id]
        for atom in nei_mol.GetAtoms():
            if atom.GetIdx() not in amap:
                new_atom = copy_atom(atom)
                amap[atom.GetIdx()] = ctr_mol.AddAtom(new_atom)

        if nei_mol.GetNumBonds() == 0:
            nei_atom = nei_mol.GetAtomWithIdx(0)
            ctr_atom = ctr_mol.GetAtomWithIdx(amap[0])
            ctr_atom.SetAtomMapNum(nei_atom.GetAtomMapNum())
        else:
            for bond in nei_mol.GetBonds():
                a1 = amap[bond.GetBeginAtom().GetIdx()]
                a2 = amap[bond.GetEndAtom().GetIdx()]
                if ctr_mol.GetBondBetweenAtoms(a1, a2) is None:
                    ctr_mol.AddBond(a1, a2, bond.GetBondType())
                elif nei_id in prev_nids:  # father node overrides
                    ctr_mol.RemoveBond(a1, a2)
                    ctr_mol.AddBond(a1, a2, bond.GetBondType())
    return ctr_mol


def local_attach(ctr_mol, neighbors, prev_nodes, amap_list):
    ctr_mol = copy_edit_mol(ctr_mol)
    nei_amap = {nei.nid: {} for nei in prev_nodes + neighbors}

    for nei_id, ctr_atom, nei_atom in amap_list:
        nei_amap[nei_id][nei_atom] = ctr_atom

    ctr_mol = attach_mols(ctr_mol, neighbors, prev_nodes, nei_amap)
    return ctr_mol.GetMol()


# This version records idx mapping between ctr_mol and nei_mol
def enum_attach(ctr_mol, nei_mol):
    try:
        Chem.Kekulize(ctr_mol)
        Chem.Kekulize(nei_mol)
    except:
        return []
    att_confs = []
    valence_ctr = {i: 0 for i in range(ctr_mol.GetNumAtoms())}
    valence_nei = {i: 0 for i in range(nei_mol.GetNumAtoms())}
    ctr_bonds = [bond for bond in ctr_mol.GetBonds() if bond.GetBeginAtom().GetAtomMapNum() == 1 and bond.GetEndAtom().GetAtomMapNum() == 1]
    ctr_atoms = [atom for atom in ctr_mol.GetAtoms() if atom.GetAtomMapNum() == 1]
    if nei_mol.GetNumBonds() == 1:  # neighbor is a bond
        bond = nei_mol.GetBondWithIdx(0)
        #bond_val = int(bond.GetBondType())
        bond_val = int(bond.GetBondTypeAsDouble())
        b1, b2 = bond.GetBeginAtom(), bond.GetEndAtom()

        for atom in ctr_atoms:
            # Optimize if atom is carbon (other atoms may change valence)
            if atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() < bond_val:
                continue
            if atom_equal(atom, b1):
                new_amap = {b1.GetIdx(): atom.GetIdx()}
                att_confs.append(new_amap)
            elif atom_equal(atom, b2):
                new_amap = {b2.GetIdx(): atom.GetIdx()}
                att_confs.append(new_amap)
    else:
        # intersection is an atom
        for a1 in ctr_atoms:
            for a2 in nei_mol.GetAtoms():
                if atom_equal(a1, a2):
                    # Optimize if atom is carbon (other atoms may change valence)
                    if a1.GetAtomicNum() == 6 and a1.GetTotalNumHs() + a2.GetTotalNumHs() < 4:
                        continue
                    amap = {a2.GetIdx(): a1.GetIdx()}
                    att_confs.append(amap)

        # intersection is an bond
        if ctr_mol.GetNumBonds() > 1:
            for b1 in ctr_bonds:
                for b2 in nei_mol.GetBonds():
                    if ring_bond_equal(b1, b2):
                        amap = {b2.GetBeginAtom().GetIdx(): b1.GetBeginAtom().GetIdx(),
                                b2.GetEndAtom().GetIdx(): b1.GetEndAtom().GetIdx()}
                        att_confs.append(amap)

                    if ring_bond_equal(b1, b2, reverse=True):
                        amap = {b2.GetEndAtom().GetIdx(): b1.GetBeginAtom().GetIdx(),
                                b2.GetBeginAtom().GetIdx(): b1.GetEndAtom().GetIdx()}
                        att_confs.append(amap)
    return att_confs


def enumerate_assemble(mol, idxs, current, next):
    ctr_mol = get_submol(mol, idxs, mark=current.clique)
    ground_truth = get_submol(mol, list(set(idxs) | set(next.clique)))
    # submol can also obtained with get_clique_mol, future exploration
    ground_truth_smiles = get_smiles(ground_truth)
    cand_smiles = []
    cand_mols = []
    cand_amap = enum_attach(ctr_mol, next.mol)
    for amap in cand_amap:
        try:
            cand_mol, _ = attach(ctr_mol, next.mol, amap)
            cand_mol = sanitize(cand_mol)
        except:
            continue
        if cand_mol is None:
            continue
        smiles = get_smiles(cand_mol)
        if smiles in cand_smiles or smiles == ground_truth_smiles:
            continue
        cand_smiles.append(smiles)
        cand_mols.append(cand_mol)
    if len(cand_mols) >= 1:
        cand_mols = sample(cand_mols, 1)
        cand_mols.append(ground_truth)
        labels = torch.tensor([0, 1])
    else:
        cand_mols = [ground_truth]
        labels = torch.tensor([1])

    return labels, cand_mols


# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


# For inference
def assemble(mol_list, next_motif_smiles):
    attach_fail = torch.zeros(len(mol_list)).bool()
    cand_mols, cand_batch, new_atoms, cand_smiles, one_atom_attach, intersection = [], [], [], [], [], []
    for i in range(len(mol_list)):
        next = Chem.MolFromSmiles(next_motif_smiles[i])
        cand_amap = enum_attach(mol_list[i], next)
        if len(cand_amap) == 0:
            attach_fail[i] = True
            cand_mols.append(mol_list[i])
            cand_batch.append(i)
            one_atom_attach.append(-1)
            intersection.append([])
            new_atoms.append([])
        else:
            valid_cand = 0
            for amap in cand_amap:
                amap_len = len(amap)
                iter_atoms = [v for v in amap.values()]
                ctr_mol = deepcopy(mol_list[i])
                cand_mol, amap1 = attach(ctr_mol, next, amap)
                if sanitize(deepcopy(cand_mol)) is None:
                    continue
                smiles = get_smiles(cand_mol)
                cand_smiles.append(smiles)
                cand_mols.append(cand_mol)
                cand_batch.append(i)
                new_atoms.append([v for v in amap1.values()])
                one_atom_attach.append(amap_len)
                intersection.append(iter_atoms)
                valid_cand+=1
            if valid_cand==0:
                attach_fail[i] = True
                cand_mols.append(mol_list[i])
                cand_batch.append(i)
                one_atom_attach.append(-1)
                intersection.append([])
                new_atoms.append([])
    cand_batch = torch.tensor(cand_batch)
    one_atom_attach = torch.tensor(one_atom_attach) == 1
    return cand_mols, cand_batch, new_atoms, one_atom_attach, intersection, attach_fail


if __name__ == "__main__":
    import sys
    from mol_tree import MolTree

    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    smiles = ["O=C1[C@@H]2C=C[C@@H](C=CC2)C1(c1ccccc1)c1ccccc1", "O=C([O-])CC[C@@]12CCCC[C@]1(O)OC(=O)CC2",
              "ON=C1C[C@H]2CC3(C[C@@H](C1)c1ccccc12)OCCO3",
              "C[C@H]1CC(=O)[C@H]2[C@@]3(O)C(=O)c4cccc(O)c4[C@@H]4O[C@@]43[C@@H](O)C[C@]2(O)C1",
              'Cc1cc(NC(=O)CSc2nnc3c4ccccc4n(C)c3n2)ccc1Br', 'CC(C)(C)c1ccc(C(=O)N[C@H]2CCN3CCCc4cccc2c43)cc1',
              "O=c1c2ccc3c(=O)n(-c4nccs4)c(=O)c4ccc(c(=O)n1-c1nccs1)c2c34", "O=C(N1CCc2c(F)ccc(F)c2C1)C1(O)Cc2ccccc2C1"]
    mol_tree = MolTree("C")
    assert len(mol_tree.nodes) > 0


    def count():
        cnt, n = 0, 0
        for s in sys.stdin:
            s = s.split()[0]
            tree = MolTree(s)
            tree.recover()
            tree.assemble()
            for node in tree.nodes:
                cnt += len(node.cands)
            n += len(tree.nodes)
            # print cnt * 1.0 / n
    count()
