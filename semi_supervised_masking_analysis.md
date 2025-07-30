# PDBbind半监督数据处理与掩蔽机制详细分析

## 1. 半监督（Semi）数据的定义与识别

### 1.1 数据类型识别
在`get_complex`函数中，通过文件名格式来识别数据类型：
```python
if '-' in f_name:
    data_type = 'semi'
else:
    data_type = 'pdbbind'
```
- **半监督数据**：文件名包含连字符，格式为`{protein_idx}-{ligand_idx}`
- **正常PDBbind数据**：标准的PDB ID格式，如`1a1e.npz`

### 1.2 半监督数据加载流程
```python
if data_type == 'semi':
    complex_idx = f_name[:-4]
    protein_idx = complex_idx.split('-')[0]
    ligand_idx = complex_idx.split('-')[1]
    
    # 分别从蛋白质和配体NPZ文件中加载
    dic_pocket = np.load(f'{self.p_npz_path}/{protein_idx}.npz', allow_pickle=False)
    dic_ligand = np.load(f'{self.l_npz_path}/{ligand_idx}.npz', allow_pickle=True)
    
    # 关键区别：无真实的亲和力标签和配体坐标
    aff_true = -1        # 无亲和力标签
    aff_mask = 0         # 不计算亲和力损失
    coor_mask = 0        # 不计算坐标损失
    ligand_position_true = np.zeros((len(ligand_node_feature_init), 3))  # 零坐标
```

## 2. 掩蔽（Masking）机制详细分析

### 2.1 掩蔽率参数
```python
# 在__init__中设置
self.mask_rate_l = args.mask_rate_l  # 配体掩蔽率
self.mask_rate_p = args.mask_rate_p  # 蛋白质掩蔽率
```

### 2.2 掩蔽位置生成算法
```python
def gen_mask_index(self, feat_label_list, mask_rate=0.15):
    # 1. 确定允许掩蔽的位置（特征标签有效的位置）
    allow_mask_pos = torch.cat([
        feat_label.sum(dim=-1, keepdim=True) == 1 for feat_label in feat_label_list
    ], dim=-1).prod(dim=-1)
    
    # 2. 计算需要掩蔽的数量
    origin_shape_flat = allow_mask_pos.numel()
    n_mask = max(int(mask_rate * origin_shape_flat), 1)
    
    # 3. 随机选择掩蔽位置
    mask_index = torch.randperm(origin_shape_flat)[:n_mask]
    mask = torch.zeros(origin_shape_flat).index_fill_(-1, mask_index, 1)
    
    return mask.reshape(origin_shape).bool()
```

### 2.3 蛋白质节点特征掩蔽
```python
# 提取原子和残基标签
p_atom_label = protein_node_feature_init[:, -37:]    # 原子类型标签（37维）
p_res_label = protein_node_feature_init[:, -57:-37]   # 残基类型标签（20维）

# 处理等价原子（对称原子的标准化）
for res, a1, a2 in self.fix_p_atom_label_list:
    equ_loc = ((p_res_label * res).sum(dim=-1) * (p_atom_label * a1).sum(dim=-1)) == 1
    p_atom_label[equ_loc] = repeat(a2, 'd -> n d', n=int(equ_loc.float().sum()))

# 生成掩蔽位置和标签
p_x_mask_bool = self.gen_mask_index([p_atom_label, p_res_label], self.mask_rate_p)
p_x_mask_label_1 = p_atom_label.argmax(dim=-1)      # 原子类型标签
p_x_mask_label_2 = p_res_label.argmax(dim=-1)       # 残基类型标签
```

### 2.4 配体节点特征掩蔽
```python
# 提取配体原子标签
l_atom_label = ligand_node_feature_init[:, :10]      # 配体原子类型（10维）

# 生成掩蔽位置和标签
l_x_mask_bool = self.gen_mask_index([l_atom_label], self.mask_rate_l)
l_x_mask_label = l_atom_label.argmax(dim=-1)
```

### 2.5 边特征掩蔽
```python
# 蛋白质边掩蔽
p_edge_label = torch.from_numpy(protein_edge_feature_init)
p_edge_mask_bool = self.gen_mask_index([p_edge_label], self.mask_rate_p)
# 确保边掩蔽的对称性
p_edge_mask_bool = (
    p_edge_mask_bool.triu().float() + p_edge_mask_bool.float().triu().transpose(1, 0)
).bool()

# 配体边掩蔽
l_edge_label = torch.from_numpy(ligand_edge_feature_init)
l_edge_mask_bool = self.gen_mask_index([l_edge_label], self.mask_rate_l)
l_edge_mask_bool = (
    l_edge_mask_bool.triu().float() + l_edge_mask_bool.float().triu().transpose(1, 0)
).bool()

# 合并为复合物边掩蔽
edge_mask_bool = torch.cat([
    F.pad(p_edge_mask_bool, (0, len_ligand), 'constant', False),
    F.pad(l_edge_mask_bool, (len_protein_before_sampling, 0), 'constant', False)
], dim=0).bool()
```

## 3. 特征掩蔽的具体实现

### 3.1 半监督数据的特征掩蔽
```python
if data_type == 'semi':
    # 蛋白质节点特征掩蔽
    protein_node_feature_init = F.pad(protein_node_feature_init, (0, 1), 'constant', 0)
    protein_node_feature_init[p_x_mask_bool] = 0        # 被掩蔽位置置零
    protein_node_feature_init[p_x_mask_bool, -1] = 1     # 添加掩蔽标记

    # 配体节点特征掩蔽
    ligand_node_feature_init = F.pad(ligand_node_feature_init, (0, 1), 'constant', 0)
    ligand_node_feature_init[l_x_mask_bool] = 0          # 被掩蔽位置置零
    ligand_node_feature_init[l_x_mask_bool, -1] = 1       # 添加掩蔽标记

    # 边特征掩蔽
    edge_feature_init = rearrange(F.pad(edge_feature_init, (0, 1), 'constant', 0), 'i j d -> (i j) d')
    edge_feature_init[edge_mask_bool.reshape(-1)] = 0     # 被掩蔽位置置零
    edge_feature_init[edge_mask_bool.reshape(-1), -1] = 1  # 添加掩蔽标记
    edge_feature_init = rearrange(edge_feature_init, '(i j) d -> i j d', i=len_protein_before_sampling + len_ligand)
```

### 3.2 正常数据的特征处理
```python
else:
    # 只添加padding，不进行掩蔽
    protein_node_feature_init = F.pad(protein_node_feature_init, (0, 1), 'constant', 0)
    ligand_node_feature_init = F.pad(ligand_node_feature_init, (0, 1), 'constant', 0)
    edge_feature_init = F.pad(edge_feature_init, (0, 1), 'constant', 0)
```

## 4. 坐标噪声处理

### 4.1 蛋白质坐标噪声
```python
# 生成坐标噪声掩蔽
p_coor_noise_bool = self.gen_mask_index([p_res_label], self.mask_rate_p)

if data_type == 'semi':
    # 为半监督数据添加坐标噪声
    coor_init[coor_noise_bool] = coor_init[coor_noise_bool] + torch.randn(
        coor_init[coor_noise_bool].shape) * self.noise_distance / self.coor_scale
    flex_coor_mask = F.pad(p_coor_noise_bool, (0, len_ligand), 'constant', 1).float()
else:
    # 正常数据不添加蛋白质噪声，但配体仍然灵活
    flex_coor_mask = F.pad(torch.zeros(len_protein_before_sampling), (0, len_ligand), 'constant', 1).to(torch.float)
```

### 4.2 距离图处理
```python
if data_type == 'semi':
    # 对被掩蔽的节点，将其距离信息也掩蔽
    edge_feature_init[p_x_mask_bool, :, -1] = -1
    edge_feature_init[:, p_x_mask_bool, -1] = -1
    edge_feature_init[l_x_mask_bool, :, -1] = -1
    edge_feature_init[:, l_x_mask_bool, -1] = -1
```

## 5. 训练时的半监督数据采样

### 5.1 数据采样策略
```python
def __getitem__(self, i):
    if self.mode == 'train':
        # 根据semi_rate决定是否使用半监督数据
        f_name = self.pdbbind_list[i] if torch.rand(1) > self.semi_rate else random.choice(self.semi_list)
```

- `semi_rate`：半监督数据的采样概率
- 训练时会随机选择使用标注数据还是半监督数据
- 半监督数据从单独的文件列表中随机选择

## 6. 掩蔽机制的训练目标

### 6.1 节点特征预测
- **蛋白质**：预测被掩蔽的原子类型和残基类型
- **配体**：预测被掩蔽的原子类型

### 6.2 边特征预测
- 预测被掩蔽的边的特征类型

### 6.3 坐标去噪
- 从噪声坐标恢复真实坐标（仅针对半监督数据）

## 7. 关键设计思想

1. **自监督学习**：通过掩蔽部分特征并预测它们来学习分子表示
2. **无标签数据利用**：半监督数据没有亲和力标签，但可以用于特征学习
3. **多层次掩蔽**：同时掩蔽节点特征、边特征和坐标信息
4. **对称性保持**：边掩蔽保持对称性，符合分子图的物理性质
5. **渐进式学习**：通过控制semi_rate来平衡监督和自监督学习

这种设计使得模型能够有效利用大量无标签的蛋白质-配体复合物数据，通过自监督的掩蔽预测任务学习更好的分子表示，从而提升在有标签数据上的性能。
