import os
from utils.training_utils import split_pdbbind_semi

# 测试配置

split_rate = [0.8, 0.1, 0.1]  # 80% 训练, 10% 验证, 10% 测试

labeled_set_path = "/root/work_file/refined_set_preprocessed"
unlabeled_set_path = "/path/to/general_set"

# 调用函数
train_list, val_list, test_listr = split_pdbbind_semi(
    pdbbind_path=pdbbind_path,
    split_rate=split_rate,
    core_list_path=core_list_path,
    labeled_set_path=labeled_set_path,
    unlabeled_set_path=unlabeled_set_path
)

# 输出结果统计
print(f"训练集大小: {len(train_list)}")
print(f"其中有标签样本: {sum(1 for x in train_list if x[1] == 1)}")
print(f"其中无标签样本: {sum(1 for x in train_list if x[1] == 0)}")
print(f"验证集大小: {len(val_list)}")
print(f"测试集大小: {len(test_list)}")

# 检查几个样本
print("\n训练集示例:")
for i in range(min(5, len(train_list))):
    print(f"样本 {i}: {train_list[i]}")