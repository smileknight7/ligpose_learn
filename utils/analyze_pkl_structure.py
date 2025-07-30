#!/usr/bin/env python3
"""
详细分析pkl文件内容结构的脚本
"""

import pickle
import numpy as np
import argparse
import os
from collections import defaultdict


def analyze_pkl_structure(pkl_file_path):
    """
    详细分析pkl文件的数据结构
    """
    try:
        print(f"正在分析文件: {pkl_file_path}")
        print("=" * 60)
        
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"文件大小: {os.path.getsize(pkl_file_path) / 1024:.2f} KB")
        print(f"根数据类型: {type(data)}")
        
        if isinstance(data, dict):
            print(f"字典包含 {len(data)} 个键值对")
            print("\n字典键列表:")
            for i, key in enumerate(data.keys()):
                print(f"  [{i+1}] {key}")
            
            print("\n详细键值分析:")
            print("-" * 40)
            
            for key, value in data.items():
                print(f"\n键: '{key}'")
                print(f"  值类型: {type(value)}")
                
                if isinstance(value, np.ndarray):
                    print(f"  数组形状: {value.shape}")
                    print(f"  数组数据类型: {value.dtype}")
                    print(f"  数组大小: {value.size}")
                    if value.size < 10:
                        print(f"  数组内容: {value}")
                    else:
                        print(f"  数组前5个元素: {value.flat[:5]}")
                
                elif isinstance(value, (list, tuple)):
                    print(f"  序列长度: {len(value)}")
                    if len(value) < 10:
                        print(f"  完整内容: {value}")
                    else:
                        print(f"  前5个元素: {value[:5]}")
                        print(f"  元素类型: {[type(x) for x in value[:3]]}")
                
                elif isinstance(value, (int, float)):
                    print(f"  数值: {value}")
                
                elif isinstance(value, str):
                    print(f"  字符串长度: {len(value)}")
                    if len(value) < 100:
                        print(f"  内容: '{value}'")
                    else:
                        print(f"  前100字符: '{value[:100]}...'")
                
                else:
                    print(f"  值: {value}")
                    if hasattr(value, '__dict__'):
                        print(f"  对象属性: {list(value.__dict__.keys())}")
        
        elif isinstance(data, (list, tuple)):
            print(f"序列长度: {len(data)}")
            print("前几个元素的类型:")
            for i, item in enumerate(data[:5]):
                print(f"  [{i}] {type(item)}: {item}")
        
        elif isinstance(data, np.ndarray):
            print(f"数组形状: {data.shape}")
            print(f"数组数据类型: {data.dtype}")
            print(f"数组大小: {data.size}")
            if data.size < 20:
                print(f"数组内容: {data}")
        
        else:
            print(f"数据内容: {data}")
            if hasattr(data, '__dict__'):
                print(f"对象属性: {list(data.__dict__.keys())}")
        
        return True
        
    except Exception as e:
        print(f"分析失败: {e}")
        return False


def compare_pkl_files(pkl_files):
    """
    比较多个pkl文件的结构
    """
    print("\n" + "=" * 60)
    print("多文件结构比较")
    print("=" * 60)
    
    structures = {}
    
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict):
                structure = {
                    'type': 'dict',
                    'keys': list(data.keys()),
                    'key_types': {k: type(v).__name__ for k, v in data.items()},
                    'key_shapes': {}
                }
                
                for k, v in data.items():
                    if isinstance(v, np.ndarray):
                        structure['key_shapes'][k] = v.shape
                    elif isinstance(v, (list, tuple)):
                        structure['key_shapes'][k] = f"sequence({len(v)})"
                    else:
                        structure['key_shapes'][k] = "scalar"
                
                structures[os.path.basename(pkl_file)] = structure
        
        except Exception as e:
            print(f"无法分析 {pkl_file}: {e}")
    
    # 找出共同的键
    if structures:
        all_keys = set()
        for struct in structures.values():
            if struct['type'] == 'dict':
                all_keys.update(struct['keys'])
        
        print(f"\n所有文件中发现的键: {sorted(all_keys)}")
        
        # 检查键的一致性
        common_keys = set(structures[list(structures.keys())[0]]['keys'])
        for filename, struct in structures.items():
            if struct['type'] == 'dict':
                common_keys &= set(struct['keys'])
        
        print(f"所有文件共有的键: {sorted(common_keys)}")
        
        # 显示每个文件的键类型
        print("\n每个文件的键类型对比:")
        for filename, struct in structures.items():
            print(f"\n{filename}:")
            if struct['type'] == 'dict':
                for key in sorted(struct['keys']):
                    shape_info = struct['key_shapes'].get(key, 'unknown')
                    print(f"  {key}: {struct['key_types'][key]} ({shape_info})")


def main():
    parser = argparse.ArgumentParser(description='详细分析pkl文件内容结构')
    parser.add_argument('--pkl_files', nargs='+', 
                       default=['/home/smileknight/learn/work_file/tmp/1p28.pkl'],
                       help='要分析的pkl文件路径列表')
    parser.add_argument('--compare', action='store_true',
                       help='比较多个文件的结构')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    valid_files = []
    for pkl_file in args.pkl_files:
        if os.path.exists(pkl_file):
            valid_files.append(pkl_file)
        else:
            print(f"警告: 文件不存在 {pkl_file}")
    
    if not valid_files:
        print("错误: 没有找到有效的pkl文件")
        return
    
    # 逐个分析文件
    for pkl_file in valid_files:
        analyze_pkl_structure(pkl_file)
        print("\n" + "=" * 60 + "\n")
    
    # 如果有多个文件且用户要求比较
    if len(valid_files) > 1 and args.compare:
        compare_pkl_files(valid_files)


if __name__ == '__main__':
    main()
