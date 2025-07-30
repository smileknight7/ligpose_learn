#!/usr/bin/env python3
"""
Script to read and convert pkl files to txt format
Specifically designed for ligand-protein binding data files
"""

import pickle
import numpy as np
import sys
import os
from pathlib import Path

def read_pkl_file(pkl_path):
    """Read and analyze pkl file contents"""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error reading pkl file: {e}")
        return None

def format_array_info(arr):
    """Format numpy array information"""
    if isinstance(arr, np.ndarray):
        return f"ndarray(shape={arr.shape}, dtype={arr.dtype})"
    else:
        return str(type(arr))

def write_data_to_txt(data, output_path):
    """Write data to txt file with proper formatting"""
    with open(output_path, 'w') as f:
        f.write("PKL File Contents Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        if isinstance(data, dict):
            f.write(f"Data type: Dictionary with {len(data)} keys\n\n")
            for key, value in data.items():
                f.write(f"Key: {key}\n")
                f.write(f"Type: {type(value)}\n")
                
                if isinstance(value, np.ndarray):
                    f.write(f"Shape: {value.shape}\n")
                    f.write(f"Dtype: {value.dtype}\n")
                    f.write(f"Min: {np.min(value)}, Max: {np.max(value)}\n")
                    if value.size <= 100:  # Only show small arrays
                        f.write(f"Values:\n{value}\n")
                    else:
                        f.write(f"First 10 values: {value.flatten()[:10]}\n")
                elif isinstance(value, (list, tuple)):
                    f.write(f"Length: {len(value)}\n")
                    if len(value) <= 20:
                        f.write(f"Values: {value}\n")
                    else:
                        f.write(f"First 10 values: {value[:10]}\n")
                elif isinstance(value, str):
                    f.write(f"Value: {value}\n")
                else:
                    f.write(f"Value: {value}\n")
                
                f.write("-" * 30 + "\n")
                
        elif isinstance(data, (list, tuple)):
            f.write(f"Data type: {type(data)} with {len(data)} elements\n\n")
            for i, item in enumerate(data[:20]):  # Show first 20 items
                f.write(f"Item {i}: {format_array_info(item)}\n")
                if isinstance(item, np.ndarray) and item.size <= 50:
                    f.write(f"Values: {item}\n")
                f.write("-" * 20 + "\n")
                
        elif isinstance(data, np.ndarray):
            f.write(f"Data type: numpy array\n")
            f.write(f"Shape: {data.shape}\n")
            f.write(f"Dtype: {data.dtype}\n")
            f.write(f"Min: {np.min(data)}, Max: {np.max(data)}\n")
            if data.size <= 1000:
                f.write(f"Values:\n{data}\n")
            else:
                f.write(f"First 100 values:\n{data.flatten()[:100]}\n")
        else:
            f.write(f"Data type: {type(data)}\n")
            f.write(f"Value: {data}\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: python read_pkl_to_txt.py <pkl_file_path>")
        sys.exit(1)
    
    pkl_path = sys.argv[1]
    
    if not os.path.exists(pkl_path):
        print(f"Error: File {pkl_path} does not exist")
        sys.exit(1)
    
    # Read pkl file
    print(f"Reading pkl file: {pkl_path}")
    data = read_pkl_file(pkl_path)
    
    if data is None:
        print("Failed to read pkl file")
        sys.exit(1)
    
    # Generate output filename
    pkl_file = Path(pkl_path)
    output_path = pkl_file.with_suffix('.txt')
    
    # Write to txt file
    print(f"Writing analysis to: {output_path}")
    write_data_to_txt(data, output_path)
    
    print("Conversion completed!")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()
