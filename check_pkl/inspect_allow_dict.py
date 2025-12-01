#!/usr/bin/env python3
"""
Script to inspect the content of pdbbind_screening_allow_dict.pkl
"""

import pickle
import argparse
import os
from collections import Counter

def inspect_allow_dict(pkl_path, pdbbind_path=None):
    """
    Inspect the content of allow_dict pkl file
    
    Args:
        pkl_path: Path to the pkl file
        pdbbind_path: Optional path to pdbbind data directory for comparison
    """
    print(f"Inspecting allow_dict file: {pkl_path}")
    print("=" * 60)
    
    # Load the pkl file
    try:
        with open(pkl_path, 'rb') as f:
            allow_dict = pickle.load(f)
        print(f"✓ Successfully loaded allow_dict")
        print(f"Type: {type(allow_dict)}")
        print(f"Total entries: {len(allow_dict)}")
        print()
    except Exception as e:
        print(f"✗ Error loading pkl file: {e}")
        return
    
    # Basic statistics
    print("Basic Statistics:")
    print("-" * 30)
    all_ligand_ids = list(allow_dict.keys())
    all_protein_ids = []
    for protein_list in allow_dict.values():
        all_protein_ids.extend(protein_list)
    
    print(f"Unique ligand IDs: {len(all_ligand_ids)}")
    print(f"Total protein entries: {len(all_protein_ids)}")
    print(f"Unique protein IDs: {len(set(all_protein_ids))}")
    print()
    
    # Show some examples
    print("Sample entries:")
    print("-" * 30)
    for i, (ligand_id, protein_ids) in enumerate(list(allow_dict.items())[:5]):
        print(f"Ligand {ligand_id}: {len(protein_ids)} proteins -> {protein_ids[:3]}{'...' if len(protein_ids) > 3 else ''}")
    print()
    
    # Distribution of protein counts per ligand
    protein_counts = [len(protein_list) for protein_list in allow_dict.values()]
    print("Distribution of protein counts per ligand:")
    print("-" * 30)
    print(f"Min proteins per ligand: {min(protein_counts)}")
    print(f"Max proteins per ligand: {max(protein_counts)}")
    print(f"Average proteins per ligand: {sum(protein_counts) / len(protein_counts):.2f}")
    
    # Count distribution
    count_distribution = Counter(protein_counts)
    print("\nProtein count distribution:")
    for count, freq in sorted(count_distribution.items())[:10]:
        print(f"  {count} proteins: {freq} ligands")
    if len(count_distribution) > 10:
        print(f"  ... and {len(count_distribution) - 10} more")
    print()
    
    # Most common proteins
    protein_counter = Counter(all_protein_ids)
    print("Most common proteins:")
    print("-" * 30)
    for protein_id, count in protein_counter.most_common(10):
        print(f"  {protein_id}: appears {count} times")
    print()
    
    # Check against actual data files if pdbbind_path is provided
    if pdbbind_path and os.path.exists(pdbbind_path):
        print(f"Checking against actual data files in: {pdbbind_path}")
        print("-" * 30)
        
        # Get actual files
        try:
            actual_files = set()
            for file in os.listdir(pdbbind_path):
                if file.endswith('.npz'):
                    actual_files.add(file.replace('.npz', ''))
            
            print(f"Actual .npz files found: {len(actual_files)}")
            
            # Check ligand IDs
            missing_ligands = set(all_ligand_ids) - actual_files
            extra_files = actual_files - set(all_ligand_ids)
            
            print(f"Ligands in allow_dict but missing .npz files: {len(missing_ligands)}")
            if missing_ligands:
                print(f"  Examples: {list(missing_ligands)[:5]}")
            
            print(f"Actual .npz files not in allow_dict: {len(extra_files)}")
            if extra_files:
                print(f"  Examples: {list(extra_files)[:5]}")
            
            # Check protein IDs
            unique_proteins = set(all_protein_ids)
            missing_proteins = unique_proteins - actual_files
            print(f"Proteins in allow_dict but missing .npz files: {len(missing_proteins)}")
            if missing_proteins:
                print(f"  Examples: {list(missing_proteins)[:5]}")
                
        except Exception as e:
            print(f"Error checking actual files: {e}")
    
    print("\nInspection completed!")

def main():
    parser = argparse.ArgumentParser(description='Inspect pdbbind_screening_allow_dict.pkl content')
    parser.add_argument('--pkl_path', type=str, 
                        default='./suppl/pdbbind_screening_allow_dict.pkl',
                        help='Path to the allow_dict pkl file')
    parser.add_argument('--pdbbind_path', type=str, 
                        default=None,
                        help='Path to pdbbind data directory for comparison')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pkl_path):
        print(f"Error: pkl file not found at {args.pkl_path}")
        return
    
    inspect_allow_dict(args.pkl_path, args.pdbbind_path)

if __name__ == "__main__":
    main()
