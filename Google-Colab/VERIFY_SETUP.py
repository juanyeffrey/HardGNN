#!/usr/bin/env python3
"""
HardGNN Setup Verification Script
================================

This script verifies that all necessary components are in place for running
HardGNN with hard negative sampling on Google Colab.

Run this script to ensure everything is properly configured before starting
your experiment.
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a required file exists."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (MISSING)")
        return False

def check_directory_exists(dirpath, description):
    """Check if a required directory exists."""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        contents = os.listdir(dirpath)
        print(f"✅ {description}: {dirpath} ({len(contents)} items)")
        return True
    else:
        print(f"❌ {description}: {dirpath} (MISSING)")
        return False

def check_hard_neg_implementation():
    """Check if hard negative sampling is implemented in model.py."""
    try:
        with open('model.py', 'r') as f:
            content = f.read()
            
        checks = [
            ('sample_hard_negatives', 'Hard negative sampling method'),
            ('compute_infonce_loss', 'InfoNCE loss computation'),
            ('args.use_hard_neg', 'Hard negative sampling flag'),
            ('args.hard_neg_top_k', 'Hard negative top-K parameter'),
            ('args.contrastive_weight', 'Contrastive loss weight'),
            ('self.contrastive_loss', 'Contrastive loss variable')
        ]
        
        all_present = True
        for check, desc in checks:
            if check in content:
                print(f"✅ {desc}: Found")
            else:
                print(f"❌ {desc}: Missing")
                all_present = False
                
        return all_present
    except Exception as e:
        print(f"❌ Error checking model.py: {e}")
        return False

def check_dataset_configuration():
    """Check if dataset configuration is properly implemented."""
    try:
        with open('HardGNN_Colab_Script.py', 'r') as f:
            content = f.read()
            
        required_parts = [
            "DATASET = 'gowalla'",
            "def configure_dataset",
            "args.use_hard_neg = True",
            "args.hard_neg_top_k = 5",
            "args.contrastive_weight = 0.1"
        ]
        
        all_present = True
        for part in required_parts:
            if part in content:
                print(f"✅ Configuration: {part}")
            else:
                print(f"❌ Configuration: {part} (Missing)")
                all_present = False
                
        return all_present
    except Exception as e:
        print(f"❌ Error checking script configuration: {e}")
        return False

def main():
    """Main verification function."""
    print("🔍 HardGNN Setup Verification")
    print("=" * 50)
    
    # Check core files
    print("\n📁 Core Files:")
    files_ok = True
    files_ok &= check_file_exists('README.md', 'Main documentation')
    files_ok &= check_file_exists('HardGNN_Colab_Script.py', 'Main script')
    files_ok &= check_file_exists('model.py', 'Model implementation')
    files_ok &= check_file_exists('Params.py', 'Parameters')
    files_ok &= check_file_exists('DataHandler.py', 'Data handler')
    files_ok &= check_file_exists('requirements.txt', 'Dependencies')
    
    # Check directories
    print("\n📂 Directories:")
    dirs_ok = True
    dirs_ok &= check_directory_exists('Utils', 'Utility modules')
    dirs_ok &= check_directory_exists('Datasets', 'Dataset directory')
    
    # Check datasets
    print("\n💾 Available Datasets:")
    dataset_dirs = ['yelp', 'amazon', 'gowalla', 'movielens']
    available_datasets = []
    for dataset in dataset_dirs:
        path = f'Datasets/{dataset}'
        if check_directory_exists(path, f'{dataset.capitalize()} dataset'):
            available_datasets.append(dataset)
    
    # Check implementation
    print("\n🔧 Hard Negative Sampling Implementation:")
    impl_ok = check_hard_neg_implementation()
    
    # Check configuration
    print("\n⚙️  Script Configuration:")
    config_ok = check_dataset_configuration()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    if files_ok and dirs_ok and impl_ok and config_ok:
        print("🎉 ALL CHECKS PASSED!")
        print("\n✅ Your HardGNN setup is ready for Google Colab!")
        print("\nNext steps:")
        print("1. Upload this folder to Google Drive")
        print("2. Open Google Colab with GPU runtime")
        print("3. Set DATASET parameter in HardGNN_Colab_Script.py")
        print("4. Copy and run the script sections in Colab")
        
        if available_datasets:
            print(f"\n📈 Available datasets: {', '.join(available_datasets)}")
        else:
            print("\n⚠️  Note: No dataset files found. You'll need to add dataset files to Datasets/ directories.")
            
    else:
        print("❌ SETUP ISSUES DETECTED!")
        print("\nPlease fix the missing components before proceeding.")
        
        if not files_ok:
            print("- Missing core files")
        if not dirs_ok:
            print("- Missing directories")
        if not impl_ok:
            print("- Hard negative sampling implementation issues")
        if not config_ok:
            print("- Script configuration issues")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main() 