#!/usr/bin/env python
"""
Pre-Training Verification Script

This script checks all critical configuration and setup issues before training
to ensure the best chance of reproducing paper results.

Based on issues identified in logs/analysis.txt

Usage:
    python verify_setup.py --name <experiment_name> --dataroot <data_path>
"""

import os
import argparse
import json
from pathlib import Path
import warnings


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_section(title):
    """Print section header."""
    print("\n" + "="*80)
    print(f"{bcolors.HEADER}{bcolors.BOLD}{title}{bcolors.ENDC}")
    print("="*80)


def check_pass(message):
    """Print pass message."""
    print(f"{bcolors.OKGREEN}✅ {message}{bcolors.ENDC}")


def check_warning(message):
    """Print warning message."""
    print(f"{bcolors.WARNING}⚠️  {message}{bcolors.ENDC}")


def check_fail(message):
    """Print fail message."""
    print(f"{bcolors.FAIL}❌ {message}{bcolors.ENDC}")


def check_info(message):
    """Print info message."""
    print(f"{bcolors.OKCYAN}ℹ️  {message}{bcolors.ENDC}")


def check_dataset(dataroot, vertebra_json):
    """Check dataset configuration and size."""
    print_section("🔍 DATASET VERIFICATION")
    
    issues = []
    
    # Check CT folder
    ct_folder = os.path.join(dataroot, "CT")
    if not os.path.exists(ct_folder):
        check_fail(f"CT folder not found: {ct_folder}")
        issues.append("missing_ct_folder")
    else:
        # Count CT files
        ct_files = list(Path(ct_folder).glob("*.nii*"))
        n_ct = len(ct_files)
        check_info(f"Found {n_ct} CT files in {ct_folder}")
        
        if n_ct < 1000:
            check_warning(f"CT files ({n_ct}) < 1217 (paper's training set)")
            check_warning("Your dataset may be incomplete!")
            issues.append("insufficient_data")
        else:
            check_pass(f"CT files: {n_ct} (sufficient)")
    
    # Check label folder
    label_folder = os.path.join(dataroot, "label")
    if not os.path.exists(label_folder):
        check_warning(f"Label folder not found: {label_folder}")
        issues.append("missing_label_folder")
    else:
        label_files = list(Path(label_folder).glob("*.nii*"))
        check_info(f"Found {len(label_files)} label files")
    
    # Check heatmap/CAM folder
    cam_folder = os.path.join(dataroot, "heatmap")
    if not os.path.exists(cam_folder):
        check_warning(f"Heatmap folder not found: {cam_folder}")
        check_warning("You may need to run grad_CAM_3d_sagittal.py first")
        issues.append("missing_cam_folder")
    else:
        cam_files = list(Path(cam_folder).glob("*.nii*"))
        check_info(f"Found {len(cam_files)} heatmap files")
        if len(cam_files) < n_ct:
            check_warning(f"Heatmap files ({len(cam_files)}) < CT files ({n_ct})")
    
    # Check vertebra_data.json
    if not os.path.exists(vertebra_json):
        check_fail(f"Vertebra data JSON not found: {vertebra_json}")
        issues.append("missing_json")
    else:
        with open(vertebra_json, 'r') as f:
            data = json.load(f)
        
        n_train = len(data.get('train', {}))
        n_val = len(data.get('val', {}))
        n_test = len(data.get('test', {}))
        n_total = n_train + n_val + n_test
        
        check_info(f"Vertebra JSON statistics:")
        print(f"   Train: {n_train} samples")
        print(f"   Val:   {n_val} samples")
        print(f"   Test:  {n_test} samples")
        print(f"   Total: {n_total} samples")
        
        if n_train < 1000:
            check_warning(f"Training samples ({n_train}) < 1217 (paper)")
            issues.append("insufficient_training_samples")
        else:
            check_pass(f"Training samples: {n_train}")
        
        # Check if all entries have labels
        labels_present = all(isinstance(v, int) for v in data['train'].values())
        if labels_present:
            check_pass("All training samples have labels")
        else:
            check_warning("Some training samples missing labels")
    
    return issues


def check_training_config(checkpoints_dir, name):
    """Check training configuration from saved options."""
    print_section("⚙️  TRAINING CONFIGURATION")
    
    issues = []
    
    opt_path = os.path.join(checkpoints_dir, name, 'train_opt.txt')
    
    if not os.path.exists(opt_path):
        check_info("No previous training config found (starting fresh)")
        check_warning("Manual verification required:")
        print("   - n_epochs: Should be ~100-1000")
        print("   - n_epochs_decay: Should be ~900")
        print("   - lr_decay_iters: Should be 50-100")
        print("   - lambda_L1: Should be 200")
        print("   - lambda_edge: Should be 80-800")
        print("   - lambda_height: Should be 40-80")
        return issues
    
    # Parse config file
    with open(opt_path, 'r') as f:
        config = {}
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                config[key.strip()] = value.strip()
    
    # Check n_epochs
    if 'n_epochs' in config:
        n_epochs = int(config['n_epochs'])
        n_epochs_decay = int(config.get('n_epochs_decay', 0))
        total_epochs = n_epochs + n_epochs_decay
        
        check_info(f"n_epochs: {n_epochs}")
        check_info(f"n_epochs_decay: {n_epochs_decay}")
        check_info(f"Total epochs: {total_epochs}")
        
        if total_epochs < 200:
            check_warning(f"Total epochs ({total_epochs}) << 1000 (paper)")
            check_warning("Insufficient training time may lead to poor results")
            issues.append("insufficient_epochs")
        else:
            check_pass(f"Total epochs: {total_epochs}")
    
    # Check loss weights
    loss_weights = {
        'lambda_L1': (200.0, 'L1 reconstruction loss'),
        'lambda_edge': (80.0, 'Edge loss (paper: 80, current default: 800)'),
        'lambda_height': (40.0, 'Height restoration loss'),
        'lambda_dice': (15.0, 'Dice loss'),
        'lambda_coarse_dice': (10.0, 'Coarse Dice loss')
    }
    
    print("\nLoss Weights:")
    for key, (expected, desc) in loss_weights.items():
        if key in config:
            actual = float(config[key])
            print(f"   {key}: {actual} ({desc})")
            if abs(actual - expected) > expected * 0.5:
                check_warning(f"{key} significantly different from default {expected}")
        else:
            check_warning(f"{key} not found in config")
    
    # Check GPU config
    if 'gpu_ids' in config:
        gpu_ids = config['gpu_ids']
        check_info(f"GPU IDs: {gpu_ids}")
        if ',' in str(gpu_ids):
            check_warning("Multi-GPU training detected")
            check_warning("Paper used single GPU - may affect batch normalization")
            issues.append("multi_gpu")
    
    # Check batch size
    if 'batch_size' in config:
        batch_size = int(config['batch_size'])
        check_info(f"Batch size: {batch_size}")
    
    return issues


def check_code_issues():
    """Check for known code bugs."""
    print_section("🐛 CODE VERIFICATION")
    
    issues = []
    
    # Check train.py for SSIM bug
    train_path = "train.py"
    if os.path.exists(train_path):
        with open(train_path, 'r') as f:
            train_code = f.read()
        
        # Check for fixed SSIM calculation
        if 'multichannel=True' in train_code and 'ssim(' in train_code:
            check_fail("Found 'multichannel=True' in SSIM call")
            check_fail("This is a bug for grayscale images - SSIM will be incorrect")
            issues.append("ssim_multichannel_bug")
        else:
            check_pass("SSIM multichannel bug not detected")
        
        # Check for divide-by-zero protection in PSNR
        if 'if mse < 1e-10:' in train_code or 'if mse == 0:' in train_code:
            check_pass("PSNR divide-by-zero protection found")
        else:
            check_warning("No divide-by-zero protection for PSNR detected")
            check_warning("May crash if MSE = 0")
            issues.append("missing_psnr_protection")
        
        # Check for data_range calculation
        if 'data_range = max' in train_code:
            check_pass("Proper data_range calculation found")
        else:
            check_warning("data_range calculation may be inconsistent")
    
    # Check pix2pix_model.py for loss weights
    model_path = "models/pix2pix_model.py"
    if os.path.exists(model_path):
        with open(model_path, 'r') as f:
            model_code = f.read()
        
        # Check for configurable loss weights
        if 'lambda_edge' in model_code and 'parser.add_argument' in model_code:
            check_pass("Configurable loss weights detected")
        else:
            check_warning("Loss weights may be hardcoded")
            check_warning("Check for '* 800' or '* 40' etc. in backward_G()")
    
    return issues


def check_module_presence():
    """Check for critical model modules."""
    print_section("🏗️  MODEL ARCHITECTURE")
    
    issues = []
    
    check_warning("Manual verification required for:")
    print("   [ ] AHVS (Adjacent Healthy Vertebrae Segmentation) module")
    print("   [ ] EEM (Edge Enhancement Module)")
    print("   [ ] SHRM (Self-adaptive Height Restoration Module)")
    print("\n   Search your code for these keywords:")
    print("   - 'AHVS', 'Adjacent', 'healthy_vertebra'")
    print("   - 'edge', 'sobel', 'edge_loss'")
    print("   - 'height', 'SHRM', 'height_restore'")
    print("\n   If missing, Dice=0.63 vs paper's 0.894 is expected!")
    
    return issues


def main():
    parser = argparse.ArgumentParser(description='Verify setup before training')
    parser.add_argument('--name', type=str, default='healthivert_full', 
                       help='Experiment name')
    parser.add_argument('--dataroot', type=str, 
                       default='verse19/straighten',
                       help='Path to dataset root')
    parser.add_argument('--vertebra_json', type=str,
                       default='vertebra_data_test.json',
                       help='Path to vertebra_data.json')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                       help='Path to checkpoints directory')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f"{bcolors.HEADER}{bcolors.BOLD}PRE-TRAINING VERIFICATION SCRIPT{bcolors.ENDC}")
    print("="*80)
    print(f"Experiment: {args.name}")
    print(f"Dataroot: {args.dataroot}")
    print(f"Vertebra JSON: {args.vertebra_json}")
    
    all_issues = []
    
    # Run checks
    all_issues.extend(check_dataset(args.dataroot, args.vertebra_json))
    all_issues.extend(check_training_config(args.checkpoints_dir, args.name))
    all_issues.extend(check_code_issues())
    all_issues.extend(check_module_presence())
    
    # Summary
    print_section("📋 SUMMARY")
    
    if len(all_issues) == 0:
        check_pass("No critical issues detected!")
        print("\n✅ Ready to train. Expected behavior:")
        print("   - SSIM: 0.93-0.95")
        print("   - PSNR: 30-35 dB")
        print("   - Dice: 0.85-0.89")
        print("   - Training time: 1000+ epochs")
    else:
        check_warning(f"Found {len(all_issues)} potential issues:")
        for issue in set(all_issues):
            print(f"   - {issue}")
        
        print("\n⚠️  Address these issues before training to improve results.")
        print("\nCRITICAL ISSUES to fix:")
        if 'ssim_multichannel_bug' in all_issues:
            print("   1. Remove 'multichannel=True' from SSIM calls in train.py")
        if 'insufficient_data' in all_issues:
            print("   2. Ensure you have full Verse2019 dataset (1217+ samples)")
        if 'insufficient_epochs' in all_issues:
            print("   3. Set --n_epochs 100 --n_epochs_decay 900")
    
    print("\n" + "="*80)
    print("Verification complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
