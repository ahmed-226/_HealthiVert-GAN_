"""
UPDATED: Added RHDR (Relative Height Difference Ratio) metric calculation
Original script calculated: global_psnr, global_ssim, patch_psnr, patch_ssim, iou, rv_diff, dice
Updated script now includes: RHDR metric for height-based evaluation

UPDATED: Added command-line argument parsing for flexible folder paths
"""

import os
import nibabel as nib
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import json
import pandas as pd
from sklearn.model_selection import ParameterGrid
import math
import argparse

def calculate_iou(ori_seg, fake_seg):
    intersection = np.sum(ori_seg * fake_seg)
    union = np.sum(ori_seg + fake_seg > 0)
    if union == 0:
        return 0
    else:
        return intersection / union
    
def calculate_dice(ori_seg, fake_seg):
    # 计算两个分割之间的交集
    intersection = np.sum(ori_seg * fake_seg)
    # 计算两个分割之间的并集
    union = np.sum(ori_seg) + np.sum(fake_seg)
    # 如果并集为零，返回0，否则返回Dice系数
    if union == 0:
        return 0
    else:
        return 2.0 * intersection / union


def relative_volume_difference(ori_seg, fake_seg):
    volume_ori = np.sum(ori_seg)
    volume_fake = np.sum(fake_seg)
    if volume_ori == 0:
        return 0
    else:
        return np.abs(volume_ori - volume_fake) / volume_ori

# ===== UPDATED: New function to calculate RHDR (Relative Height Difference Ratio) =====
def calculate_rhdr(ori_seg, fake_seg):
    """
    Calculate Relative Height Difference Ratio (RHDR) based on vertebra height
    
    RHDR = (|H_predicted - H_actual| / H_actual) * 100%
    
    where H = number of slices where segmentation exists (height in axial direction)
    
    Args:
        ori_seg: Original (ground truth) segmentation volume
        fake_seg: Generated (predicted) segmentation volume
    
    Returns:
        RHDR value as percentage. Lower is better (closer to healthy vertebra).
    """
    # Calculate height = number of unique slices with segmentation
    ori_height = np.sum(np.any(ori_seg, axis=(0, 1)))  # Count slices in z-direction
    fake_height = np.sum(np.any(fake_seg, axis=(0, 1)))
    
    if ori_height == 0:
        return 0
    else:
        rhdr = (np.abs(ori_height - fake_height) / ori_height) * 100
        return rhdr
# ===== END UPDATED SECTION ====

def process_images(ori_ct_path, fake_ct_path, ori_seg_path, fake_seg_path):
    ori_ct = nib.load(ori_ct_path).get_fdata()
    fake_ct = nib.load(fake_ct_path).get_fdata()
    ori_seg_temp = nib.load(ori_seg_path).get_fdata()
    ori_seg = np.zeros_like(ori_seg_temp)
    fake_seg_temp = nib.load(fake_seg_path).get_fdata()
    fake_seg = np.zeros_like(fake_seg_temp)
    
    # Extract vertebra ID - handle both .nii and .nii.gz formats
    seg_filename = os.path.basename(ori_seg_path)
    if seg_filename.endswith('.nii.gz'):
        base_name = seg_filename[:-7]
    elif seg_filename.endswith('.nii'):
        base_name = seg_filename[:-4]
    else:
        raise ValueError(f"Unknown file format: {seg_filename}")
    
    label = int(base_name.split('_')[-1])
    ori_seg[ori_seg_temp==label] = 1
    fake_seg[fake_seg_temp==label] = 1

    patch_psnr_list = []
    patch_ssim_list = []
    global_psnr_list = []
    global_ssim_list = []
    
    iou_value = calculate_iou(ori_seg, fake_seg)
    dice_value = calculate_dice(ori_seg, fake_seg)
    rv_diff = relative_volume_difference(ori_seg, fake_seg)
    # UPDATED: Calculate RHDR metric for height-based evaluation
    rhdr_value = calculate_rhdr(ori_seg, fake_seg)
    # UPDATED: Calculate RHDR metric for height-based evaluation
    rhdr_value = calculate_rhdr(ori_seg, fake_seg)
    
    loc = np.where(ori_seg)
    z0 = min(loc[2])
    z1 = max(loc[2])
    range_length = z1 - z0 + 1
    new_range_length = int(range_length * 4 / 5)
    new_z0 = z0 + (range_length - new_range_length) // 2
    new_z1 = new_z0 + new_range_length - 1
    


    for z in range(new_z0, new_z1 + 1):
        if np.sum(ori_seg[:,:,z]) > 400:
            coords = np.argwhere(ori_seg[:,:,z])
            x1, x2 = min(coords[:, 0]), max(coords[:, 0])

            crop_ori_ct = ori_ct[x1:x2+1, :, z]
            crop_fake_ct = fake_ct[x1:x2+1, :, z]

            psnr_value = compare_psnr(crop_ori_ct, crop_fake_ct, data_range=crop_ori_ct.max() - crop_ori_ct.min())
            ssim_value = compare_ssim(crop_ori_ct, crop_fake_ct, data_range=crop_ori_ct.max() - crop_ori_ct.min())

            if not np.isnan(psnr_value):
                patch_psnr_list.append(psnr_value)
            if not np.isnan(ssim_value):
                patch_ssim_list.append(ssim_value)
            
    for z in range(new_z0, new_z1 + 1):
        if np.sum(ori_seg[:,:,z]) > 400:
            psnr_value = compare_psnr(ori_ct[:,:,z], fake_ct[:,:,z], data_range=ori_ct[:,:,z].max() - ori_ct[:,:,z].min())
            ssim_value = compare_ssim(ori_ct[:,:,z], fake_ct[:,:,z], data_range=ori_ct[:,:,z].max() - ori_ct[:,:,z].min())

            if not np.isnan(psnr_value):
                global_psnr_list.append(psnr_value)
            if not np.isnan(ssim_value):
                global_ssim_list.append(ssim_value)
            
    avg_patch_psnr = np.mean(patch_psnr_list) if patch_psnr_list else 0  # 检查列表是否为空
    avg_patch_ssim = np.mean(patch_ssim_list) if patch_ssim_list else 0  # 检查列表是否为空
    avg_global_psnr = np.mean(global_psnr_list) if global_psnr_list else 0  # 检查列表是否为空
    avg_global_ssim = np.mean(global_ssim_list) if global_ssim_list else 0  # 检查列表是否为空

    # UPDATED: Added rhdr_value to return statement
    return avg_global_psnr, avg_global_ssim, avg_patch_psnr, avg_patch_ssim, iou_value, rv_diff, dice_value, rhdr_value

def average_metrics(lists):
    return np.mean(lists)

def main():
    # ===== UPDATED: Add command-line argument parsing =====
    parser = argparse.ArgumentParser(description='Generate evaluation metrics for 3D CT reconstructions')
    parser.add_argument('--ori-ct-folder', type=str, default='CT', help='Path to original CT folder')
    parser.add_argument('--ori-seg-folder', type=str, default='label', help='Path to original segmentation folder')
    parser.add_argument('--fake-ct-folder', type=str, required=True, help='Path to generated CT folder')
    parser.add_argument('--fake-seg-folder', type=str, required=True, help='Path to generated segmentation folder')
    parser.add_argument('--json-path', type=str, default='vertebra_data.json', help='Path to vertebra data JSON file')
    parser.add_argument('--output-folder', type=str, default='evaluation/generation_metric', help='Path to save results')
    args = parser.parse_args()
    
    ori_ct_folder = args.ori_ct_folder
    ori_seg_folder = args.ori_seg_folder
    json_path = args.json_path
    save_folder = args.output_folder
    fake_ct_folder = args.fake_ct_folder
    fake_seg_folder = args.fake_seg_folder
    # ===== END UPDATED SECTION =====
    
    print(f"\n📊 Starting evaluation...")
    print(f"   Original CT: {ori_ct_folder}")
    print(f"   Original Segmentation: {ori_seg_folder}")
    print(f"   Generated CT: {fake_ct_folder}")
    print(f"   Generated Segmentation: {fake_seg_folder}")
    print(f"   JSON Config: {json_path}\n")
    
    with open(json_path, 'r') as file:
        vertebra_set = json.load(file)
    
    val_normal_vert = []
    for patient_vert_id in vertebra_set['val'].keys():
        if int(vertebra_set['val'][patient_vert_id]) == 0:
            val_normal_vert.append(patient_vert_id)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Initialize metrics lists
    metrics_lists = {
        'global_psnr': [], 
        'global_ssim': [], 
        'patch_psnr': [], 
        'patch_ssim': [], 
        'iou': [], 
        'rv_diff': [], 
        'dice': [], 
        'rhdr': []
    }
    count = 0
    
    for filename in os.listdir(ori_ct_folder):
        # Support both .nii.gz and .nii file formats
        if filename.endswith(".nii.gz"):
            base_name = filename[:-7]
        elif filename.endswith(".nii"):
            base_name = filename[:-4]
        else:
            continue
        
        if base_name in val_normal_vert:
            ori_ct_path = os.path.join(ori_ct_folder, filename)
            ori_seg_path = os.path.join(ori_seg_folder, filename)
            
            # For fake files, try both formats (.nii.gz and .nii)
            fake_ct_path = os.path.join(fake_ct_folder, filename)
            fake_seg_path = os.path.join(fake_seg_folder, filename)
            
            # If .nii.gz doesn't exist, try .nii
            if not os.path.exists(fake_ct_path) and filename.endswith('.nii.gz'):
                fake_ct_alt = os.path.join(fake_ct_folder, base_name + '.nii')
                if os.path.exists(fake_ct_alt):
                    fake_ct_path = fake_ct_alt
            
            if not os.path.exists(fake_seg_path) and filename.endswith('.nii.gz'):
                fake_seg_alt = os.path.join(fake_seg_folder, base_name + '.nii')
                if os.path.exists(fake_seg_alt):
                    fake_seg_path = fake_seg_alt
            
            # Check if fake files exist
            if not os.path.exists(fake_ct_path) or not os.path.exists(fake_seg_path):
                print(f"⚠️  Skipping {filename}: generated files not found")
                continue
            
            # ===== UPDATED: Unpack rhdr_value from process_images return =====
            global_psnr, global_ssim, patch_psnr, patch_ssim, iou, rv_diff, dice, rhdr = process_images(
                ori_ct_path, fake_ct_path, ori_seg_path, fake_seg_path)
            # ===== END UPDATED SECTION =====
            
            if math.isnan(patch_psnr) or math.isnan(patch_ssim):
                print(f"⚠️  {filename}: PSNR or SSIM returned NaN, skipping")
                continue
            if patch_psnr == 0 or patch_ssim == 0:
                print(f"⚠️  {filename}: PSNR or SSIM returned 0, skipping")
                continue
                
            metrics_lists['global_psnr'].append(global_psnr)
            metrics_lists['global_ssim'].append(global_ssim)
            metrics_lists['patch_psnr'].append(patch_psnr)
            metrics_lists['patch_ssim'].append(patch_ssim)
            metrics_lists['iou'].append(iou)
            metrics_lists['rv_diff'].append(rv_diff)
            metrics_lists['dice'].append(dice)
            # ===== UPDATED: Append RHDR value to metrics list =====
            metrics_lists['rhdr'].append(rhdr)
            # ===== END UPDATED SECTION =====
            count += 1

    # Calculate average metrics
    avg_metrics = {key: average_metrics(value) for key, value in metrics_lists.items()}
    
    # Save results
    output_file = os.path.join(save_folder, "evaluation_metrics.txt")
    with open(output_file, "w") as file:
        for metric, value in avg_metrics.items():
            file.write(f"Average {metric.upper()}: {value:.4f}\n")
    
    print(f"\n✅ Evaluation Complete!")
    print(f"   Processed {count} samples")
    print(f"   Results saved to: {output_file}")
    print(f"\n📈 Average Metrics:")
    for metric, value in avg_metrics.items():
        print(f"   {metric.upper()}: {value:.4f}")

if __name__ == "__main__":
    main()