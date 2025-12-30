
"""
Step 0 (Optional): Generate centroid JSON files for each patient
This script finds the center of mass for each vertebra label in the segmentation mask
and saves it as a JSON file for use in the straightening process.

Usage:
    python location_json_local.py --input-dir <path_to_raw_data>
    
Expected input structure:
    input-dir/
        patient_id/
            patient_id_seg.nii.gz  (or patient_id_msk.nii.gz)
            
Output:
    input-dir/
        patient_id/
            patient_id.json  (contains centroid coordinates for each vertebra)
"""

import json
import nibabel as nib
import numpy as np
import os
import argparse

def load_nifti_data(file_path):
    nii = nib.load(file_path)
    return nii.get_fdata().astype(np.uint8)

def calculate_center_of_mass(data, label):
    center = np.mean(np.where(data == label), axis=1)
    return center # Reverse the order to match X, Y, Z

def process_directory(root_dir, sample_limit=None):
    """
    Process all patient folders to generate centroid JSON files.
    
    Args:
        root_dir: Root directory containing patient folders
        sample_limit: If set, only process this many patient folders (for testing)
    """
    all_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    
    if sample_limit is not None:
        all_folders = all_folders[:sample_limit]
        print(f"[Sample Test Mode] Processing only {len(all_folders)} patient folders")
    
    for base_filename in all_folders:
        
        #if base_filename!='0020':
        #    continue
        print(base_filename)
        original_nifti_path = os.path.join(root_dir,base_filename, base_filename+'_seg.nii.gz')
        if not os.path.exists(original_nifti_path):
            original_nifti_path = os.path.join(root_dir,base_filename, base_filename+'_msk.nii.gz')
        json_path = os.path.join(root_dir,base_filename, f'{base_filename}.json')
        #if os.path.exists(json_path):
        #    continue
        
        data_orig = nib.load(original_nifti_path).get_fdata().astype(np.uint8)
        labels_orig = np.unique(data_orig)
        labels_orig = labels_orig[labels_orig !=0]
        
        if not os.path.exists(os.path.dirname(json_path)):
            os.makedirs(os.path.dirname(json_path))
            
        json_data = []
        for label in labels_orig:
            if label==0:
                continue
            if np.sum(data_orig==label)<8000 and label==max(labels_orig):
                continue
            if np.sum(data_orig==label)<6000 and label==min(labels_orig):
                continue
            center = calculate_center_of_mass(data_orig, label)
            json_data.append({"label": int(label), "X": center[0], "Y": center[1], "Z": center[2]})

        json_data.sort(key=lambda x: x.get("label", 0))
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate centroid JSON files for vertebra segmentation masks')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to directory containing patient folders with segmentation masks')
    parser.add_argument('--min-voxels-top', type=int, default=8000,
                        help='Minimum voxels for topmost vertebra (default: 8000)')
    parser.add_argument('--min-voxels-bottom', type=int, default=6000,
                        help='Minimum voxels for bottommost vertebra (default: 6000)')
    parser.add_argument('--sample-test', type=int, default=None,
                        help='Sample test mode: process only N patient folders (for pipeline testing)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    process_directory(args.input_dir, sample_limit=args.sample_test)