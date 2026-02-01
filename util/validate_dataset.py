"""
Dataset Validation Script

This script validates all vertebrae in your dataset to identify which ones can be
successfully processed with your current slice extraction settings. It tests different
maxheight thresholds to help you decide whether to relax constraints or skip problematic samples.

Usage:
    python validate_dataset.py \
        --dataroot "/path/to/data" \
        --vertebra_json "/path/to/vertebra_data.json" \
        --cam_folder "/path/to/heatmap" \
        --phase train
"""

import os
import json
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import label
from tqdm import tqdm
from datetime import datetime


def remove_small_connected_components(input_array, min_size):
    """Remove small connected components from binary mask."""
    structure = np.ones((3, 3), dtype=np.int32)
    labeled, ncomponents = label(input_array, structure)
    
    for i in range(1, ncomponents + 1):
        if np.sum(labeled == i) < min_size:
            input_array[labeled == i] = 0
    
    return input_array


def get_weighted_random_slice(z0, z1):
    """Get a weighted random slice from the z range."""
    range_length = z1 - z0 + 1
    new_range_length = int(range_length * 4 / 5)
    
    new_z0 = z0 + (range_length - new_range_length) // 2
    new_z1 = new_z0 + new_range_length - 1
    
    center_index = (new_z0 + new_z1) // 2
    weights = [1 - abs(i - center_index) / (new_z1 - new_z0) for i in range(new_z0, new_z1 + 1)]
    
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    random_index = np.random.choice(range(new_z0, new_z1 + 1), p=normalized_weights)
    index_ratio = abs(random_index - center_index) / range_length * 2
    
    return random_index, index_ratio


def find_file_with_fallback(base_path):
    """Try to find a file with .nii.gz extension first, then .nii extension."""
    path_gz = base_path + '.nii.gz'
    if os.path.exists(path_gz):
        return path_gz
    
    path_nii = base_path + '.nii'
    if os.path.exists(path_nii):
        return path_nii
    
    return None


def test_vertebra_slice_extraction(vertebra_id, dataroot, maxheight=40, max_attempts=100):
    """
    Test if a vertebra can be successfully processed with the given maxheight.
    
    Returns:
        (success: bool, height: int or None, reason: str)
    """
    try:
        # Load data
        ct_base_path = os.path.join(dataroot, "CT", vertebra_id)
        label_base_path = os.path.join(dataroot, "label", vertebra_id)
        
        ct_path = find_file_with_fallback(ct_base_path)
        label_path = find_file_with_fallback(label_base_path)
        
        if not ct_path or not label_path:
            return False, None, f"Missing files (CT or label)"
        
        label_data = nib.load(label_path).get_fdata()
        patient_id, vert_id = vertebra_id.rsplit('_', 1)
        vert_id = int(vert_id)
        
        # Extract vertebra label
        vert_label = np.zeros_like(label_data)
        vert_label[label_data == vert_id] = 1
        
        # Find z-range
        loc = np.where(vert_label)
        if len(loc[2]) == 0:
            return False, None, f"No vertebra found in label (vert_id={vert_id})"
        
        z0 = min(loc[2])
        z1 = max(loc[2])
        
        # Try to find valid slice
        for attempt in range(max_attempts):
            slice_index, _ = get_weighted_random_slice(z0, z1)
            vert_label_copy = vert_label.copy()
            vert_label_copy[:, :, slice_index] = remove_small_connected_components(
                vert_label_copy[:, :, slice_index], 50
            )
            
            if np.sum(vert_label_copy[:, :, slice_index]) > 50:
                coords = np.argwhere(vert_label_copy[:, :, slice_index])
                x1, x2 = min(coords[:, 0]), max(coords[:, 0])
                height = x2 - x1
                
                if height < maxheight:
                    return True, height, f"Success (height={height})"
        
        # If we reach here, couldn't find valid slice
        return False, None, f"No valid slice found after {max_attempts} attempts (vertebra too fragmented or too large)"
        
    except Exception as e:
        return False, None, f"Error: {str(e)}"


def validate_dataset(dataroot, vertebra_json_path, phase="train", maxheight_list=[40, 50, 60]):
    """Validate all vertebrae in the dataset."""
    
    print(f"\n{'='*80}")
    print(f"Dataset Validation - Phase: {phase}")
    print(f"{'='*80}\n")
    
    # Load vertebra list
    with open(vertebra_json_path, 'r') as f:
        vertebra_set = json.load(f)
    
    vertebra_list = list(vertebra_set[phase].keys())
    print(f"Total vertebrae to validate: {len(vertebra_list)}\n")
    
    # Test with each maxheight value
    results_by_maxheight = {}
    
    for maxheight in maxheight_list:
        print(f"\nTesting with maxheight = {maxheight}:")
        print("-" * 80)
        
        valid_vertebrae = []
        invalid_vertebrae = []
        heights = []
        failure_reasons = {}
        
        for vertebra_id in tqdm(vertebra_list, desc=f"maxheight={maxheight}"):
            success, height, reason = test_vertebra_slice_extraction(
                vertebra_id, dataroot, maxheight=maxheight
            )
            
            if success:
                valid_vertebrae.append(vertebra_id)
                heights.append(height)
            else:
                invalid_vertebrae.append(vertebra_id)
                reason_key = reason.split("(")[0].strip() if "(" in reason else reason
                failure_reasons[reason_key] = failure_reasons.get(reason_key, 0) + 1
        
        # Calculate statistics
        valid_count = len(valid_vertebrae)
        invalid_count = len(invalid_vertebrae)
        valid_ratio = 100.0 * valid_count / len(vertebra_list)
        
        stats = {
            'maxheight': maxheight,
            'total': len(vertebra_list),
            'valid': valid_count,
            'invalid': invalid_count,
            'valid_ratio': valid_ratio,
            'valid_vertebrae': valid_vertebrae,
            'invalid_vertebrae': invalid_vertebrae,
            'failure_reasons': failure_reasons,
            'heights_stats': {
                'mean': np.mean(heights) if heights else 0,
                'median': np.median(heights) if heights else 0,
                'min': np.min(heights) if heights else 0,
                'max': np.max(heights) if heights else 0,
                'std': np.std(heights) if heights else 0,
            }
        }
        
        results_by_maxheight[maxheight] = stats
        
        # Print statistics
        print(f"\n✅ Valid vertebrae:   {valid_count:4d} ({valid_ratio:6.2f}%)")
        print(f"❌ Invalid vertebrae: {invalid_count:4d} ({100-valid_ratio:6.2f}%)")
        print(f"\nHeight Statistics (for valid vertebrae):")
        print(f"  Mean:   {stats['heights_stats']['mean']:.2f} pixels")
        print(f"  Median: {stats['heights_stats']['median']:.2f} pixels")
        print(f"  Min:    {stats['heights_stats']['min']:.2f} pixels")
        print(f"  Max:    {stats['heights_stats']['max']:.2f} pixels")
        print(f"  Std:    {stats['heights_stats']['std']:.2f} pixels")
        
        if failure_reasons:
            print(f"\nFailure Reasons:")
            for reason, count in sorted(failure_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {reason}: {count}")
    
    return results_by_maxheight


def save_report(results_by_maxheight, output_dir, vertebra_json_path):
    """Save validation report to file."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"validation_report_{timestamp}.txt")
    skip_list_path = os.path.join(output_dir, f"skip_list_{timestamp}.json")
    
    # Write detailed report
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DATASET VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Vertebra JSON: {vertebra_json_path}\n\n")
        
        for maxheight, stats in sorted(results_by_maxheight.items()):
            f.write(f"\n{'='*80}\n")
            f.write(f"THRESHOLD: maxheight = {maxheight}\n")
            f.write(f"{'='*80}\n\n")
            
            f.write(f"Valid vertebrae:   {stats['valid']:4d} / {stats['total']} ({stats['valid_ratio']:6.2f}%)\n")
            f.write(f"Invalid vertebrae: {stats['invalid']:4d} / {stats['total']} ({100-stats['valid_ratio']:6.2f}%)\n\n")
            
            f.write("Height Statistics (for valid vertebrae):\n")
            f.write(f"  Mean:   {stats['heights_stats']['mean']:.2f} pixels\n")
            f.write(f"  Median: {stats['heights_stats']['median']:.2f} pixels\n")
            f.write(f"  Min:    {stats['heights_stats']['min']:.2f} pixels\n")
            f.write(f"  Max:    {stats['heights_stats']['max']:.2f} pixels\n")
            f.write(f"  Std:    {stats['heights_stats']['std']:.2f} pixels\n\n")
            
            if stats['failure_reasons']:
                f.write("Failure Reasons:\n")
                for reason, count in sorted(stats['failure_reasons'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  - {reason}: {count}\n")
            
            f.write(f"\n{'-'*80}\n")
            f.write(f"Invalid Vertebrae (maxheight={maxheight}):\n")
            f.write(f"{'-'*80}\n")
            for vert_id in sorted(stats['invalid_vertebrae']):
                f.write(f"  {vert_id}\n")
    
    # Write skip list for strictest threshold
    strictest_maxheight = min(results_by_maxheight.keys())
    skip_list = {
        'maxheight': strictest_maxheight,
        'vertebrae_to_skip': results_by_maxheight[strictest_maxheight]['invalid_vertebrae'],
        'count': len(results_by_maxheight[strictest_maxheight]['invalid_vertebrae']),
        'timestamp': timestamp
    }
    
    with open(skip_list_path, 'w') as f:
        json.dump(skip_list, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ Report saved to: {report_path}")
    print(f"✅ Skip list saved to: {skip_list_path}")
    print(f"{'='*80}\n")
    
    return report_path, skip_list_path


def main():
    parser = argparse.ArgumentParser(
        description="Validate dataset vertebrae slice extraction"
    )
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--vertebra_json', type=str, required=True,
                        help='Path to vertebra_data.json')
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'test', 'val'],
                        help='Dataset phase to validate')
    parser.add_argument('--maxheight_list', type=int, nargs='+', default=[40, 50, 60],
                        help='List of maxheight thresholds to test')
    parser.add_argument('--output_dir', type=str, default='./validation_reports',
                        help='Directory to save validation reports')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.dataroot):
        print(f"❌ Error: dataroot not found: {args.dataroot}")
        return
    
    if not os.path.exists(args.vertebra_json):
        print(f"❌ Error: vertebra_json not found: {args.vertebra_json}")
        return
    
    # Run validation
    results = validate_dataset(
        args.dataroot,
        args.vertebra_json,
        phase=args.phase,
        maxheight_list=sorted(args.maxheight_list)
    )
    
    # Save report
    report_path, skip_list_path = save_report(
        results,
        args.output_dir,
        args.vertebra_json
    )
    
    print("\n📊 SUMMARY:")
    print(f"Phase: {args.phase}")
    print(f"Total vertebrae: {results[sorted(results.keys())[0]]['total']}")
    
    for maxheight in sorted(results.keys()):
        stats = results[maxheight]
        print(f"\nmaxheight={maxheight}:")
        print(f"  Valid:   {stats['valid']} ({stats['valid_ratio']:.2f}%)")
        print(f"  Invalid: {stats['invalid']} ({100-stats['valid_ratio']:.2f}%)")


if __name__ == '__main__':
    main()
