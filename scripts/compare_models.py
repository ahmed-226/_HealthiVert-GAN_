"""
HealthiVert Model Comparison Script (Python version for Windows/Kaggle)
Compares GAN vs Diffusion model performance on validation set

Usage:
  python compare_models.py \
    --gan-checkpoint ./checkpoints/gan/latest_net_G.pth \
    --diffusion-checkpoint ./checkpoints/diffusion/latest_net_G.pth \
    --ct-folder ./datasets/straightened/CT \
    --cam-folder ./Attention/heatmap \
    --label-folder ./datasets/straightened/label \
    --json-path ./vertebra_data.json \
    --gpu 0
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare GAN vs Diffusion models on HealthiVert dataset'
    )
    
    # Checkpoints
    parser.add_argument('--gan-checkpoint', 
                       default='./checkpoints/healthivert_gan/latest_net_G.pth',
                       help='Path to GAN model checkpoint')
    parser.add_argument('--diffusion-checkpoint',
                       default='./checkpoints/healthivert_diffusion/latest_net_G.pth',
                       help='Path to Diffusion model checkpoint')
    
    # Data paths
    parser.add_argument('--ct-folder',
                       default='./datasets/straightened/CT',
                       help='Path to CT images folder')
    parser.add_argument('--cam-folder',
                       default='./Attention/heatmap',
                       help='Path to CAM heatmaps folder')
    parser.add_argument('--label-folder',
                       default='./datasets/straightened/label',
                       help='Path to ground truth labels folder')
    parser.add_argument('--json-path',
                       default='./vertebra_data.json',
                       help='Path to vertebra data JSON file')
    
    # GPU settings
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--ngf-gan', type=int, default=16,
                       help='Number of generator filters for GAN')
    parser.add_argument('--ngf-diffusion', type=int, default=64,
                       help='Number of generator filters for Diffusion')
    
    # Output settings
    parser.add_argument('--output-dir',
                       default='./results',
                       help='Base output directory for results')
    parser.add_argument('--skip-gan', action='store_true',
                       help='Skip GAN generation')
    parser.add_argument('--skip-diffusion', action='store_true',
                       help='Skip Diffusion generation')
    parser.add_argument('--skip-metrics', action='store_true',
                       help='Skip metric evaluation')
    
    return parser.parse_args()

# ============================================================================
# Main Script
# ============================================================================

def run_command(cmd):
    """Run a command and print output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True

def main():
    args = parse_args()
    
    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = Path(args.output_dir) / f'comparison_{timestamp}'
    gan_output = output_base / 'gan'
    diffusion_output = output_base / 'diffusion'
    metrics_output = output_base / 'metrics'
    
    gan_output.mkdir(parents=True, exist_ok=True)
    diffusion_output.mkdir(parents=True, exist_ok=True)
    metrics_output.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("HealthiVert Model Comparison")
    print("=" * 70)
    print()
    print(f"📁 Output directory: {output_base}")
    print()
    
    # Step 1: Generate with GAN
    if not args.skip_gan:
        print("=" * 70)
        print("Step 1/4: Generating with GAN model")
        print("=" * 70)
        print()
        
        if Path(args.gan_checkpoint).exists():
            cmd = [
                sys.executable, 'eval_3d_sagittal_twostage.py',
                '--model-path', args.gan_checkpoint,
                '--ct-folder', args.ct_folder,
                '--cam-folder', args.cam_folder,
                '--output-folder', str(gan_output),
                '--model-type', 'gan',
                '--gpu', str(args.gpu),
                '--ngf', str(args.ngf_gan)
            ]
            if run_command(cmd):
                print("✅ GAN generation complete")
        else:
            print(f"⚠️  GAN checkpoint not found: {args.gan_checkpoint}")
            print("   Skipping GAN generation...")
    else:
        print("⏭️  Skipping GAN generation (--skip-gan flag)")
    
    # Step 2: Generate with Diffusion
    if not args.skip_diffusion:
        print()
        print("=" * 70)
        print("Step 2/4: Generating with Diffusion model")
        print("=" * 70)
        print()
        
        if Path(args.diffusion_checkpoint).exists():
            cmd = [
                sys.executable, 'eval_3d_sagittal_twostage.py',
                '--model-path', args.diffusion_checkpoint,
                '--ct-folder', args.ct_folder,
                '--cam-folder', args.cam_folder,
                '--output-folder', str(diffusion_output),
                '--model-type', 'diffusion',
                '--gpu', str(args.gpu),
                '--ngf', str(args.ngf_diffusion)
            ]
            if run_command(cmd):
                print("✅ Diffusion generation complete")
        else:
            print(f"⚠️  Diffusion checkpoint not found: {args.diffusion_checkpoint}")
            print("   Skipping diffusion generation...")
    else:
        print("⏭️  Skipping Diffusion generation (--skip-diffusion flag)")
    
    # Step 3: Evaluate metrics
    if not args.skip_metrics:
        print()
        print("=" * 70)
        print("Step 3/4: Evaluating generation metrics")
        print("=" * 70)
        print()
        
        # GAN metrics
        if (gan_output / 'CT_fake').exists():
            print("Evaluating GAN...")
            cmd = [
                sys.executable, 'evaluation/generation_eval_sagittal.py',
                '--ori-ct-folder', args.ct_folder,
                '--ori-seg-folder', args.label_folder,
                '--fake-ct-folder', str(gan_output / 'CT_fake'),
                '--fake-seg-folder', str(gan_output / 'label_fake'),
                '--json-path', args.json_path,
                '--save-folder', str(metrics_output / 'gan')
            ]
            if run_command(cmd):
                print("✅ GAN metrics saved")
        
        # Diffusion metrics
        if (diffusion_output / 'CT_fake').exists():
            print("Evaluating Diffusion...")
            cmd = [
                sys.executable, 'evaluation/generation_eval_sagittal.py',
                '--ori-ct-folder', args.ct_folder,
                '--ori-seg-folder', args.label_folder,
                '--fake-ct-folder', str(diffusion_output / 'CT_fake'),
                '--fake-seg-folder', str(diffusion_output / 'label_fake'),
                '--json-path', args.json_path,
                '--save-folder', str(metrics_output / 'diffusion')
            ]
            if run_command(cmd):
                print("✅ Diffusion metrics saved")
    else:
        print("⏭️  Skipping metric evaluation (--skip-metrics flag)")
    
    # Step 4: Generate comparison report
    print()
    print("=" * 70)
    print("Step 4/4: Generating comparison report")
    print("=" * 70)
    print()
    
    generate_comparison_report(metrics_output, output_base)
    
    # Final summary
    print()
    print("=" * 70)
    print("✅ Comparison Complete!")
    print("=" * 70)
    print()
    print(f"📊 Results saved to: {output_base}")
    print()
    print("Next steps:")
    print("  1. Review comparison_report.txt")
    print("  2. Check visual quality in CT_fake/ folders")
    print("  3. Run RHLV quantification:")
    print("     python evaluation/RHLV_quantification.py --input-folder <output>")
    print("  4. Run SVM grading:")
    print("     python evaluation/SVM_grading.py --input-folder <output>")
    print()

def generate_comparison_report(metrics_output, output_base):
    """Generate comparison report from metrics."""
    import pandas as pd
    import numpy as np
    
    def load_metrics(model_name):
        """Load metrics from CSV file."""
        csv_path = metrics_output / model_name / 'generation_metrics.csv'
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return None
    
    # Load metrics
    gan_df = load_metrics('gan')
    diff_df = load_metrics('diffusion')
    
    # Generate report
    report = []
    report.append("=" * 70)
    report.append("HealthiVert Model Comparison Report")
    report.append("=" * 70)
    report.append("")
    
    if gan_df is not None and diff_df is not None:
        # Compute average metrics
        metrics = ['PSNR', 'SSIM', 'Dice', 'IoU', 'RHDR']
        
        report.append("Average Metrics:")
        report.append("-" * 70)
        report.append(f"{'Metric':<15} {'GAN':<15} {'Diffusion':<15} {'Diff (%)':<15}")
        report.append("-" * 70)
        
        for metric in metrics:
            if metric in gan_df.columns and metric in diff_df.columns:
                gan_mean = gan_df[metric].mean()
                diff_mean = diff_df[metric].mean()
                percent_diff = ((diff_mean - gan_mean) / gan_mean) * 100
                
                report.append(f"{metric:<15} {gan_mean:<15.4f} {diff_mean:<15.4f} {percent_diff:+.2f}%")
        
        report.append("-" * 70)
        report.append("")
        
        # Statistical comparison
        report.append("Statistical Summary:")
        report.append("-" * 70)
        
        try:
            from scipy import stats
            
            for metric in metrics:
                if metric in gan_df.columns and metric in diff_df.columns:
                    gan_vals = gan_df[metric].values
                    diff_vals = diff_df[metric].values
                    
                    # Paired t-test (assuming same samples)
                    if len(gan_vals) == len(diff_vals):
                        t_stat, p_val = stats.ttest_rel(gan_vals, diff_vals)
                        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        report.append(f"{metric}: t={t_stat:.3f}, p={p_val:.4f} {significance}")
            
            report.append("")
            report.append("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns=not significant")
        except ImportError:
            report.append("scipy not available, skipping statistical tests")
        
        report.append("-" * 70)
    else:
        if gan_df is None:
            report.append("⚠️  GAN metrics not found")
        if diff_df is None:
            report.append("⚠️  Diffusion metrics not found")
    
    report.append("")
    report.append("Output locations:")
    report.append(f"  GAN outputs: {output_base / 'gan'}")
    report.append(f"  Diffusion outputs: {output_base / 'diffusion'}")
    report.append(f"  Metrics: {metrics_output}")
    report.append("")
    report.append("=" * 70)
    
    # Print and save report
    report_text = "\n".join(report)
    print(report_text)
    
    report_path = output_base / 'comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"\n📊 Report saved to: {report_path}")

if __name__ == '__main__':
    main()


# python compare_models.py \
#   --gan-checkpoint /kaggle/input/models/gan_latest.pth \
#   --diffusion-checkpoint /kaggle/input/models/diffusion_latest.pth \
#   --ct-folder /kaggle/input/data/CT \
#   --label-folder /kaggle/input/data/label \
#   --gpu 0