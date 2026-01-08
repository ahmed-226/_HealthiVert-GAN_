# evaluation_metrics.py
import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from glob import glob

results_dir = "/content/results/healthivert_full/train_5/images"

# Get all generated images (fake_B)
fake_B_files = sorted(glob(os.path.join(results_dir, "*_fake_B.png")))

print(f"Found {len(fake_B_files)} generated images")

ssim_scores = []
psnr_scores = []
dice_scores = []

for fake_B_path in fake_B_files:
    # Get corresponding real_A and real_B paths
    base_name = fake_B_path.replace("_fake_B.png", "")
    real_A_path = base_name + "_real_A.png"
    real_B_path = base_name + "_real_B.png"
    
    # Load images
    fake_B = np.array(Image.open(fake_B_path).convert('L'))  # Grayscale
    real_A = np.array(Image.open(real_A_path).convert('L'))
    
    # Check if ground truth exists
    if os.path.exists(real_B_path):
        real_B = np.array(Image.open(real_B_path).convert('L'))
        
        # Calculate SSIM (generated vs ground truth)
        ssim_val = ssim(fake_B, real_B, data_range=255)
        ssim_scores.append(ssim_val)
        
        # Calculate PSNR (generated vs ground truth)
        psnr_val = psnr(fake_B, real_B, data_range=255)
        psnr_scores.append(psnr_val)
    
    # Calculate improvement over input (always available)
    ssim_improvement = ssim(fake_B, real_A, data_range=255)
    
    print(f"Processed: {os.path.basename(fake_B_path)}")

# Print results
print("\n" + "="*50)
print("📊 EVALUATION RESULTS")
print("="*50)

if ssim_scores:
    print(f"\n✅ Comparison with Ground Truth:")
    print(f"   Average SSIM: {np.mean(ssim_scores):.4f} ± {np.std(ssim_scores):.4f}")
    print(f"   Average PSNR: {np.mean(psnr_scores):.2f} ± {np.std(psnr_scores):.2f} dB")
else:
    print(f"\n⚠️  No ground truth found - cannot calculate accuracy metrics")
    print(f"   (This is normal if you only have fractured vertebrae)")

print(f"\n📈 Total images processed: {len(fake_B_files)}")