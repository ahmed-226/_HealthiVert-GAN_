# comparison_plot.py - Auto-discovers samples and creates visualization
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob

# Auto-detect results directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, "verse19", "results", "healthivert_full", "train_5", "images")

# Auto-discover available samples by finding all *_fake_B.png files
fake_B_files = glob.glob(os.path.join(results_dir, "*_fake_B.png"))
if not fake_B_files:
    print(f"❌ No results found in: {results_dir}")
    print("Available directories:")
    results_base = os.path.join(base_dir, "verse19", "results")
    if os.path.exists(results_base):
        for root, dirs, files in os.walk(results_base):
            if 'images' in dirs:
                print(f"  - {os.path.join(root, 'images')}")
    exit(1)

# Extract sample IDs (remove _fake_B.png suffix)
all_samples = [os.path.basename(f).replace('_fake_B.png', '') for f in fake_B_files]
print(f"Found {len(all_samples)} generated vertebrae")

# Select 3 samples for visualization (first, middle, last)
if len(all_samples) >= 3:
    samples = [all_samples[0], all_samples[len(all_samples)//2], all_samples[-1]]
else:
    samples = all_samples[:3]  # Use whatever is available

print(f"Selected samples: {samples}")

fig, axes = plt.subplots(len(samples), 4, figsize=(16, 4*len(samples)))
if len(samples) == 1:
    axes = axes.reshape(1, -1)  # Ensure 2D array
fig.suptitle('HealthiVert-GAN: Vertebra Reconstruction Results', fontsize=16)

for i, sample in enumerate(samples):
    # Input (fractured)
    input_path = os.path.join(results_dir, f'{sample}_real_A.png')
    if os.path.exists(input_path):
        axes[i, 0].imshow(Image.open(input_path), cmap='gray')
    axes[i, 0].set_title('Input (Fractured)')
    axes[i, 0].axis('off')
    
    # Generated (healthy)
    gen_path = os.path.join(results_dir, f'{sample}_fake_B.png')
    if os.path.exists(gen_path):
        axes[i, 1].imshow(Image.open(gen_path), cmap='gray')
    axes[i, 1].set_title('Generated (Healthy)')
    axes[i, 1].axis('off')
    
    # Attention Heatmap
    cam_path = os.path.join(results_dir, f'{sample}_CAM.png')
    if os.path.exists(cam_path):
        axes[i, 2].imshow(Image.open(cam_path), cmap='hot')
    axes[i, 2].set_title('Attention Heatmap')
    axes[i, 2].axis('off')
    
    # Predicted Mask
    mask_path = os.path.join(results_dir, f'{sample}_fake_B_mask_raw.png')
    if os.path.exists(mask_path):
        axes[i, 3].imshow(Image.open(mask_path), cmap='gray')
    axes[i, 3].set_title('Predicted Mask')
    axes[i, 3].axis('off')

plt.tight_layout()
output_path = os.path.join(base_dir, 'results_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Saved comparison figure to {output_path}")