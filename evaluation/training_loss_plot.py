# training_loss_plot.py - Visualize training losses
import matplotlib.pyplot as plt
import numpy as np
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
loss_file = os.path.join(base_dir, "verse19", "checkpoints", "healthivert_full", "loss_log.txt")

if not os.path.exists(loss_file):
    print(f"❌ Loss log not found: {loss_file}")
    exit(1)

# Load loss log (format: epoch, iteration, loss_G, loss_D, etc.)
print(f"Loading losses from: {loss_file}")
with open(loss_file, 'r') as f:
    lines = f.readlines()

# Parse losses (skip header if present)
iterations = []
loss_G_GAN = []
loss_G_L1 = []
loss_D = []

for line in lines:
    if line.startswith('epoch') or line.startswith('==='):
        continue
    parts = line.strip().split()
    if len(parts) >= 4:
        try:
            # Extract iteration and losses
            for part in parts:
                if 'iter' in part or 'iteration' in part:
                    iterations.append(int(part.split(':')[-1]))
                elif 'G_GAN' in part:
                    loss_G_GAN.append(float(part.split(':')[-1]))
                elif 'G_L1' in part:
                    loss_G_L1.append(float(part.split(':')[-1]))
                elif 'D_real' in part or 'D_fake' in part or 'D:' in part:
                    loss_D.append(float(part.split(':')[-1]))
        except (ValueError, IndexError):
            continue

# If parsing failed, try simple numeric parsing
if len(iterations) == 0:
    data = []
    for line in lines:
        try:
            nums = [float(x) for x in line.strip().split() if x.replace('.', '').replace('-', '').isdigit()]
            if len(nums) >= 3:
                data.append(nums)
        except:
            continue
    
    if data:
        data = np.array(data)
        iterations = list(range(len(data)))
        loss_G_GAN = data[:, 0] if data.shape[1] > 0 else []
        loss_G_L1 = data[:, 1] if data.shape[1] > 1 else []
        loss_D = data[:, 2] if data.shape[1] > 2 else []

print(f"Loaded {len(iterations)} iterations")

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('HealthiVert-GAN Training Losses (5 Epochs)', fontsize=16)

# Generator GAN Loss
if len(loss_G_GAN) > 0:
    axes[0].plot(iterations[:len(loss_G_GAN)], loss_G_GAN, color='blue', linewidth=1.5)
    axes[0].set_title('Generator Adversarial Loss (G_GAN)')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)

# Generator L1 Loss
if len(loss_G_L1) > 0:
    axes[1].plot(iterations[:len(loss_G_L1)], loss_G_L1, color='green', linewidth=1.5)
    axes[1].set_title('Generator L1 Loss (G_L1)')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)

# Discriminator Loss
if len(loss_D) > 0:
    axes[2].plot(iterations[:len(loss_D)], loss_D, color='red', linewidth=1.5)
    axes[2].set_title('Discriminator Loss (D)')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(base_dir, 'training_losses.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Saved training loss plot to {output_path}")

# Print summary statistics
if len(loss_G_GAN) > 0:
    print(f"\nLoss Statistics:")
    print(f"  G_GAN: {np.mean(loss_G_GAN):.4f} ± {np.std(loss_G_GAN):.4f}")
if len(loss_G_L1) > 0:
    print(f"  G_L1:  {np.mean(loss_G_L1):.4f} ± {np.std(loss_G_L1):.4f}")
if len(loss_D) > 0:
    print(f"  D:     {np.mean(loss_D):.4f} ± {np.std(loss_D):.4f}")
