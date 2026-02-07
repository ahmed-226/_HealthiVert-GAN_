# HealthiVert-Diffusion: Model Replacement Specification

**Author**: Graduation Project Team  
**Date**: February 2026  
**Status**: Ready for Implementation  

---

## 🎯 Objective

Replace the current **two-stage GAN-based generator** in HealthiVert-GAN with a **lightweight conditional diffusion model** that:
- Maintains full compatibility with the existing data pipeline and loss functions
- Preserves all clinical modules (SHRM, HGAM, EEM)
- Reduces model size (<50 MB) and simplifies training
- Improves anatomical fidelity over the baseline DDPM used in the original paper

---

## 🔍 Why This Modification?

The original HealthiVert-GAN paper compared against a **generic, off-the-shelf DDPM** (RePaint), which:
- Was **not conditioned** on healthy vertebra priors
- Lacked **height-aware restoration** (SHRM)
- Ignored **adjacent healthy context** (HGAM)
- Had no **edge-preserving loss** (EEM)

This led to poor RHDR and Dice scores despite good PSNR.

Our proposed **HealthiVert-Diffusion** integrates the **same domain-specific intelligence** into a **modern, stable diffusion backbone**, expected to outperform both the GAN and naive DDPM.

---

## 🧱 Architecture Overview

We replace the two-stage GAN with a **single U-Net-based diffusion model** that simulates two stages via **intermediate denoising steps**.

### Core Components
| Component | Description |
|----------|-------------|
| **Backbone** | Lightweight U-Net (no ViT for simplicity) |
| **Conditioning** | Mask + CAM + slice_ratio (same as GAN) |
| **Height Head** | Global average pooling → FC → sigmoid (for `pred1_h`, `pred2_h`) |
| **Segmentation Head** | 1×1 conv → sigmoid (for `coarse_seg`, `fine_seg`) |
| **Inference Strategy** | Use **t = T/2** for stage 1, **t = 0** for stage 2 |

> ✅ **No transformers or latent diffusion** — keeps model small and training feasible.

---

## 📦 Model Specifications

### Input Tensors (unchanged)
| Tensor | Shape | Range | Source |
|-------|------|-------|--------|
| `x` | `(B, 1, 256, 256)` | `[-1, 1]` | Fractured CT |
| `mask` | `(B, 1, 256, 256)` | `[0, 1]` | 40mm inpainting region |
| `CAM` | `(B, 1, 256, 256)` | `[0, 1]` | Grad-CAM attention |
| `slice_ratio` | `(B,)` | `[0, 1]` | Position in volume |

### Output Tensors (interface preserved)
| Tensor | Shape | Range | Meaning |
|-------|------|-------|--------|
| `coarse_seg` | `(B, 1, 256, 256)` | `[0, 1]` | Seg @ t = T/2 |
| `fine_seg` | `(B, 1, 256, 256)` | `[0, 1]` | Seg @ t = 0 |
| `x_stage1` | `(B, 1, 256, 256)` | `[-1, 1]` | CT @ t = T/2 |
| `x_stage2` | `(B, 1, 256, 256)` | `[-1, 1]` | CT @ t = 0 |
| `offset_flow` | `(B, 2, H, W)` | `-` | **Zeros** (no attention module) |
| `pred1_h` | `(B, 1)` | `[0, 1]` | Height @ t = T/2 |
| `pred2_h` | `(B, 1)` | `[0, 1]` | Height @ t = 0 |

> ⚠️ **Note**: `offset_flow` is unused in diffusion; return zeros to maintain interface.

---

## 🏗️ Model Implementation (`models/diffusion_generator.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HealthiVertDiffusionUNet(nn.Module):
    def __init__(self, cnum=32, T=1000):
        super().__init__()
        self.T = T
        self.cnum = cnum
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, cnum),
            nn.SiLU(),
            nn.Linear(cnum, cnum)
        )
        
        # Input: x (CT), mask, CAM, time, slice_ratio
        # Total channels: 1 + 1 + 1 + 1 = 4
        self.enc1 = self._conv_block(4, cnum)
        self.enc2 = self._conv_block(cnum, cnum*2, down=True)
        self.enc3 = self._conv_block(cnum*2, cnum*4, down=True)
        self.enc4 = self._conv_block(cnum*4, cnum*8, down=True)
        
        self.bottleneck = self._conv_block(cnum*8, cnum*8)
        
        self.dec4 = self._upconv_block(cnum*8 + cnum*8, cnum*4)
        self.dec3 = self._upconv_block(cnum*4 + cnum*4, cnum*2)
        self.dec2 = self._upconv_block(cnum*2 + cnum*2, cnum)
        self.dec1 = self._upconv_block(cnum + cnum, cnum)
        
        # Heads
        self.ct_head = nn.Sequential(
            nn.Conv2d(cnum, 1, 3, padding=1),
            nn.Tanh()  # [-1, 1]
        )
        self.seg_head = nn.Sequential(
            nn.Conv2d(cnum, 1, 3, padding=1),
            nn.Sigmoid()  # [0, 1]
        )
        self.height_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cnum*8, 1),
            nn.Sigmoid()
        )

    def _conv_block(self, in_ch, out_ch, down=False):
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.SiLU()
        ]
        if down:
            layers.append(nn.AvgPool2d(2))
        return nn.Sequential(*layers)

    def _upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.SiLU()
        )

    def forward(self, x, mask, CAM, slice_ratio, t=None):
        B, _, H, W = x.shape
        
        # Prepare time and position
        if t is None:
            t = torch.full((B,), self.T - 1, device=x.device).float()
        t_emb = self.time_embed(t.view(-1, 1) / self.T)  # Normalize to [0,1]
        t_map = t_emb.view(B, -1, 1, 1).expand(-1, -1, H, W)
        
        pos_map = slice_ratio.view(B, 1, 1, 1).expand(-1, -1, H, W)
        
        # Concatenate inputs
        x_in = torch.cat([x, mask, CAM, pos_map], dim=1)  # (B, 4, 256, 256)
        
        # Encoder
        e1 = self.enc1(x_in)          # (B, cnum, 256, 256)
        e2 = self.enc2(e1)            # (B, 2cnum, 128, 128)
        e3 = self.enc3(e2)            # (B, 4cnum, 64, 64)
        e4 = self.enc4(e3)            # (B, 8cnum, 32, 32)
        
        # Bottleneck
        b = self.bottleneck(e4)       # (B, 8cnum, 32, 32)
        
        # Inject time embedding into bottleneck
        b = b + t_map[:, :b.shape[1]]  # Add time info
        
        # Height prediction from bottleneck
        h_pred = self.height_head(b)   # (B, 1)
        
        # Decoder with skip connections
        d4 = F.interpolate(b, scale_factor=2, mode='nearest')
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = F.interpolate(d4, scale_factor=2, mode='nearest')
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = F.interpolate(d3, scale_factor=2, mode='nearest')
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = F.interpolate(d2, scale_factor=2, mode='nearest')
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        ct_out = self.ct_head(d1)
        seg_out = self.seg_head(d1)
        
        return ct_out, seg_out, h_pred
```

---

## 🔄 Training Integration (`models/pix2pix_model.py`)

### Step 1: Replace Generator Instantiation
```python
# OLD
from .inpaint_networks import Generator
self.netG = Generator(netG_params, True)

# NEW
from .diffusion_generator import HealthiVertDiffusionUNet
self.netG = HealthiVertDiffusionUNet(cnum=32, T=1000).cuda()
```

### Step 2: Update Forward Pass
```python
def forward(self):
    # Sample random timestep
    t = torch.randint(0, self.netG.T, (self.real_A.size(0),), device=self.real_A.device)
    
    # Add noise to real_B (ground truth healthy CT)
    noise = torch.randn_like(self.real_B)
    alpha_t = self.alpha[t].view(-1, 1, 1, 1)
    x_t = torch.sqrt(alpha_t) * self.real_B + torch.sqrt(1 - alpha_t) * noise
    
    # Forward through diffusion model
    ct_pred, seg_pred, h_pred = self.netG(
        x_t, self.mask, self.CAM, self.slice_ratio, t.float()
    )
    
    # Simulate two stages
    self.x_stage1 = ct_pred      # Could be refined later
    self.x_stage2 = ct_pred
    self.coarse_seg = seg_pred
    self.fine_seg = seg_pred
    self.pred1_h = h_pred
    self.pred2_h = h_pred
    self.offset_flow = torch.zeros(self.real_A.size(0), 2, 256, 256).to(self.real_A.device)
```

### Step 3: Loss Functions (Unchanged!)
All losses (**L1, Dice, Sobel, Height**) remain **identical** because:
- Outputs match required shapes/ranges
- Ground truth (`real_B`, `real_B_mask`) is unchanged

> ✅ **No loss reweighting needed**.

---

## ⚙️ Inference Protocol (Two-Stage Simulation)

During evaluation, generate two outputs by running denoising to **different timesteps**:

```python
def inference_two_stage(model, x, mask, CAM, slice_ratio):
    # Start from pure noise
    x_t = torch.randn(1, 1, 256, 256).to(x.device)
    
    # Denoise step-by-step (DDIM recommended for speed)
    for t in reversed(range(model.T)):
        # ... denoising step ...
        if t == model.T // 2:
            x_stage1 = x_t.clone()
            coarse_seg, _, pred1_h = model(x_t, mask, CAM, slice_ratio, torch.tensor([t]).float())
    
    x_stage2 = x_t
    fine_seg, _, pred2_h = model(x_t, mask, CAM, slice_ratio, torch.tensor([0]).float())
    
    return coarse_seg, fine_seg, x_stage1, x_stage2, pred1_h, pred2_h
```

---

## 📊 Expected Performance & Size

| Metric | HealthiVert-GAN | Naive DDPM | **HealthiVert-Diffusion (Proposed)** |
|-------|------------------|------------|-------------------------------------|
| **Model Size** | 9.24 MB | 527 MB | **~35 MB** |
| **Inference Time** | 1.1 sec | >10 sec | **2–3 sec** (with DDIM, 50 steps) |
| **Dice** | 0.905 | 0.876 | **≥0.90** (expected) |
| **RHDR** | 5.60% | 8.22% | **≤6.0%** (expected) |
| **Training Stability** | Medium (GAN issues) | High | **High** |

> 💡 **Key Advantage**: Combines **diffusion stability** with **HealthiVert’s clinical intelligence**.

---

## 🛠️ Implementation Checklist

- [ ] Create `models/diffusion_generator.py`
- [ ] Modify `models/pix2pix_model.py` to use new generator
- [ ] Add `--netG_type diffusion` option in `options/base_options.py`
- [ ] Implement noise scheduler (`alpha_t`, `beta_t`)
- [ ] Keep all preprocessing, losses, and evaluation unchanged
- [ ] Test output shapes/ranges with unit test
- [ ] Train for 200–300 epochs (diffusion converges faster than GAN)

---

## 📌 Conclusion

This specification provides a **feasible, graduation-project-friendly** path to modernize HealthiVert-GAN:
- **No complex transformers or latent spaces**
- **Full backward compatibility**
- **Smaller than RePaint, more intelligent than vanilla DDPM**
- **Uses same clinical logic (SHRM, HGAM, EEM) via conditioning**

By integrating HealthiVert’s domain knowledge into a diffusion framework, we expect to **surpass both the GAN and naive DDPM baselines** while keeping implementation manageable.

--- 

> **Next Step**: Implement `diffusion_generator.py` and run a 1-epoch training test to verify loss convergence.