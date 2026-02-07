"""
Diffusion-based generator for HealthiVert.
Lightweight U-Net with time embeddings and conditional inputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    """Sinusoidal time embeddings for diffusion timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        """
        Args:
            t: Timesteps, shape (B,)
        Returns:
            Time embeddings, shape (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    """Basic convolutional block with normalization and activation."""
    
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, out_ch)
        
        # Time embedding projection
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_ch)
            )
        else:
            self.time_mlp = None
            
    def forward(self, x, t_emb=None):
        h = F.silu(self.bn1(self.conv1(x)))
        
        # Add time embedding
        if self.time_mlp is not None and t_emb is not None:
            time_emb = self.time_mlp(t_emb)
            h = h + time_emb[:, :, None, None]
        
        h = F.silu(self.bn2(self.conv2(h)))
        return h


class DownBlock(nn.Module):
    """Downsampling block."""
    
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, time_emb_dim)
        self.pool = nn.AvgPool2d(2)
        
    def forward(self, x, t_emb):
        h = self.conv(x, t_emb)
        return self.pool(h), h  # Return pooled and skip connection


class UpBlock(nn.Module):
    """Upsampling block with skip connections."""
    
    def __init__(self, in_ch, out_ch, skip_ch, time_emb_dim):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, time_emb_dim)  # in_ch + skip_ch for concatenation
        
    def forward(self, x, skip, t_emb):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x, t_emb)


class HealthiVertDiffusionUNet(nn.Module):
    """
    Diffusion U-Net generator for HealthiVert.
    
    Inputs:
        - x_t: Noisy CT image (B, 1, 256, 256)
        - mask: Inpainting region (B, 1, 256, 256)
        - CAM: Grad-CAM attention (B, 1, 256, 256)
        - slice_ratio: Position in volume (B,)
        - t: Diffusion timestep (B,)
    
    Outputs (matching GAN interface):
        - coarse_seg: Stage 1 segmentation (B, 1, 256, 256), [0, 1]
        - fine_seg: Stage 2 segmentation (B, 1, 256, 256), [0, 1]
        - x_stage1: Stage 1 CT output (B, 1, 256, 256), [-1, 1]
        - x_stage2: Stage 2 CT output (B, 1, 256, 256), [-1, 1]
        - offset_flow: Zeros (B, 2, 256, 256) - unused, for interface compatibility
        - pred1_h: Stage 1 height (1, B), [0, 1]
        - pred2_h: Stage 2 height (1, B), [0, 1]
    """
    
    def __init__(self, cnum=32, T=1000):
        super().__init__()
        self.cnum = cnum
        self.T = T
        
        # Time embedding
        time_emb_dim = cnum * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(cnum),
            nn.Linear(cnum, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Input: x_t (1) + mask (1) + CAM (1) + slice_ratio (1) = 4 channels
        self.input_conv = ConvBlock(4, cnum, time_emb_dim)
        
        # Encoder
        self.down1 = DownBlock(cnum, cnum * 2, time_emb_dim)      # 256 -> 128
        self.down2 = DownBlock(cnum * 2, cnum * 4, time_emb_dim)   # 128 -> 64
        self.down3 = DownBlock(cnum * 4, cnum * 8, time_emb_dim)   # 64 -> 32
        
        # Bottleneck
        self.bottleneck = ConvBlock(cnum * 8, cnum * 8, time_emb_dim)
        
        # Height prediction head (from bottleneck features)
        self.height_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cnum * 8, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Decoder
        self.up3 = UpBlock(cnum * 8, cnum * 4, cnum * 8, time_emb_dim)   # 32 -> 64, skip from encoder has cnum*8 channels
        self.up2 = UpBlock(cnum * 4, cnum * 2, cnum * 4, time_emb_dim)   # 64 -> 128, skip from encoder has cnum*4 channels
        self.up1 = UpBlock(cnum * 2, cnum, cnum * 2, time_emb_dim)       # 128 -> 256, skip from encoder has cnum*2 channels
        
        # Output heads
        self.noise_head = nn.Sequential(
            nn.Conv2d(cnum, cnum, 3, padding=1),
            nn.GroupNorm(8, cnum),
            nn.SiLU(),
            nn.Conv2d(cnum, 1, 3, padding=1),
            nn.Tanh()  # Noise prediction in [-1, 1]
        )
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(cnum, 1, 1),
            nn.Sigmoid()  # Segmentation in [0, 1]
        )
        
    def forward(self, x, mask, CAM, slice_ratio, t=None):
        """
        Forward pass for both training and inference.
        
        Args:
            x: Input CT (noisy during training, or x_t during sampling)
            mask: Inpainting mask
            CAM: Grad-CAM attention
            slice_ratio: Slice position (B,)
            t: Timesteps (B,) or None (defaults to T-1)
            
        Returns:
            Tuple of 7 tensors matching GAN interface
        """
        B, _, H, W = x.shape
        device = x.device
        
        # Default timestep if not provided
        if t is None:
            t = torch.full((B,), self.T - 1, device=device, dtype=torch.float32)
        
        # Prepare time embeddings
        t_emb = self.time_embed(t)
        
        # Prepare position map
        slice_ratio_map = slice_ratio.view(B, 1, 1, 1).expand(-1, -1, H, W)
        
        # Concatenate conditional inputs
        x_in = torch.cat([x, mask, CAM, slice_ratio_map], dim=1)  # (B, 4, H, W)
        
        # Encoder path
        x0 = self.input_conv(x_in, t_emb)
        x1, skip1 = self.down1(x0, t_emb)
        x2, skip2 = self.down2(x1, t_emb)
        x3, skip3 = self.down3(x2, t_emb)
        
        # Bottleneck
        bottleneck = self.bottleneck(x3, t_emb)
        
        # Height prediction from bottleneck
        height = self.height_head(bottleneck)  # (B, 1)
        
        # Decoder path
        d3 = self.up3(bottleneck, skip3, t_emb)
        d2 = self.up2(d3, skip2, t_emb)
        d1 = self.up1(d2, skip1, t_emb)
        
        # Output predictions
        pred_noise = self.noise_head(d1)    # Predicted noise epsilon
        seg = self.seg_head(d1)              # Segmentation mask
        
        # Return all 7 outputs to match GAN interface
        # For now, use same outputs for both stages (can refine later)
        coarse_seg = seg
        fine_seg = seg
        x_stage1 = pred_noise  # Will be converted to x_0 in pix2pix_model
        x_stage2 = pred_noise
        offset_flow = torch.zeros(B, 2, H, W, device=device)  # Unused, for compatibility
        pred1_h = height.T  # (1, B) to match GAN output shape
        pred2_h = height.T
        
        return coarse_seg, fine_seg, x_stage1, x_stage2, offset_flow, pred1_h, pred2_h
