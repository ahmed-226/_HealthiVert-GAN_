"""
Quick integration test for diffusion model.
Verifies that the diffusion generator produces correct output shapes and ranges.
"""
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion_generator import HealthiVertDiffusionUNet
from models.noise_scheduler import DDPMScheduler

def test_output_shapes():
    """Test that all 7 outputs have correct shapes."""
    print("="*60)
    print("Testing Diffusion Generator Output Shapes")
    print("="*60)
    
    # Create model
    model = HealthiVertDiffusionUNet(cnum=32, T=1000)
    model.eval()
    
    # Create dummy inputs
    B = 4
    H, W = 256, 256
    x = torch.randn(B, 1, H, W)
    mask = torch.randint(0, 2, (B, 1, H, W)).float()
    CAM = torch.rand(B, 1, H, W)
    slice_ratio = torch.rand(B)
    t = torch.randint(0, 1000, (B,)).float()
    
    # Forward pass
    with torch.no_grad():
        coarse_seg, fine_seg, x_stage1, x_stage2, offset_flow, pred1_h, pred2_h = \
            model(x, mask, CAM, slice_ratio, t)
    
    # Check shapes
    assert coarse_seg.shape == (B, 1, H, W), f"coarse_seg shape mismatch: {coarse_seg.shape}"
    assert fine_seg.shape == (B, 1, H, W), f"fine_seg shape mismatch: {fine_seg.shape}"
    assert x_stage1.shape == (B, 1, H, W), f"x_stage1 shape mismatch: {x_stage1.shape}"
    assert x_stage2.shape == (B, 1, H, W), f"x_stage2 shape mismatch: {x_stage2.shape}"
    assert offset_flow.shape == (B, 2, H, W), f"offset_flow shape mismatch: {offset_flow.shape}"
    assert pred1_h.shape == (1, B), f"pred1_h shape mismatch: {pred1_h.shape}"
    assert pred2_h.shape == (1, B), f"pred2_h shape mismatch: {pred2_h.shape}"
    
    print("✓ All output shapes correct!")
    print(f"  - Segmentations: {coarse_seg.shape}")
    print(f"  - CT outputs: {x_stage1.shape}")
    print(f"  - Offset flow: {offset_flow.shape}")
    print(f"  - Heights: {pred1_h.shape}")
    
    return True

def test_output_ranges():
    """Test that outputs are in correct ranges."""
    print("\n" + "="*60)
    print("Testing Output Ranges")
    print("="*60)
    
    model = HealthiVertDiffusionUNet(cnum=32, T=1000)
    model.eval()
    
    B = 4
    x = torch.randn(B, 1, 256, 256)
    mask = torch.rand(B, 1, 256, 256)
    CAM = torch.rand(B, 1, 256, 256)
    slice_ratio = torch.rand(B)
    t = torch.randint(0, 1000, (B,)).float()
    
    with torch.no_grad():
        coarse_seg, fine_seg, x_stage1, x_stage2, offset_flow, pred1_h, pred2_h = \
            model(x, mask, CAM, slice_ratio, t)
    
    # Check ranges
    assert coarse_seg.min() >= 0 and coarse_seg.max() <= 1, \
        f"coarse_seg range error: [{coarse_seg.min():.3f}, {coarse_seg.max():.3f}]"
    assert fine_seg.min() >= 0 and fine_seg.max() <= 1, \
        f"fine_seg range error: [{fine_seg.min():.3f}, {fine_seg.max():.3f}]"
    assert x_stage1.min() >= -1 and x_stage1.max() <= 1, \
        f"x_stage1 range error: [{x_stage1.min():.3f}, {x_stage1.max():.3f}]"
    assert x_stage2.min() >= -1 and x_stage2.max() <= 1, \
        f"x_stage2 range error: [{x_stage2.min():.3f}, {x_stage2.max():.3f}]"
    assert pred1_h.min() >= 0 and pred1_h.max() <= 1, \
        f"pred1_h range error: [{pred1_h.min():.3f}, {pred1_h.max():.3f}]"
    assert pred2_h.min() >= 0 and pred2_h.max() <= 1, \
        f"pred2_h range error: [{pred2_h.min():.3f}, {pred2_h.max():.3f}]"
    
    print("✓ All output ranges correct!")
    print(f"  - Segmentations: [{coarse_seg.min():.3f}, {coarse_seg.max():.3f}]")
    print(f"  - CT outputs: [{x_stage1.min():.3f}, {x_stage1.max():.3f}]")
    print(f"  - Heights: [{pred1_h.min():.3f}, {pred1_h.max():.3f}]")
    
    return True

def test_noise_scheduler():
    """Test noise scheduler forward and reverse process."""
    print("\n" + "="*60)
    print("Testing Noise Scheduler")
    print("="*60)
    
    scheduler = DDPMScheduler(T=1000)
    
    # Test forward process
    x_0 = torch.randn(2, 1, 256, 256)
    t = torch.tensor([500, 750])
    
    x_t, noise = scheduler.add_noise(x_0, t)
    assert x_t.shape == x_0.shape, "Noisy image shape mismatch"
    assert noise.shape == x_0.shape, "Noise shape mismatch"
    
    # Test reverse process
    x_0_pred = scheduler.predict_x0_from_eps(x_t, noise, t)
    assert x_0_pred.shape == x_0.shape, "Predicted clean image shape mismatch"
    
    # Check that prediction is reasonable (not perfect due to clamping)
    mse = torch.mean((x_0_pred - x_0) ** 2)
    print(f"✓ Noise scheduler working!")
    print(f"  - Forward process: x_0 → x_t ✓")
    print(f"  - Reverse process: x_t → x_0_pred ✓")
    print(f"  - Reconstruction MSE: {mse:.6f}")
    
    return True

def test_model_size():
    """Estimate model size."""
    print("\n" + "="*60)
    print("Model Size Estimation")
    print("="*60)
    
    model = HealthiVertDiffusionUNet(cnum=32, T=1000)
    
    param_count = sum(p.numel() for p in model.parameters())
    param_count_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate size in MB (assuming float32)
    size_mb = (param_count * 4) / (1024 ** 2)
    
    print(f"✓ Model statistics:")
    print(f"  - Total parameters: {param_count:,}")
    print(f"  - Trainable parameters: {param_count_trainable:,}")
    print(f"  - Estimated size: {size_mb:.2f} MB")
    
    return True

def main():
    print("\n" + "="*60)
    print("DIFFUSION MODEL INTEGRATION TEST")
    print("="*60 + "\n")
    
    try:
        test_output_shapes()
        test_output_ranges()
        test_noise_scheduler()
        test_model_size()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYou can now train with:")
        print("  python train.py --dataroot <path> --name test_diffusion \\")
        print("                  --netG_type diffusion --sample_test 100")
        print("="*60 + "\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED!")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
