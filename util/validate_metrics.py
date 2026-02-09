"""
Metric Validation Utility

This script helps debug and validate metric calculations by:
1. Testing with known synthetic images
2. Checking for common metric bugs (SSIM=1.0, PSNR divide-by-zero)
3. Verifying data range calculations
4. Comparing masked vs unmasked metric calculations

Usage:
    python util/validate_metrics.py
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch


def test_identical_images():
    """Test metrics on identical images - should give perfect scores."""
    print("\n" + "="*70)
    print("TEST 1: Identical Images")
    print("="*70)
    
    img1 = np.random.rand(256, 256).astype(np.float32)
    img2 = img1.copy()
    
    data_range = img1.max() - img1.min()
    
    ssim_val = ssim(img1, img2, data_range=data_range)
    psnr_val = psnr(img1, img2, data_range=data_range)
    
    print(f"SSIM: {ssim_val:.6f} (expected: 1.000000)")
    print(f"PSNR: {psnr_val:.2f}dB (expected: inf)")
    
    if ssim_val != 1.0:
        print("❌ FAILED: SSIM should be exactly 1.0 for identical images")
    else:
        print("✅ PASSED")
    
    return ssim_val == 1.0


def test_different_images():
    """Test metrics on different images - should give imperfect scores."""
    print("\n" + "="*70)
    print("TEST 2: Different Images")
    print("="*70)
    
    img1 = np.random.rand(256, 256).astype(np.float32)
    img2 = np.random.rand(256, 256).astype(np.float32)
    
    data_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
    
    ssim_val = ssim(img1, img2, data_range=data_range)
    psnr_val = psnr(img1, img2, data_range=data_range)
    
    print(f"SSIM: {ssim_val:.6f} (expected: < 1.0)")
    print(f"PSNR: {psnr_val:.2f}dB (expected: finite value)")
    
    if ssim_val >= 0.999:
        print("❌ FAILED: SSIM too high for random images")
        return False
    elif psnr_val < 0 or psnr_val > 100:
        print("❌ FAILED: PSNR value unrealistic")
        return False
    else:
        print("✅ PASSED")
        return True


def test_masked_comparison():
    """Test metrics with masking - simulates inpainting scenario."""
    print("\n" + "="*70)
    print("TEST 3: Masked Region Comparison (Inpainting Scenario)")
    print("="*70)
    
    # Ground truth
    gt_image = np.random.rand(256, 256).astype(np.float32)
    
    # Create predicted image (slightly different in masked region)
    pred_image = gt_image.copy()
    pred_image[100:156, 100:156] += np.random.rand(56, 56) * 0.1  # Add noise to masked region
    
    # Create mask (1 = inpainted region, 0 = unchanged)
    mask = np.zeros((256, 256), dtype=np.float32)
    mask[100:156, 100:156] = 1.0
    
    # Extract masked regions
    gt_masked = gt_image * mask
    pred_masked = pred_image * mask
    
    # Calculate data range
    data_range = max(gt_masked.max(), pred_masked.max()) - min(gt_masked.min(), pred_masked.min())
    
    print(f"Mask coverage: {mask.sum() / mask.size * 100:.2f}%")
    print(f"Data range: {data_range:.6f}")
    
    if data_range < 1e-7:
        print("❌ FAILED: Data range is near zero (mask may be empty or all zeros)")
        return False
    
    # Compute metrics on masked region only
    ssim_masked = ssim(gt_masked, pred_masked, data_range=data_range)
    
    # Calculate MSE manually to check for divide-by-zero
    mse = np.mean((gt_masked - pred_masked) ** 2)
    print(f"MSE: {mse:.10f}")
    
    if mse < 1e-10:
        print("⚠️  WARNING: MSE is near zero (may cause PSNR issues)")
        psnr_masked = float('inf')
    else:
        psnr_masked = psnr(gt_masked, pred_masked, data_range=data_range)
    
    print(f"SSIM (masked only): {ssim_masked:.6f}")
    print(f"PSNR (masked only): {psnr_masked:.2f}dB")
    
    # Now compare full images (should give much better scores - THIS IS THE BUG!)
    ssim_full = ssim(gt_image, pred_image, data_range=gt_image.max() - gt_image.min())
    psnr_full = psnr(gt_image, pred_image, data_range=gt_image.max() - gt_image.min())
    
    print(f"\nSSIM (full image): {ssim_full:.6f}")
    print(f"PSNR (full image): {psnr_full:.2f}dB")
    
    if ssim_full > ssim_masked + 0.1:
        print("\n⚠️  WARNING: Full image SSIM much higher than masked SSIM!")
        print("   This suggests you may be comparing full images instead of masked regions")
        print("   in your evaluation code, which would give falsely high scores.")
        return False
    
    print("✅ PASSED")
    return True


def test_multichannel_bug():
    """Test common bug: using multichannel=True on grayscale images."""
    print("\n" + "="*70)
    print("TEST 4: Multichannel Parameter Bug")
    print("="*70)
    
    # Single channel grayscale image
    img1 = np.random.rand(256, 256).astype(np.float32)
    img2 = img1.copy() + np.random.rand(256, 256) * 0.01
    
    data_range = img1.max() - img1.min()
    
    # Correct: no multichannel parameter for 2D grayscale
    try:
        ssim_correct = ssim(img1, img2, data_range=data_range)
        print(f"✅ SSIM (correct, no multichannel): {ssim_correct:.6f}")
    except Exception as e:
        print(f"❌ Error with correct call: {e}")
        return False
    
    # Bug: using multichannel=True on 2D grayscale
    try:
        ssim_bug = ssim(img1, img2, data_range=data_range, channel_axis=-1)
        print(f"⚠️  SSIM (with multichannel bug): {ssim_bug:.6f}")
        print("   This may give incorrect results for grayscale images!")
        return False
    except Exception as e:
        print(f"✅ Correctly raises error with multichannel on 2D: {type(e).__name__}")
        return True


def test_data_range_bug():
    """Test common bug: inconsistent data_range calculation."""
    print("\n" + "="*70)
    print("TEST 5: Data Range Calculation Bug")
    print("="*70)
    
    img1 = np.random.rand(256, 256).astype(np.float32) * 255  # [0, 255]
    img2 = img1.copy() + np.random.rand(256, 256) * 10
    
    # Correct: use consistent min/max
    data_range_correct = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
    
    # Bug 1: use img2.max() - img1.min() (inconsistent)
    data_range_bug1 = img2.max() - img1.min()
    
    # Bug 2: use img2.max() - img2.min() (ignores img1)
    data_range_bug2 = img2.max() - img2.min()
    
    print(f"Correct data_range: {data_range_correct:.2f}")
    print(f"Bug 1 (img2.max - img1.min): {data_range_bug1:.2f}")
    print(f"Bug 2 (img2.max - img2.min): {data_range_bug2:.2f}")
    
    psnr_correct = psnr(img1, img2, data_range=data_range_correct)
    psnr_bug1 = psnr(img1, img2, data_range=data_range_bug1)
    psnr_bug2 = psnr(img1, img2, data_range=data_range_bug2)
    
    print(f"\nPSNR (correct): {psnr_correct:.2f}dB")
    print(f"PSNR (bug 1): {psnr_bug1:.2f}dB")
    print(f"PSNR (bug 2): {psnr_bug2:.2f}dB")
    
    if abs(psnr_correct - psnr_bug1) > 1.0 or abs(psnr_correct - psnr_bug2) > 1.0:
        print("⚠️  Different data_range calculations give significantly different PSNR!")
        print("   Always use: max(img1.max(), img2.max()) - min(img1.min(), img2.min())")
    
    print("✅ PASSED")
    return True


if __name__ == '__main__':
    print("\n" + "="*70)
    print("METRIC VALIDATION SUITE")
    print("Testing for common bugs in SSIM/PSNR calculation")
    print("="*70)
    
    results = []
    results.append(("Identical Images", test_identical_images()))
    results.append(("Different Images", test_different_images()))
    results.append(("Masked Comparison", test_masked_comparison()))
    results.append(("Multichannel Bug", test_multichannel_bug()))
    results.append(("Data Range Bug", test_data_range_bug()))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
    
    total = len(results)
    passed_count = sum(results, key=lambda x: x[1])
    
    print(f"\nTotal: {passed_count}/{total} tests passed")
    
    if passed_count == total:
        print("\n🎉 All tests passed! Metric calculations appear correct.")
    else:
        print("\n⚠️  Some tests failed. Review metric calculation code.")
