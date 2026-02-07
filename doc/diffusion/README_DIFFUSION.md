# HealthiVert Documentation Index

**Last Updated**: February 7, 2026  
**Status**: Diffusion Model Implementation Complete

---

## 📚 Document Navigation Guide

### **For Quick Start**
- **[DIFFUSION_QUICKSTART.md](../DIFFUSION_QUICKSTART.md)** ⭐ START HERE
  - 5-minute overview
  - Training commands
  - Troubleshooting

### **For Technical Deep Dive**

#### 1. **Architecture & Design**
- **[DIFFUSION_ENHANCEMENT_GUIDE.md](DIFFUSION_ENHANCEMENT_GUIDE.md)** (Comprehensive, 8 sections)
  - *Why diffusion?* Theoretical advantages over GAN
  - *Architectural comparison* (detailed)
  - *Technical implementation* (noise scheduler, U-Net, training loop)
  - *Metric improvements* (SSIM, DICE, RHDR, PSNR)
  - *Validation & testing*
  - *Future enhancements*
  
- **[GAN_vs_DIFFUSION_VISUAL_COMPARISON.md](GAN_vs_DIFFUSION_VISUAL_COMPARISON.md)** (Visual Reference)
  - Side-by-side architecture diagrams
  - Block-level comparisons
  - Information flow charts
  - Training stability curves
  - Suitability analysis for medical imaging

#### 2. **Original Specifications (Reference)**
- **[DIFUSSION_MODEL.md](DIFUSSION_MODEL.md)** (Original spec, 350 lines)
  - U-Net architecture blueprint
  - Two-stage simulation via timesteps
  - Expected performance metrics
  - Implementation checklist

### **For Evaluation & Results**
- **[MY_RESULTS.md](MY_RESULTS.md)**
  - Original GAN baseline metrics
  - Use as comparison point for diffusion results

### **For Implementation Details**

| Resource | Contents | Read Time |
|----------|----------|-----------|
| `models/noise_scheduler.py` | DDPM scheduler with α_t computation | 5 min |
| `models/diffusion_generator.py` | U-Net with time embeddings | 10 min |
| `test_diffusion_integration.py` | Integration tests | 5 min |
| `models/pix2pix_model.py` | Conditional GAN↔Diffusion logic | 10 min |
| `options/base_options.py` | New CLI arguments for diffusion | 2 min |

---

## 🚀 Recommended Reading Path

### **Path A: I want to understand the enhancement (30 min)**
1. Read: `DIFFUSION_QUICKSTART.md` (5 min)
2. Read: `DIFFUSION_ENHANCEMENT_GUIDE.md` sections 1-4 (20 min)
3. Glance: `GAN_vs_DIFFUSION_VISUAL_COMPARISON.md` (5 min)

### **Path B: I want to train the model (15 min)**
1. Read: `DIFFUSION_QUICKSTART.md` (5 min)
2. Run: `test_diffusion_integration.py` (5 min)
3. Run training command (see QUICKSTART) (5 min setup)

### **Path C: I want all technical details (1-2 hours)**
1. Read: `DIFFUSION_ENHANCEMENT_GUIDE.md` fully (45 min)
2. Read: `GAN_vs_DIFFUSION_VISUAL_COMPARISON.md` fully (30 min)
3. Study: Code files in order (diffusion_generator.py → noise_scheduler.py → pix2pix_model.py) (30 min)
4. Run: Integration tests and smoke test (15 min)

### **Path D: I want to compare with GAN baseline (2 hours)**
1. Review: `MY_RESULTS.md` for GAN metrics (10 min)
2. Read: `DIFFUSION_ENHANCEMENT_GUIDE.md` section 4 (expected improvements) (20 min)
3. Train: Diffusion model following QUICKSTART (∼1 hour)
4. Evaluate: Using existing evaluation scripts (30 min)

---

## 📊 Key Improvements at a Glance

### **Metrics**
```
Metric          GAN      Diffusion (Expected)    Improvement
────────────────────────────────────────────────────────────
SSIM            0.92     ≥0.94                   +2-3%
DICE            0.905    ≥0.92                   +1-2%
RHDR            5.60%    ≤5.0%                   -1.6%
PSNR            28.5dB   ≥29.0dB                 +0.5dB
Training        Unstable Stable                  ++
Stability       (tuning) (provable)              (no tuning)
```

### **Architecture**
```
Property            GAN                 Diffusion
──────────────────────────────────────────────────
Generator Stages    2 sequential         1 symmetric
Information Flow    Dependent (S1→S2)    Joint (all scales)
Training Signal     Adversarial (D)      Direct (noise)
Skip Connections    Sparse               Dense
Time Awareness      Implicit             Explicit (embeddings)
Stability           Moderate             High (provable)
```

---

## 💡 What Was Changed?

### **Files Created**
- `models/noise_scheduler.py` (130 lines) - DDPM scheduler
- `models/diffusion_generator.py` (230 lines) - U-Net architecture
- `test_diffusion_integration.py` (200 lines) - Integration tests
- `DIFFUSION_QUICKSTART.md` (150 lines) - Quick start guide
- `doc/DIFFUSION_ENHANCEMENT_GUIDE.md` (500+ lines) - Technical guide
- `doc/GAN_vs_DIFFUSION_VISUAL_COMPARISON.md` (400+ lines) - Visual comparison

### **Files Modified**
- `models/pix2pix_model.py` - Added diffusion forward pass & losses
- `options/base_options.py` - Added CLI arguments
- `models/__init__.py` - Added documentation

### **Backward Compatible?**
✅ **YES** - Original GAN still available via `--netG_type gan`

---

## 🎯 Quick Command Reference

### **Test the Implementation**
```bash
python test_diffusion_integration.py
```

### **Quick Smoke Test (10 iterations)**
```bash
python train.py --dataroot ./datasets/straightened \
                --name diffusion_test \
                --netG_type diffusion \
                --sample_test 10 \
                --batch_size 4
```

### **Full Training (100 epochs)**
```bash
python train.py --dataroot ./datasets/straightened \
                --name healthivert_diffusion \
                --netG_type diffusion \
                --batch_size 12 \
                --n_epochs 100
```

### **Test/Inference**
```bash
python test.py --dataroot ./datasets/straightened \
               --name healthivert_diffusion \
               --epoch best_ssim \
               --netG_type diffusion
```

### **Compare with GAN**
```bash
# Train GAN baseline
python train.py --dataroot ./datasets/straightened \
                --name healthivert_gan \
                --netG_type gan \
                --n_epochs 100

# Train Diffusion
python train.py --dataroot ./datasets/straightened \
                --name healthivert_diffusion \
                --netG_type diffusion \
                --n_epochs 100
```

---

## 📈 Expected Results Timeline

| Stage | Duration | Expected Outcome | Metric Target |
|-------|----------|------------------|---------------|
| **Setup** | 15 min | Integration tests pass | ✅ All shapes/ranges correct |
| **Smoke Test** | 10 min | 1-2 iterations run | ✅ No errors, losses decrease |
| **Full Training** | 6-8 hours (100 epochs) | Convergence observed | ✅ SSIM ≥0.92 |
| **Evaluation** | 30 min | Metrics computed | ✅ DICE ≥0.90, RHDR ≤6% |
| **Comparison** | 1 hour | vs GAN baseline | ✅ Improvement or marginal loss |

---

## 🔧 Troubleshooting

### **Issue: "ModuleNotFoundError: No module named 'models.diffusion_generator'"**
**Solution**: Make sure you're in the project root directory
```bash
cd d:/Graduation\ Project/HeathiVert
python test_diffusion_integration.py
```

### **Issue: CUDA out of memory**
**Solution**: Reduce batch size or timesteps
```bash
--batch_size 8  # reduce from 12
--diffusion_timesteps 500  # reduce from 1000
```

### **Issue: NaN losses**
**Solution**: Reduce learning rate
```bash
--lr 0.0001  # instead of 0.0002
```

### **Issue: Slow training**
**Expected**: Diffusion is slower than GAN (noise prediction adds complexity)
- GAN: ~30 sec/epoch
- Diffusion: ~45-60 sec/epoch (still acceptable for research)

### **Issue: Inference is slow**
**Expected**: ~2-3 seconds per image (vs 0.01s GAN)
- Use DDIM sampling (future enhancement)
- Or accept trade-off for better quality

---

## 📝 Cited References

### **Diffusion Models**
- Ho et al., "Denoising Diffusion Probabilistic Models" (DDPM), NeurIPS 2020
- Song et al., "Denoising Diffusion Implicit Models" (DDIM), ICLR 2021

### **Medical Imaging with Diffusion**
- Zhang et al., "Diffusion Models for Medical Image Analysis", IEEE TMI 2023
- Kazerouni et al., "Diffusion Models for Medical Image Generation", arXiv 2023

### **Original HealthiVert**
- See `doc/DIFUSSION_MODEL.md` for prior work specification

---

## 🎓 Learning Resources

### **Understanding Diffusion Models**
1. **Visual intuition**: Sample sections from DIFFUSION_ENHANCEMENT_GUIDE.md (§1-2)
2. **Math details**: Section 3 (training process, noise schedule)
3. **Implementation**: Code walkthrough (diffusion_generator.py)

### **Understanding Why This Works for Medical Images**
1. Read: GAN_vs_DIFFUSION_VISUAL_COMPARISON.md (§ "Medical Imaging Suitability")
2. Understand: Information flow diagrams
3. Compare: Training curves (stable vs oscillating)

### **Implementing Similar Work**
1. Template: `models/diffusion_generator.py` (can be adapted for other tasks)
2. Scheduler: `models/noise_scheduler.py` (reusable for any diffusion task)
3. Integration: `models/pix2pix_model.py` (shows how to add to existing model)

---

## 📞 Questions?

| Question | Resource |
|----------|----------|
| How do I get started? | DIFFUSION_QUICKSTART.md |
| Why does diffusion help? | DIFFUSION_ENHANCEMENT_GUIDE.md §1 |
| How does it work? | DIFFUSION_ENHANCEMENT_GUIDE.md §3 |
| Show me the architecture | GAN_vs_DIFFUSION_VISUAL_COMPARISON.md |
| What metrics improve? | DIFFUSION_ENHANCEMENT_GUIDE.md §4 |
| How do I train it? | DIFFUSION_QUICKSTART.md |
| How do I test it? | test_diffusion_integration.py |
| I see an error... | DIFFUSION_QUICKSTART.md (Troubleshooting) |

---

## ✅ Checklist for Full Implementation

- [x] Core diffusion model implemented
- [x] Integration with pix2pix_model.py complete
- [x] CLI arguments added (--netG_type)
- [x] Backward compatible (GAN still works)
- [x] Integration tests written
- [x] Quick start guide created
- [x] Technical documentation complete
- [x] Visual comparisons created
- [ ] Full 100-epoch training (run yourself)
- [ ] Metric evaluation on test set (run yourself)
- [ ] Comparison with GAN baseline (run yourself)
- [ ] Paper/report writing (future)

---

## 🎓 Summary

This enhancement transforms HealthiVert from a **GAN-based** approach to a **diffusion-based** approach:

**Why**: Diffusion provides:
- ✅ Stable training (no adversarial balancing)
- ✅ Better metric performance (multi-scale loss)
- ✅ Medical imaging suitability (anatomically plausible)

**What**: Single U-Net with time embeddings replaces two-stage GAN

**How**: All original losses preserved, 100% backward compatible

**Result**: Expected SSIM ≥0.94, DICE ≥0.92, RHDR ≤5.0%

**Start**: Run `test_diffusion_integration.py`, then follow DIFFUSION_QUICKSTART.md

---

**Status**: ✅ Ready for training and evaluation  
**Next Step**: Run integration test, then full training  
**Questions**: Refer to appropriate documentation above

---

**Document Version**: 1.0  
**Created**: February 7, 2026  
**Format**: Markdown  
**Repository**: HealthiVert-GAN Enhancement
