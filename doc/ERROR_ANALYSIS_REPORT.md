# HealthiVert-GAN Pipeline: Error Log & Solutions

**Project**: HealthiVert-GAN - Vertebral Compression Fracture Reconstruction  
**Date**: December 30, 2024  
**Author**: Ahmed  
**Purpose**: Document all errors encountered and solutions applied during pipeline implementation

---

## 📋 Table of Contents
1. [Training Errors](#training-errors)
2. [Testing Errors](#testing-errors)
3. [Dataset Errors](#dataset-errors)
4. [Evaluation Errors](#evaluation-errors)
5. [Dependency Errors](#dependency-errors)

---

## 1. Training Errors

### Error 1.1: Unrecognized Training Arguments in TestOptions
**Location**: `train.py` line 178  
**Error Message**: 
```
error: unrecognized arguments: --n_epochs 3 --n_epochs_decay 2 --display_id 0
```

**Cause**: 
- `train.py` called `TestOptions().parse()` which tried to parse the command line again
- Training-only arguments (`--n_epochs`, `--n_epochs_decay`, `--display_id`) not defined in `TestOptions`
- Both parsers reading the same command line arguments

**Solution**:
```python
# Before (❌):
opt_test = TestOptions().parse()  # Parses command line again

# After (✅):
import copy
opt_test = copy.deepcopy(opt)  # Copy from training options
opt_test.phase = 'test'
opt_test.batch_size = 5
opt_test.serial_batches = True
opt_test.isTrain = False
```

**Fix Type**: Code modification - replaced command-line parsing with option copying

---

### Error 1.2: Training Stopped After 100 Iterations
**Location**: `train.py` lines 215-221  
**Error Message**: None (intentional behavior)  
**Issue**: Training stopped at 100 iterations instead of completing 5 epochs

**Cause**: 
- `--sample-test 100` flag limits training to 100 **iterations** (not epochs)
- Designed for quick pipeline testing, not full training

**Solution**: Remove `--sample-test` flag or increase value
```bash
# Before (only 100 iterations):
--sample-test 100

# After (full training):
# Remove the flag entirely
```

**Fix Type**: Command-line argument adjustment

---

## 2. Testing Errors

### Error 2.1: Empty Test Results Directory
**Location**: `test.py`  
**Error Message**: None (0 files generated)  
**Issue**: Test generated directories but no images

**Cause**: 
- `test.py` defaults to `phase='test'`
- JSON file had 89 samples in `train` but 0 in `test`
- Dataset found no samples to process

**Solution**: Add `--phase train` to test on training data
```bash
# Before (❌ 0 samples):
python test.py --name healthivert_full --epoch 5

# After (✅ 89 samples):
python test.py --name healthivert_full --epoch 5 --phase train
```

**Fix Type**: Command-line argument addition

---

## 3. Dataset Errors

### Error 3.1: CAM Heatmap Files Not Found
**Location**: `aligned_sagittal_dataset.py` line 168-176  
**Error Message**: 
```
Warning: CAM file not found: .../sub-verse004_ct_23_1.nii.gz
```

**Cause**: 
- Dataset code expected files with `_0` or `_1` suffix (legacy format)
- Generated heatmaps had no suffix: `sub-verse004_ct_23.nii.gz`

**Solution**: Update dataset to try without suffix first
```python
# Before (❌):
CAM_path_0 = f"{vertebra_id}_0.nii.gz"  # Always tries suffix

# After (✅):
CAM_path = f"{vertebra_id}.nii.gz"  # Try without suffix first
if not os.path.exists(CAM_path):
    CAM_path_0 = f"{vertebra_id}_0.nii.gz"  # Fallback to legacy
```

**Fix Type**: Code modification - added fallback logic

---

### Error 3.2: AttributeError - 'vertebra_id' Not Defined
**Location**: `aligned_sagittal_dataset.py` lines 63-81  
**Error Message**: 
```
AttributeError: 'AlignedSagittalDataset' object has no attribute 'vertebra_id'
```

**Cause**: 
- `self.vertebra_id = np.array(...)` was **inside** the patient loop
- Only executed once per loop, then overwritten
- Final execution happened inside the loop, not accessible outside

**Solution**: Unindent assignment to be **outside** the loop
```python
# Before (❌):
for patient in patients:
    # ...process patient...
    if opt.vert_class == "normal":
        self.vertebra_id = np.array(...)  # Inside loop!

# After (✅):
for patient in patients:
    # ...process patient...
# Outside loop now:
if opt.vert_class == "normal":
    self.vertebra_id = np.array(...)
```

**Fix Type**: Indentation fix

---

## 4. Evaluation Errors

### Error 4.1: NumPy/Pandas Binary Incompatibility
**Location**: Multiple evaluation scripts  
**Error Message**: 
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject
```

**Cause**: 
- Pandas compiled against NumPy 1.x (dtype size 96 bytes)
- NumPy 2.x installed (dtype size 88 bytes)
- C-level ABI mismatch

**Solution**: Reinstall both packages
```bash
pip install numpy pandas --force-reinstall --no-cache-dir
```

**Installed Versions**:
- numpy: 2.4.0
- pandas: 2.3.3

**Fix Type**: Dependency reinstallation

---

### Error 4.2: RHLV Quantification - Empty Output
**Location**: `evaluation/RHLV_quantification.py`  
**Error Message**: None (silent failure - directories created but no Excel file)

**Cause**: 
- Script expected nested folder structure: `output/experiment/label_fake/`
- User provided direct path: `output/CT_fake/`
- Script couldn't find `label_fake` subdirectory

**Solution**: Auto-detect direct vs nested structure
```python
# Added detection logic:
if 'CT_fake' in output_folder or 'label_fake' in output_folder:
    output_folder = os.path.dirname(output_folder)  # Use parent
    # Process direct structure: output/{CT_fake, label_fake}
```

**Fix Type**: Code modification - added path detection

---

### Error 4.3: SVM Grading - Zero Validation Samples
**Location**: `evaluation/SVM_grading.py` line 45-46  
**Error Message**: 
```
ValueError: Found array with 0 sample(s) (shape=(0, 3)) while a minimum of 1 is required
```

**Cause**: 
- All data had `Dataset='train'`
- No rows with `Dataset='val'`
- Validation set was empty

**Solution**: Auto-split when validation missing
```python
# Added check:
if val_data.empty:
    from sklearn.model_selection import train_test_split
    train_test_data, val_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data['Label']
    )
```

**Fix Type**: Code modification - added auto-splitting

---

### Error 4.4: SVM Grading - Pandas Index Mismatch
**Location**: `evaluation/SVM_grading.py` line 82  
**Error Message**: 
```
KeyError: '[16, 22, 23, ...] not in index'
```

**Cause**: 
- Pandas Series retained original Excel row indices (16, 22, 23...)
- StratifiedKFold returns 0-based indices (0, 1, 2...)
- Indexing with new indices on old DataFrame caused mismatch

**Solution**: Convert to numpy arrays before splitting
```python
# Before (❌):
X_train_test = train_test_data[features]  # Pandas Series
y_train_test = train_test_data['Label']   # Pandas Series

# After (✅):
X_train_test = train_test_data[features].values  # NumPy array
y_train_test = train_test_data['Label'].values   # NumPy array
```

**Fix Type**: Data type conversion

---

### Error 4.5: SVM Grading - Insufficient Data for Cross-Validation
**Location**: `evaluation/SVM_grading.py` line 77  
**Error Message**: 
```
UserWarning: The least populated class in y has only 3 members, which is less than n_splits=5
```

**Cause**: 
- Fixed 5-fold cross-validation
- User manually created labels with only 3 samples in smallest class

**Solution**: Dynamically adjust folds
```python
# Calculate minimum possible folds:
n_splits = min(5, min(np.bincount(y_train_test)))

# If < 2 samples in any class:
if n_splits < 2:
    # Skip cross-validation, do single train/test evaluation
```

**Fix Type**: Code modification - dynamic fold calculation

---

## 5. Dependency Errors

### Error 5.1: Missing evaluation_metrics.py Output
**Location**: User-created script  
**Error Message**: 
```
Average SSIM: nan
Average PSNR: nan
RuntimeWarning: Mean of empty slice
```

**Cause**: 
- Lists `ssim_scores` and `psnr_scores` were empty
- Script had placeholder comment: "Add your metric calculation code here"
- `np.mean([])` returns NaN

**Solution**: Provided complete implementation
```python
# Added actual metric calculation:
for fake_B_path in fake_B_files:
    fake_B = np.array(Image.open(fake_B_path).convert('L'))
    real_B = np.array(Image.open(real_B_path).convert('L'))
    
    ssim_val = ssim(fake_B, real_B, data_range=255)
    ssim_scores.append(ssim_val)
```

**Fix Type**: Implementation completion

---

## 📊 Summary Statistics

| Error Category | Count | Resolution Method |
|----------------|-------|-------------------|
| Training Issues | 2 | Code modification, flag removal |
| Testing Issues | 1 | Command-line adjustment |
| Dataset Issues | 2 | Code modification, indentation |
| Evaluation Issues | 5 | Code + dependency fixes |
| Total Errors Fixed | 10 | - |

---

## 🔑 Key Lessons Learned

1. **Option Parsing**: Avoid parsing command line multiple times - use `copy.deepcopy()` for derived options
2. **Dataset Phases**: Always verify JSON has data in the phase you're testing (train/val/test)
3. **File Naming**: Support multiple file naming conventions with fallback logic
4. **Data Structures**: Convert pandas to numpy before sklearn operations to avoid index issues
5. **Dependency Management**: Keep numpy/pandas versions compatible (force-reinstall when needed)
6. **Dynamic Parameters**: Auto-adjust parameters (folds, splits) based on available data
7. **Validation**: Add checks for empty datasets before processing

---

## ✅ Pipeline Status: COMPLETE

All errors resolved. Pipeline successfully executes from preprocessing through SVM grading.

**Final Results**:
- Training: 5 epochs completed
- 2D Metrics: SSIM 0.9174, PSNR 32.38 dB
- 3D Volumes: 89 reconstructed vertebrae
- RHLV: Quantification complete (results.xlsx generated)
- SVM: Classification tested (accuracy 44% on limited data)

**Recommendation**: Scale to 50+ patients and 100+ epochs for production-quality results.