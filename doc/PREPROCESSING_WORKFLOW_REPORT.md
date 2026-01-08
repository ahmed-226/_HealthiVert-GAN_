# HealthiVert-GAN Preprocessing Workflow Report

## 📋 Executive Summary

This report documents the preprocessing workflow analysis and modifications made to remove hardcoded paths and create proper command-line argument interfaces for the HealthiVert-GAN pipeline.

---

## 🔗 Preprocessing Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 0: External Segmentation (SCNet - Not in codebase)                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Input:  Raw CT scans (.nii.gz)                                             │
│  Output: Segmentation masks with vertebra labels (1-24)                     │
│  Note:   Must be done externally before using this pipeline                 │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 1: Generate Centroid JSON                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Script: straighten/location_json_local.py                                  │
│  Input:  Segmentation masks folder                                          │
│  Output: JSON files with vertebra centroids for each patient                │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 2: Straighten + De-Pedicle + Extract Volumes                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Script: straighten/straighten_mask_3d.py                                   │
│  Input:  CT folder, Label folder, Centroid JSON                             │
│  Output: Straightened CT volumes, Straightened label volumes                │
│          (de-pedicled, masked, normalized)                                  │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 3: Generate HGAM Attention Maps                                       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Script: Attention/grad_CAM_3d_sagittal.py                                  │
│  Input:  Straightened CT volumes, (Optional) trained classifier             │
│  Output: Grad-CAM++ heatmaps for attention guidance                         │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 4: GAN Training/Testing                                               │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Script: train.py / test.py                                                 │
│  Input:  Straightened volumes + HGAM heatmaps                               │
│  Output: Restored (healthy) vertebra predictions                            │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 5: RHLV Calculation                                                   │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Script: evaluation/RHLV_quantification.py                                  │
│  Input:  GAN output labels, Ground truth labels                             │
│  Output: RHLV values (Pre/Mid/Post regions)                                 │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 6: Fracture Grading (SVM)                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Script: evaluation/SVM_grading.py                                          │
│  Input:  RHLV Excel files                                                   │
│  Output: Genant grade classification metrics                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Files Modified

### 1. `straighten/location_json_local.py`

**Purpose:** Generate centroid JSON files from segmentation masks

**Hardcoded Paths Removed:**
```python
# BEFORE
root_dir = '/mnt/g/local_dataset/preprocessed/local'

# AFTER
parser.add_argument('--input-dir', type=str, required=True,
                    help='Root directory containing patient folders with segmentation masks')
```

**New Arguments:**
| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--input-dir` | str | Yes | - | Root directory with patient folders |
| `--min-voxels-top` | int | No | 150 | Min voxels for top vertebra |
| `--min-voxels-bottom` | int | No | 180 | Min voxels for bottom vertebra |

**Usage:**
```bash
python straighten/location_json_local.py \
    --input-dir datasets/raw
```

---

### 2. `straighten/straighten_mask_3d.py`

**Purpose:** Spine straightening, de-pedicle processing, and volume extraction

**Hardcoded Paths Removed:**
```python
# BEFORE
data_folder = "/dssg/home/acct-milesun/zhangqi/Dataset/HealthiVert_raw"
json_path = "/dssg/home/acct-milesun/zhangqi/Project/HealthiVert-GAN/vertebra_data_local.json"
output_folder = "/dssg/home/acct-milesun/zhangqi/Dataset/HealthiVert_straighten"

# AFTER
parser.add_argument('--data-folder', type=str, required=True, ...)
parser.add_argument('--json-path', type=str, required=True, ...)
parser.add_argument('--output-folder', type=str, required=True, ...)
```

**New Arguments:**
| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--data-folder` | str | Yes | - | Root folder with CT/ and label/ subfolders |
| `--json-path` | str | Yes | - | Path to vertebra_data.json |
| `--output-folder` | str | Yes | - | Output folder for straightened volumes |
| `--output-size` | int | No | 96 | Output volume size (cubic) |

**Usage:**
```bash
python straighten/straighten_mask_3d.py \
    --data-folder datasets/raw \
    --json-path vertebra_data.json \
    --output-folder datasets/straightened \
    --output-size 96
```

**Key Functions:**
- `remove_spine_labels_after_split()` - De-pedicle processing
- `extract_3d_volume()` - Extract and straighten vertebra volumes
- `process_mask3d()` - Main processing function

---

### 3. `Attention/grad_CAM_3d_sagittal.py`

**Purpose:** Generate Grad-CAM++ attention heatmaps using MONAI SEResNet50

**Hardcoded Paths Removed:**
```python
# BEFORE
ckpt_path = '/mnt/e/Graduation_project/classification/checkpoint.pth'
dataroot = '/mnt/e/Graduation_project/Vert_Dataset_straightened'
output_folder = '/mnt/e/Graduation_project/Vert_Dataset_straightened/heatmap_sagittal'

# AFTER
parser.add_argument('--ckpt-path', type=str, default=None, ...)
parser.add_argument('--dataroot', type=str, required=True, ...)
parser.add_argument('--output-folder', type=str, required=True, ...)
```

**New Arguments:**
| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--ckpt-path` | str | No | None | Path to trained classifier checkpoint |
| `--dataroot` | str | Yes | - | Root folder with CT/ subfolder |
| `--output-folder` | str | Yes | - | Output folder for heatmaps |
| `--target-class` | int | No | 1 | Target class for Grad-CAM (0=healthy, 1=fracture) |
| `--use-untrained` | flag | No | False | Use untrained model (for pipeline testing) |
| `--gpu` | int | No | 0 | GPU device ID |

**Usage (with trained model):**
```bash
python Attention/grad_CAM_3d_sagittal.py \
    --ckpt-path checkpoints/classifier.pth \
    --dataroot datasets/straightened \
    --output-folder datasets/straightened/heatmap \
    --target-class 1
```

**Usage (untrained for pipeline testing):**
```bash
python Attention/grad_CAM_3d_sagittal.py \
    --dataroot datasets/straightened \
    --output-folder datasets/straightened/heatmap \
    --use-untrained
```

⚠️ **Note:** Using `--use-untrained` will produce meaningless attention maps but allows testing the full pipeline flow.

---

### 4. `options/base_options.py`

**Purpose:** Central command-line options for GAN training/testing

**New Options Added:**
```python
# HealthiVert-specific parameters
parser.add_argument('--vertebra_json', type=str, 
                    default='vertebra_data.json',
                    help='path to vertebra data JSON file')
parser.add_argument('--cam_folder', type=str, 
                    default='heatmap',
                    help='folder name containing Grad-CAM heatmaps')
parser.add_argument('--vert_class', type=str, 
                    default='all',
                    help='vertebra class: thoracic, lumbar, or all')
```

---

### 5. `data/aligned_dataset.py`

**Purpose:** Dataset loader for coronal view training

**Hardcoded Paths Removed:**
```python
# BEFORE
with open('./vertebra_data.json', 'r') as file:
CAM_folder = os.path.join(self.dir_CT,'CAM_img')

# AFTER
with open(opt.vertebra_json, 'r') as file:
CAM_folder = os.path.join(self.dir_CT, opt.cam_folder)
```

**Added Fallback:**
```python
# Graceful fallback if CAM file doesn't exist
if os.path.exists(CAM_path):
    CAM_data = np.load(CAM_path)
else:
    CAM_data = np.zeros_like(CT_data)  # Use zeros if CAM not found
```

---

### 6. `data/aligned_dataset_sagittal.py`

**Purpose:** Dataset loader for sagittal view training

**Same modifications as aligned_dataset.py**

---

### 7. `evaluation/RHLV_quantification.py`

**Purpose:** Calculate Relative Height Loss of Vertebrae (three-region)

**Hardcoded Paths Removed:**
```python
# BEFORE
with open('vertebra_data_local.json', 'r') as file:
label_folder = '/dssg/home/acct-milesun/zhangqi/Dataset/HealthiVert_straighten/label'
output_folder = '/dssg/home/acct-milesun/zhangqi/Project/HealthiVert-GAN_eval/output'
result_folder = '/dssg/home/acct-milesun/zhangqi/Project/HealthiVert-GAN_eval/evaluation/RHLV_quantification'

# AFTER
parser.add_argument('--label-folder', type=str, required=True, ...)
parser.add_argument('--output-folder', type=str, required=True, ...)
parser.add_argument('--result-folder', type=str, required=True, ...)
parser.add_argument('--json-path', type=str, required=True, ...)
```

**New Arguments:**
| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--label-folder` | str | Yes | - | Ground truth label folder |
| `--output-folder` | str | Yes | - | GAN output folder with experiment subfolders |
| `--result-folder` | str | Yes | - | Folder to save RHLV Excel results |
| `--json-path` | str | Yes | - | Path to vertebra_data.json |
| `--length-divisor` | int | No | 5 | Divisor for analysis length |
| `--height-threshold` | float | No | 0.7 | Height filtering threshold |

**Usage:**
```bash
python evaluation/RHLV_quantification.py \
    --label-folder datasets/straightened/label \
    --output-folder results/output \
    --result-folder evaluation/RHLV_quantification \
    --json-path vertebra_data.json
```

---

### 8. `evaluation/SVM_grading.py`

**Purpose:** SVM classifier for Genant fracture grading

**Hardcoded Paths Removed:**
```python
# BEFORE
result_folder = 'evaluation/RHLV_quantification'
grading_folder = 'evaluation/classification_metric'
features = ['Pre RHLV', 'Mid RHLV', 'Post RHLV']

# AFTER
parser.add_argument('--rhlv-folder', type=str, default='evaluation/RHLV_quantification', ...)
parser.add_argument('--output-folder', type=str, default='evaluation/classification_metric', ...)
parser.add_argument('--features', type=str, nargs='+', default=[...], ...)
```

**New Arguments:**
| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--rhlv-folder` | str | No | evaluation/RHLV_quantification | Folder with RHLV Excel files |
| `--output-folder` | str | No | evaluation/classification_metric | Output folder for results |
| `--features` | list | No | Pre/Mid/Post RHLV | RHLV features to use |

**Usage:**
```bash
python evaluation/SVM_grading.py \
    --rhlv-folder evaluation/RHLV_quantification \
    --output-folder evaluation/classification_metric
```

---

## 🚀 Complete Pipeline Execution

### Prerequisites
1. **External Segmentation:** Run SCNet or similar to generate vertebra masks
2. **Python Environment:** Install requirements.txt

### Step-by-Step Commands

```bash
# Step 1: Generate centroid JSON files
python straighten/location_json_local.py \
    --input-dir datasets/raw

# Step 2: Straighten and extract volumes
python straighten/straighten_mask_3d.py \
    --data-folder datasets/raw \
    --json-path vertebra_data.json \
    --output-folder datasets/straightened

# Step 3: Generate HGAM attention maps (untrained for testing)
python Attention/grad_CAM_3d_sagittal.py \
    --dataroot datasets/straightened \
    --output-folder datasets/straightened/heatmap \
    --use-untrained

# Step 4: Train GAN (optional - skip for testing)
python train.py \
    --dataroot datasets/straightened \
    --vertebra_json vertebra_data.json \
    --cam_folder heatmap \
    --name healthivert_exp1

# Step 5: Test GAN
python test.py \
    --dataroot datasets/straightened \
    --vertebra_json vertebra_data.json \
    --cam_folder heatmap \
    --name healthivert_exp1

# Step 6: Calculate RHLV
python evaluation/RHLV_quantification.py \
    --label-folder datasets/straightened/label \
    --output-folder results/output \
    --result-folder evaluation/RHLV_quantification \
    --json-path vertebra_data.json

# Step 7: SVM Grading
python evaluation/SVM_grading.py \
    --rhlv-folder evaluation/RHLV_quantification \
    --output-folder evaluation/classification_metric
```

---

## 📊 Expected Folder Structure

```
HeathiVert/
├── datasets/
│   ├── raw/
│   │   ├── 0007/
│   │   │   ├── 0007.json          # Generated by Step 1
│   │   │   ├── CT.nii.gz          # Original CT
│   │   │   └── label.nii.gz       # Segmentation mask (from SCNet)
│   │   └── ...
│   └── straightened/
│       ├── CT/
│       │   ├── 0007_19.nii.gz     # Straightened CT volumes
│       │   └── ...
│       ├── label/
│       │   ├── 0007_19.nii.gz     # Straightened label volumes
│       │   └── ...
│       └── heatmap/
│           ├── 0007_19.npy        # HGAM attention maps
│           └── ...
├── vertebra_data.json              # Master vertebra metadata
├── train.py
├── test.py
└── ...
```

---

## ⚠️ Important Notes

### External Dependencies
1. **SCNet Segmentation:** Not included in codebase. Must generate masks externally.
2. **MONAI SEResNet50:** Requires training for meaningful Grad-CAM++ output.

### For Pipeline Testing (Untrained)
- Use `--use-untrained` flag in grad_CAM_3d_sagittal.py
- Attention maps will be random/meaningless
- GAN will still run but restoration quality will be poor

### File Format Requirements
- CT scans: NIfTI format (.nii.gz)
- Segmentation masks: NIfTI format with integer labels (1-24 for vertebrae)
- Attention maps: NumPy arrays (.npy)

---

## � Coronal vs Sagittal View Difference

The codebase has two dataset files: `aligned_dataset.py` (coronal) and `aligned_dataset_sagittal.py` (sagittal).

**Key Difference:** The axis indexing for slicing 3D volumes:

| View | Axis | Slicing | Index in `loc` |
|------|------|---------|----------------|
| **Sagittal** | Z-axis | `[:,:,z]` | `loc[2]` |
| **Coronal** | Y-axis | `[:,y,:]` | `loc[1]` |

**To use sagittal view:**
```bash
python train.py --dataroot ... --dataset_mode aligned_sagittal
```

The `--dataset_mode` argument controls which dataset class is loaded (see [data/__init__.py](data/__init__.py)).

---

## 🧪 Sample Test Mode

All preprocessing scripts now support `--sample-test N` to process only N samples for pipeline testing:

```bash
# Test preprocessing with only 2 patients
python straighten/location_json_local.py --input-dir datasets/raw --sample-test 2

# Test straightening with 3 patients per category (simple mode)
python straighten/straighten_mask_3d.py --data-folder datasets/raw --output-folder datasets/straightened --sample-test 3

# Test with VerSe dataset (see below)
python straighten/straighten_mask_3d.py --data-folder /path/to/verse19 --output-folder datasets/straightened --verse-mode --sample-test 2

# Test Grad-CAM with 5 files
python Attention/grad_CAM_3d_sagittal.py --dataroot datasets/straightened --output-folder heatmaps --use-untrained --sample-test 5
```

---

## 📂 Dataset Structure Support

### Mode 1: Simple Structure (Default)
All files for a patient in one folder:
```
data-folder/
└── patient_id/
    ├── patient_id.nii.gz       # CT volume
    ├── patient_id_msk.nii.gz   # Segmentation mask  
    └── patient_id.json         # Centroids
```

**Usage:**
```bash
python straighten/straighten_mask_3d.py \
    --data-folder datasets/raw \
    --output-folder datasets/straightened
```

### Mode 2: VerSe Dataset Structure (`--verse-mode`)
Separate rawdata/ and derivatives/ folders:
```
verse19/
├── dataset-verse19training/
│   ├── rawdata/
│   │   └── sub-verse004/
│   │       └── sub-verse004_ct.nii
│   └── derivatives/
│       └── sub-verse004/
│           ├── sub-verse004_seg-vert_msk.nii
│           └── sub-verse004_seg-vb_ctd.json
├── dataset-verse19validation/
│   ├── rawdata/
│   └── derivatives/
└── dataset-verse19test/
    ├── rawdata/
    └── derivatives/
```

**Usage:**
```bash
# Process entire VerSe dataset
python straighten/straighten_mask_3d.py \
    --data-folder /kaggle/input/verse-19-3d-images \
    --output-folder datasets/straightened \
    --verse-mode

# Test with 2 patients per split (training/validation/test)
python straighten/straighten_mask_3d.py \
    --data-folder /kaggle/input/verse-19-3d-images \
    --output-folder datasets/straightened \
    --verse-mode \
    --sample-test 2
```

The `--verse-mode` flag automatically:
- Scans for `dataset-verse19*` subdirectories
- Maps rawdata/ CT files to derivatives/ masks
- Handles split patients (e.g., `sub-verse414_split-verse241`)
- Categorizes into train/val/test based on folder names

---

## ⚠️ Dataset Naming Mismatch Issue

There is a **naming convention mismatch** between `vertebra_data.json` and the expected raw dataset structure.

### vertebra_data.json Format:
```json
{
  "test": {
    "sub-verse012_17": 0,    // patient_id = "sub-verse012", vertebra_id = 17
    "sub-verse012_18": 0,
    ...
  }
}
```

### Expected Raw Data Structure (by code):
```
data_folder/
└── sub-verse012/
    ├── sub-verse012.nii.gz      # CT volume
    ├── sub-verse012_msk.nii.gz  # Segmentation mask
    └── sub-verse012.json        # Centroids
```

### Actual VerSe Dataset Structure (from dataset.txt):
```
rawdata/
└── sub-verse012/
    └── sub-verse012_ct.nii      # Note: _ct suffix!

derivatives/
└── sub-verse012/
    ├── sub-verse012_seg-vert_msk.nii  # Different naming!
    └── sub-verse012_seg-vb_ctd.json
```

### Your Sample Data Structure (datasets/raw/0007):
```
0007/
├── 0007.nii.gz       # ✓ Works (matches expected)
├── 0007_msk.nii.gz   # ✓ Works
└── 0007.json         # ✓ Works
```

### Solution Options:

1. **Rename your files** to match the expected format:
   - `patient_id.nii.gz` (CT)
   - `patient_id_msk.nii.gz` or `patient_id_seg.nii.gz` (mask)
   - `patient_id.json` (centroids)

2. **Use `--sample-test` without vertebra_data.json:**
   The modified `straighten_mask_3d.py` can now scan the data folder directly if no JSON is provided:
   ```bash
   python straighten/straighten_mask_3d.py \
       --data-folder datasets/raw \
       --output-folder datasets/straightened \
       --sample-test 2
   ```

3. **Create a matching vertebra_data.json** for your test samples:
   ```json
   {
     "test": {
       "0007_19": 0,
       "0007_20": 1
     }
   }
   ```

---

## 📝 Modification Summary Table

| File | Hardcoded Paths | Status | New Entry Point |
|------|-----------------|--------|-----------------|
| `location_json_local.py` | 1 | ✅ Modified | `--input-dir`, `--sample-test` |
| `straighten_mask_3d.py` | 3 | ✅ Modified | `--data-folder`, `--json-path`, `--output-folder`, `--sample-test`, `--verse-mode` |
| `grad_CAM_3d_sagittal.py` | 3 | ✅ Modified | `--ckpt-path`, `--dataroot`, `--output-folder`, `--sample-test` |
| `base_options.py` | N/A | ✅ Modified | `--vertebra_json`, `--cam_folder` |
| `aligned_dataset.py` | 2 | ✅ Modified | Uses options from base_options |
| `aligned_dataset_sagittal.py` | 2 | ✅ Modified | Uses options from base_options |
| `RHLV_quantification.py` | 4 | ✅ Modified | `--label-folder`, `--output-folder`, `--result-folder`, `--json-path` |
| `SVM_grading.py` | 2 | ✅ Modified | `--rhlv-folder`, `--output-folder` |

---

## 🔧 Remaining Work

1. **Train SEResNet50 classifier:** For proper HGAM attention maps
2. **External Segmentation:** Obtain pre-segmented vertebra masks from SCNet
3. **Test complete pipeline:** End-to-end with sample data
4. **Verify naming conventions:** Ensure your data matches expected format

---

*Report generated for HealthiVert-GAN preprocessing workflow analysis*
