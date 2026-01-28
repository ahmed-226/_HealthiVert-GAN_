"""
Step 2: Generate Grad-CAM++ attention heatmaps for HGAM module
This script generates attention maps using a pre-trained classifier to guide
the generator to focus on healthy vertebra regions.

Usage:
    python grad_CAM_3d_sagittal.py --ckpt-path <classifier_checkpoint> --dataroot <straightened_CT_path> --output-folder <heatmap_output>
    
    For untrained model (testing pipeline only):
    python grad_CAM_3d_sagittal.py --dataroot <straightened_CT_path> --output-folder <heatmap_output> --use-untrained

Expected input:
    dataroot/
        CT/
            patient_vertebra.nii.gz

Output:
    output-folder/
        patient_vertebra.nii.gz  (3D heatmap)
        patient_vertebra.png     (visualization)
"""

import torch
import torch.nn.functional as F
from torch.autograd import Function
# Commented out - these are custom modules that may not exist
# from model import Seresnet50_Contrastive
# from utils import CustomLogger, calculate_confusion_matrix
import os
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from monai.networks.nets import SEresnet50
from pathlib import Path
import nibabel as nib
import random
import argparse

class GradCam:
    def __init__(self, model, feature_layer):
        self.model = model
        self.feature_layer = feature_layer
        self.gradients = None
        self.model.eval()

        # 注册钩子
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.features = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        # 获取目标层
        for name, module in self.model.named_modules():
            if name == self.feature_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class):
        output = self.model(input_image)
        if isinstance(output, tuple):
            output = output[0]

        # 获取目标类别的得分
        score = output[:, target_class]

        # 反向传播，获取梯度
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # 根据梯度和特征图计算权重
        gradients = self.gradients.data
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        features = self.features.data
        for i in range(pooled_gradients.size(0)):
            features[:, i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(features, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)

        return heatmap.cpu().numpy()


class GradCamPlusPlus(GradCam):
    def generate_cam(self, input_image, target_class):
        output = self.model(input_image)
        if isinstance(output, tuple):
            output = output[0]

        # 获取目标类别的得分
        score = output[:, target_class]

        # 反向传播，获取梯度
        self.model.zero_grad()
        score.backward(retain_graph=True)

        gradients = self.gradients.data
        gradient_power_2 = gradients**2
        gradient_power_3 = gradients**3

        global_sum = torch.sum(self.features.data, dim=[2, 3], keepdim=True)
        alpha_num = gradient_power_2
        alpha_denom = 2 * gradient_power_2 + global_sum * gradient_power_3 + 1e-7
        alpha = alpha_num / alpha_denom
        alpha = alpha.where(alpha_denom != 0, torch.zeros_like(alpha))

        positive_gradients = F.relu(score.exp() * gradients)
        weights = (alpha * positive_gradients).sum(dim=[2, 3], keepdim=True)

        cam = (weights * self.features.data).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam).data

        cam = cam.squeeze()
        heatmap = cam.cpu().detach().numpy()

        return heatmap

    
def get_img_with_preprocess(img, transform):
    img_arr = img.get_fdata()
    z_center = int(img_arr.shape[2] / 2)
    slices = range(max(0, z_center - 15), min(img_arr.shape[2], z_center + 15))
    output_imgs = []
    output_tensors = []

    for slice in slices:
        output_img = img_arr[:, :, slice]
        output_img = output_img.astype(np.uint8)

        output_tensor = np.expand_dims(output_img.copy(), axis=-1)
        output_tensor = transform(output_tensor)
        output_imgs.append(output_img)
        output_tensors.append(output_tensor.unsqueeze(0))

    return output_imgs, torch.cat(output_tensors, 0)

    
def apply_heatmap_to_grayscale_and_save(heatmap, image, save_path, alpha=0.4, colormap=cv2.COLORMAP_JET):
    # 确保image是float32类型
    img_gray = image.astype(np.float32)
    
    # 将灰度图像转换为三通道的彩色图像
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    
    # 将热力图调整为原图大小
    heatmap = cv2.resize(heatmap, (img_color.shape[1], img_color.shape[0]))
    
    # 确保heatmap也是float32类型
    heatmap = np.uint8(255 * heatmap.astype(np.float32))

    heatmap = cv2.applyColorMap(heatmap, colormap)

    # 叠加热力图与原图
    superimposed_img = heatmap * alpha + img_color
    superimposed_img = superimposed_img / np.max(superimposed_img) * 255
    superimposed_img = np.uint8(superimposed_img)
    
    # 保存图像
    cv2.imwrite(save_path, superimposed_img)
    print(f"Image saved to {save_path}")


def find_ct_folder(dataroot):
    """
    Intelligently find the CT folder regardless of input path.
    Handles: /path/to/CT, /path/to/straighten, /path/to/straighten/CT
    """
    dataroot = Path(dataroot).resolve()
    
    print(f"[Path Detection] Input: {dataroot}")
    
    # Case 1: User passed .../straighten/CT directly
    if dataroot.name == 'CT' and dataroot.parent.name == 'straighten':
        print(f"[Path Detection] Found CT folder (Case 1): {dataroot}")
        return dataroot
    
    # Case 2: User passed .../straighten
    if dataroot.name == 'straighten' and (dataroot / 'CT').exists():
        ct_path = dataroot / 'CT'
        print(f"[Path Detection] Found CT folder (Case 2): {ct_path}")
        return ct_path
    
    # Case 3: User passed root dataset folder
    if (dataroot / 'straighten' / 'CT').exists():
        ct_path = dataroot / 'straighten' / 'CT'
        print(f"[Path Detection] Found CT folder (Case 3): {ct_path}")
        return ct_path
    
    # Case 4: User passed .../CT directly (any level)
    if dataroot.name == 'CT' and dataroot.exists():
        print(f"[Path Detection] Found CT folder (Case 4): {dataroot}")
        return dataroot
    
    # Fallback: Try to find CT folder recursively
    for ct_candidate in dataroot.rglob('CT'):
        if ct_candidate.is_dir() and any(ct_candidate.glob('*.nii.gz')):
            print(f"[Path Detection] Found CT folder (Case 5 - recursive): {ct_candidate}")
            return ct_candidate
    
    # If nothing found, raise error
    raise FileNotFoundError(
        f"Cannot find CT folder in: {dataroot}\n"
        f"Expected one of:\n"
        f"  - {dataroot}/CT\n"
        f"  - {dataroot}/straighten/CT\n"
        f"  - {dataroot}/straighten/CT (if dataroot is root)\n"
        f"Please ensure the path contains straightened CT files."
    )



def process_and_save_nii(ct_folder, output_folder, grad_cam, target_class=1, sample_limit=None):
    """
    Process all NIfTI files in ct_folder and save heatmaps.
    """
    ct_folder = Path(ct_folder)
    nii_files = sorted(list(ct_folder.glob('*.nii.gz')))
    
    if not nii_files:
        raise FileNotFoundError(f"No .nii.gz files found in {ct_folder}")
    
    print(f"[Processing] Found {len(nii_files)} NIfTI files")
    
    # Apply sample limit if specified
    if sample_limit is not None and sample_limit < len(nii_files):
        nii_files = nii_files[:sample_limit]
        print(f"[Sample Test Mode] Processing only {len(nii_files)} files")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    for idx, img_nii_path in enumerate(nii_files, 1):
        print(f"\n[{idx}/{len(nii_files)}] Processing: {img_nii_path.name}")
        
        # Extract filename WITHOUT extensions
        # For "sub-verse004_ct.nii_16.nii.gz", we want "sub-verse004_ct.nii_16"
        filename = img_nii_path.name
        # Remove the last .gz
        filename = filename.replace('.nii.gz', '')
        # If there's another .nii, remove it too (for cases like .nii.nii.gz)
        if filename.endswith('.nii'):
            filename = filename.replace('.nii', '')
        
        print(f"[Filename] {filename}")
        
        img_nii = nib.load(str(img_nii_path))
        img_arr = img_nii.get_fdata()
        
        # Validate image
        if img_arr.shape[2] < 30:  # Not enough slices
            print(f"[Warning] Image has only {img_arr.shape[2]} slices, may have issues")
        
        image_raw_list, input_tensor = get_img_with_preprocess(img_nii, transform)
        input_tensor = input_tensor.cuda(0, non_blocking=True).float()

        # Initialize 3D heatmap array
        heatmap_3d = np.zeros(img_nii.get_fdata().shape)
        
        z_center = int(img_nii.shape[2] / 2)
        start_slice = max(0, z_center - 15)
        end_slice = min(img_nii.shape[2], z_center + 15)
        
        print(f"[Slices] Processing slices {start_slice} to {end_slice} (center={z_center})")
        
        for i, slice_idx in enumerate(range(start_slice, end_slice)):
            input_slice = input_tensor[i:i+1]
            heatmap = grad_cam.generate_cam(input_slice, target_class)
            heatmap_resized = cv2.resize(heatmap, (img_nii.shape[0], img_nii.shape[1]))
            heatmap_3d[:, :, slice_idx] = heatmap_resized

        # Save as NIfTI format
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        save_path = os.path.join(output_folder, filename + '.nii.gz')
        new_img_nii = nib.Nifti1Image(heatmap_3d, img_nii.affine, img_nii.header)
        nib.save(new_img_nii, save_path)
        print(f"[Saved] NIfTI: {save_path}")
        
        # Save visualization
        imgsave_path = os.path.join(output_folder, filename + '.png')
        apply_heatmap_to_grayscale_and_save(
            heatmap_3d[:, :, z_center], 
            img_arr[:, :, z_center], 
            imgsave_path
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM++ attention heatmaps for HGAM module')
    parser.add_argument('--ckpt-path', type=str, default=None,
                        help='Path to classifier checkpoint file (.tar or .pkl). If not provided with --use-untrained, will use random weights.')
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to straightened CT data folder (can be /path/to/CT, /path/to/straighten, or /path/to/dataset)')
    parser.add_argument('--output-folder', type=str, required=True,
                        help='Path to output folder for heatmaps')
    parser.add_argument('--target-class', type=int, default=1,
                        help='Target class for Grad-CAM (0=healthy, 1=fractured, default: 1)')
    parser.add_argument('--use-untrained', action='store_true',
                        help='Use untrained MONAI SEResNet50 (for pipeline testing only, results will be meaningless)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (default: 0)')
    parser.add_argument('--sample-test', type=int, default=None,
                        help='Sample test mode: process only N files (for pipeline testing)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    torch.cuda.set_device(args.gpu)
    
    # Initialize model
    model = SEresnet50(spatial_dims=2, in_channels=1, num_classes=2)
    model = torch.nn.DataParallel(model).cuda()
    
    # Load checkpoint if provided
    if args.ckpt_path and not args.use_untrained:
        print(f"Loading checkpoint from: {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path, map_location=torch.device('cuda', args.gpu))
        
        # Handle different checkpoint formats
        if args.ckpt_path.endswith('.tar'):
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        elif args.ckpt_path.endswith('.pkl'):
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        print("Checkpoint loaded successfully.")
    elif args.use_untrained:
        print("WARNING: Using untrained model. Heatmaps will be random/meaningless!")
        print("This is only for testing the pipeline. Train a classifier for real results.")
    else:
        print("WARNING: No checkpoint provided and --use-untrained not set.")
        print("Using untrained model. Heatmaps will be random/meaningless!")
    
    # Setup Grad-CAM++
    target_layers = [model.module.layer4[-1]]
    grad_cam = GradCamPlusPlus(model=model, feature_layer="module.layer4.2.conv1.conv")
    
    # Find CT folder automatically
    try:
        ct_folder = find_ct_folder(args.dataroot)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        exit(1)
    
    # Process and save
    print(f"\n{'='*60}")
    print(f"Starting Grad-CAM heatmap generation")
    print(f"{'='*60}")
    process_and_save_nii(ct_folder, args.output_folder, grad_cam, 
                         target_class=args.target_class, sample_limit=args.sample_test)
    print(f"\n{'='*60}")
    print(f"✅ Heatmaps saved to: {args.output_folder}")
    print(f"{'='*60}")