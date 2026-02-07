import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import nibabel as nib
from options.test_options import TestOptions
from models import create_model
import torchvision.transforms as transforms
from PIL import Image
from models.inpaint_networks import Generator
import torch.nn.functional as F
import math
from scipy.ndimage import label
import argparse

def remove_small_connected_components(input_array, min_size):


    structure = np.ones((3, 3), dtype=np.int32)  
    labeled, ncomponents = label(input_array, structure)

    for i in range(1, ncomponents + 1):
        if np.sum(labeled == i) < min_size:
            input_array[labeled == i] = 0


    return input_array

def load_model(model_path, netG_params, device, model_type='gan'):
    """Load generator model (GAN or Diffusion)."""
    if model_type == 'diffusion':
        from models.diffusion_generator import HealthiVertDiffusionUNet
        print(f"Loading Diffusion model from {model_path}")
        model = HealthiVertDiffusionUNet(cnum=32, T=1000)
    else:
        print(f"Loading GAN model from {model_path}")
        model = Generator(netG_params, True)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Model loaded successfully")
    else:
        print(f"⚠️  Warning: Model file not found at {model_path}")
    
    model.eval()
    model.to(device)
    return model

def numpy_to_pil(img_np):
    if img_np.dtype != np.uint8:
        raise ValueError("NumPy array should have uint8 data type.")
    img_pil = Image.fromarray(img_np)
    return img_pil

def run_model(model,CAM_data,label_data,ct_data,vert_id,index_ratio,A_transform,mask_transform,device,maxheight=40):
    vert_label_slice = np.zeros_like(label_data)
    vert_label_slice[label_data==vert_id]=1
    
    vert_label_slice = remove_small_connected_components(vert_label_slice,50)
    coords = np.argwhere(vert_label_slice)
    if coords.size==0:
        return None
    x1, x2 = min(coords[:, 0]), max(coords[:, 0])
    width,length = vert_label_slice.shape
    height = x2-x1
    if height>maxheight:
        x_mean = int(np.mean(coords[:, 0]))
        x1 = x_mean-20
        x2 = x1+40
                
    mask_x = (x1+x2)//2
    h2 = maxheight
    if mask_x<=h2//2:
        min_x = 0
        max_x = min_x + h2
    elif width-mask_x<=h2/2:
        max_x = width
        min_x = max_x -h2
    else:
        min_x = mask_x-h2//2
        max_x = min_x + h2
        
    mask_slice = np.zeros_like(vert_label_slice).astype(np.uint8)
    mask_slice[min_x:max_x+1] = 255
    ct_data_slice = np.zeros_like(mask_slice).astype(np.uint8)
    ct_data_slice[:min_x,:] = ct_data[(x1-min_x):x1,:]
    ct_data_slice[max_x:,:] = ct_data[x2:x2+(width-max_x),:]
                
    CAM_slice = np.zeros_like(mask_slice).astype(np.uint8)
    CAM_slice[:min_x,:] = CAM_data[(x1-min_x):x1,:]
    CAM_slice[max_x:,:] = CAM_data[x2:x2+(width-max_x),:]

    ct_batch = numpy_to_pil(ct_data_slice)
    ct_batch = A_transform(ct_batch)
                
    ori_ct = numpy_to_pil(ct_data.astype(np.uint8))
    ori_ct = A_transform(ori_ct)
                
    mask_batch = numpy_to_pil(mask_slice)
    mask_batch = mask_transform(mask_batch)
                
    CAM = numpy_to_pil(CAM_slice)
    CAM = mask_transform(CAM)

    ct_batch = ct_batch.unsqueeze(0).to(device)
    mask_batch = mask_batch.unsqueeze(0).to(device)
    CAM = CAM.unsqueeze(0).to(device)

    with torch.no_grad():
        _, fake_B_mask_sigmoid, _, fake_B_raw, _,_,pred_h = model(ct_batch, mask_batch, 1-CAM,index_ratio)
                    #print(pred_h)
    pred_h = math.ceil(pred_h[0]*maxheight)

    fake_B_mask_raw = torch.where(fake_B_mask_sigmoid > 0.5, torch.ones_like(fake_B_mask_sigmoid), torch.zeros_like(fake_B_mask_sigmoid))
    #fake_B_mask_raw = fake_B_mask_raw.squeeze().cpu().numpy()*int(vert_id)

    if pred_h<height:
        pred_h = height
    height_diff = pred_h-height
    x_upper = x1-height_diff//2
    x_bottom = x_upper+pred_h
    single_image = torch.zeros_like(fake_B_raw)
    single_image[:,:,x_upper:x_bottom,:] = fake_B_raw[:,:,x_upper:x_bottom,:]
    ct_upper = torch.zeros_like(single_image)
    ct_upper[0,:,:x_upper,:] = ori_ct[:, height_diff//2:x1, :]
    ct_bottom = torch.zeros_like(single_image)
    ct_bottom[0,:,x_bottom:,:] = ori_ct[:, x2:x2+256-x_bottom, :]
    interpolated_image = single_image+ct_upper+ct_bottom
    fake_B = interpolated_image.squeeze().cpu().numpy()
    fake_B = (fake_B+1)*127.5
    
    mid_seg = np.zeros_like(fake_B_mask_raw.squeeze().cpu().numpy())
    mid_seg[x_upper:x_bottom,:] = fake_B_mask_raw[:,:,x_upper:x_bottom,:].squeeze().cpu().numpy()*vert_id
    seg_upper = np.zeros_like(mid_seg)
    seg_upper[:x_upper,:] = label_data[height_diff//2:x1, :]
    seg_bottom = np.zeros_like(mid_seg)
    seg_bottom[x_bottom:,:] = label_data[x2:x2+256-x_bottom, :]
    interpolated_seg = mid_seg+seg_upper+seg_bottom
    fake_B_mask_raw = interpolated_seg
    
    
    return fake_B_mask_raw,fake_B,height
    

def process_nii_files(folder_path, CAM_folder, model, output_folder, device, max_samples=None):
    A_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if not os.path.exists(os.path.join(output_folder, 'CT')):
        os.makedirs(os.path.join(output_folder, 'CT'))
    if not os.path.exists(os.path.join(output_folder, 'label')):
        os.makedirs(os.path.join(output_folder, 'label'))

    count = 0
    for file_name in os.listdir(folder_path):
        # Stop after reaching max_samples limit
        if max_samples is not None and count >= max_samples:
            print(f"✅ Reached maximum sample limit ({max_samples}). Stopping processing.")
            break
        #if file_name!="sub-verse004_ct.nii_16.nii":
        #    continue
        if file_name.endswith('.nii.gz') or file_name.endswith('.nii'):  # Accept both formats
            if os.path.exists(os.path.join(output_folder, 'CT_fake', file_name)):
                continue
            
            # Strip file extension properly
            if file_name.endswith('.nii.gz'):
                base_name = file_name[:-7]  # Remove .nii.gz (7 chars)
            else:
                base_name = file_name[:-4]  # Remove .nii (4 chars)
            
            # Extract vertebra ID from format: sub-verse004_ct.nii_16 or sub-verse004_16
            try:
                parts = base_name.rsplit('_', 1)
                vert_id = int(parts[1])
                patient_id = parts[0]
            except (ValueError, IndexError):
                print(f"⚠️  Skipping {file_name}: cannot parse vertebra ID")
                continue
            
            file_path = os.path.join(folder_path, file_name)
            label_path = file_path.replace('CT', 'label')
            ct_nii = nib.load(file_path)
            ct_data = ct_nii.get_fdata()
            label_nii = nib.load(label_path)
            label_data = label_nii.get_fdata()

            # Build CAM paths - try multiple naming conventions (.nii.gz and .nii)
            CAM_paths = [
                os.path.join(CAM_folder, base_name + '_0.nii.gz'),
                os.path.join(CAM_folder, base_name + '_0.nii'),
                os.path.join(CAM_folder, base_name + '_1.nii.gz'),
                os.path.join(CAM_folder, base_name + '_1.nii'),
                os.path.join(CAM_folder, base_name + '.nii.gz'),
                os.path.join(CAM_folder, base_name + '.nii'),
            ]
            
            CAM_path = None
            for path in CAM_paths:
                if os.path.exists(path):
                    CAM_path = path
                    break
            
            if CAM_path is None:
                print(f"⚠️  No CAM file found for {file_name}, skipping...")
                continue
            
            #print(CAM_path)
            CAM_data = nib.load(CAM_path).get_fdata() * 255
            
            vert_label = np.zeros_like(label_data)
            vert_label[label_data==vert_id]=1
            
            loc = np.where(vert_label)
    
            z0 = min(loc[2])
            z1 = max(loc[2])
            range_length = z1 - z0 + 1
            new_range_length = int(range_length * 4 / 5)
            new_z0 = z0 + (range_length - new_range_length) // 2
            new_z1 = new_z0 + new_range_length - 1

            output_ct_data = np.zeros_like(ct_data)
            output_seg_data = np.zeros_like(ct_data)
            center_index = (new_z0 + new_z1) // 2
            
            maxheight = 40
            
            for z in range(new_z0, new_z1 + 1):
                index_ratio = abs(z-center_index)/range_length*2
                index_ratio = torch.tensor([index_ratio])
                if int(vert_id)>8 and np.sum(label_data[:, :, z]==int(vert_id)-1)>200:
                    #print("upper exists and sum=",np.sum(label_data[:, :, z]==int(vert_id)-1))
                    vert_id_upper = int(vert_id)-1
                    #print("upper exists")
                    fake_B_mask_upper,fake_B_ct_upper,_ = run_model(model,CAM_data[:, :, z],label_data[:, :, z],ct_data[:, :, z],vert_id_upper,index_ratio,\
                    A_transform,mask_transform,device,maxheight)
                else:
                    fake_B_mask_upper,fake_B_ct_upper = label_data[:, :, z],ct_data[:, :, z]
                    #print("upper dont exists and sum=",np.sum(label_data[:, :, z]==int(vert_id)-1))
                if int(vert_id)<24 and np.sum(label_data[:, :, z]==int(vert_id)+1)>200:
                    #print("bottom exists and sum=",np.sum(label_data[:, :, z]==int(vert_id)+1))
                    vert_id_bottom = int(vert_id)+1
                    #print("bottom exists")
                    fake_B_mask_bottom,fake_B_ct_bottom,_ = run_model(model,CAM_data[:, :, z],fake_B_mask_upper,fake_B_ct_upper,vert_id_bottom,index_ratio,\
                    A_transform,mask_transform,device,maxheight)
                else:
                    fake_B_mask_bottom,fake_B_ct_bottom = fake_B_mask_upper,fake_B_ct_upper
                    #print("bottom dont exists and sum=",np.sum(label_data[:, :, z]==int(vert_id)+1))

                    
                output = run_model(model,CAM_data[:, :, z],fake_B_mask_bottom,fake_B_ct_bottom,int(vert_id),index_ratio,\
                    A_transform,mask_transform,device,maxheight)
                if output==None:
                    continue
                else:
                    fake_B_mask_raw,fake_B,height = output
                    if height>maxheight:
                        print("Height exceeds in %s, in slice %d"%(file_name,z))
                
                output_seg_data[:, :, z] = fake_B_mask_raw
                output_ct_data[:, :, z] = fake_B

            new_ct_nii = nib.Nifti1Image(output_ct_data, ct_nii.affine)
            nib.save(new_ct_nii, os.path.join(output_folder, 'CT_fake', file_name))
            new_label_nii = nib.Nifti1Image(output_seg_data, ct_nii.affine)
            nib.save(new_label_nii, os.path.join(output_folder, 'label_fake', file_name))
            print(f"Now {file_name} has been generateed in {output_folder}")
            count+=1
            

def parse_args():
    parser = argparse.ArgumentParser(description='3D vertebra reconstruction with two-stage generation')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--ct-folder', type=str, required=True,
                        help='Path to input CT folder (e.g., verse19/straighten/CT)')
    parser.add_argument('--cam-folder', type=str, required=True,
                        help='Path to Grad-CAM heatmap folder')
    parser.add_argument('--output-folder', type=str, required=True,
                        help='Path to output folder for generated volumes')
    parser.add_argument('--model-type', type=str, default='gan', choices=['gan', 'diffusion'],
                        help='Model type: gan (two-stage GAN) or diffusion (DDPM-based)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (default: 0, use -1 for CPU)')
    parser.add_argument('--ngf', type=int, default=16,
                        help='Number of generator filters (default: 16)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process (default: None = all samples)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Model parameters
    netG_params = {'input_dim': 1, 'ngf': args.ngf}
    
    # Create output directories
    if not os.path.exists(os.path.join(args.output_folder, 'CT_fake')):
        os.makedirs(os.path.join(args.output_folder, 'CT_fake'))
    if not os.path.exists(os.path.join(args.output_folder, 'label_fake')):
        os.makedirs(os.path.join(args.output_folder, 'label_fake'))
    
    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Model path: {args.model_path}")
    print(f"Input CT folder: {args.ct_folder}")
    print(f"CAM folder: {args.cam_folder}")
    print(f"Output folder: {args.output_folder}")
    if args.max_samples:
        print(f"Max samples to process: {args.max_samples}")

    model = load_model(args.model_path, netG_params, device, model_type=args.model_type)
    process_nii_files(args.ct_folder, args.cam_folder, model, args.output_folder, device, args.max_samples)
    
    print(f"\n✅ 3D Reconstruction Complete!")
    print(f"   Generated volumes saved to: {os.path.join(args.output_folder, 'CT_fake')}")
    print(f"   Generated labels saved to: {os.path.join(args.output_folder, 'label_fake')}")

if __name__ == "__main__":
    main()
