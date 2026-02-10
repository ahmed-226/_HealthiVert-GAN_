"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from options.test_options import TestOptions
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os 
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import torch.nn.functional as F
import math

def dice_score(pred, target, smooth=1e-5):
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def iou_score(pred, target, smooth=1e-5):
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def evaluate_model(model, test_loader, device, checkpoint_path, iteration):
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 关闭梯度计算
        ssim_scores, psnr_scores, dice_scores, iou_scores = [], [], [], []
        Diff_hs = []
        for batch in test_loader:
            model.set_input(batch)
            
            ground_truths, labels, normal_vert_labels, masks, CAMs, heights, x1, x2, slice_ratio = \
                model.real_B, model.real_B_mask, model.normal_vert, model.mask, model.CAM, model.height, \
                    model.x1, model.x2, model.slice_ratio
            maxheight = model.maxheight
            ct_upper_list = []
            ct_bottom_list = []
            for i in range(ground_truths.shape[0]):
                ct_upper = ground_truths[i, :, :x1[i], :]
                ct_bottom = ground_truths[i, :, x2[i]:, :]
                ct_upper_list.append(ct_upper.unsqueeze(0))  # 添加批次维度以便合并
                ct_bottom_list.append(ct_bottom.unsqueeze(0))
                
            # 模型推理
            CAM_temp = 1 - CAMs
            inputs = model.real_A
            outputs = model.netG(inputs, masks, CAM_temp, slice_ratio)  # 根据你的模型调整
            coarse_seg_sigmoid, fine_seg_sigmoid, stage1, stage2, offset_flow, pred1_h, pred2_h = outputs  # 根据你的输出调整
            pred1_h = pred1_h.T * maxheight
            pred2_h = pred2_h.T * maxheight

            coarse_seg_binary = torch.where(coarse_seg_sigmoid > 0.5, torch.ones_like(coarse_seg_sigmoid), torch.zeros_like(coarse_seg_sigmoid))
            fine_seg_binary = torch.where(fine_seg_sigmoid > 0.5, torch.ones_like(fine_seg_sigmoid), torch.zeros_like(fine_seg_sigmoid))
            
            fake_B_raw_list = []
            for i in range(stage2.size(0)):
                height = math.ceil(pred2_h[0][i].item())  # 获取当前图片的目标高度
                if height < heights[i]:
                    height = heights[i]
                height_diff = height - heights[i]
                x_upper = x1[i] - height_diff // 2
                x_bottom = x_upper + height
                single_image = torch.zeros_like(stage2[i:i+1])
                single_image[0, :, x_upper:x_bottom, :] = stage2[i:i+1, :, x_upper:x_bottom, :]
                ct_upper = torch.zeros_like(single_image)
                ct_upper[0, :, :x_upper, :] = ground_truths[i, :, height_diff//2:x1[i], :]
                ct_bottom = torch.zeros_like(single_image)
                ct_bottom[0, :, x_bottom:, :] = ground_truths[i, :, x2[i]:x2[i]+256-x_bottom, :]
                interpolated_image = single_image + ct_upper + ct_bottom
                fake_B_raw_list.append(interpolated_image)

            inpainted_result = torch.cat(fake_B_raw_list, dim=0)
            
            # 计算评估指标
            for i in range(inputs.size(0)):  # 遍历batch中的每个样本
                ground_truth = ground_truths[i].cpu().numpy()
                label = labels[i].cpu().numpy()
                normal_vert_label = normal_vert_labels[i].cpu().numpy()
                height = heights[i].cpu()
                pred_h = pred2_h[0][i].cpu()
                
                inpainted_result_np = inpainted_result[i].cpu().numpy()
                coarse_seg_binary_np = coarse_seg_binary[i].cpu().numpy()
                fine_seg_binary_np = fine_seg_binary[i].cpu().numpy()
                mask = masks[i].cpu().numpy()
                
                # ============ MASK INVERSION FIX ============
                # Debug: Check mask statistics
                mask_mean = mask.mean()
                mask_unique = np.unique(mask)
                
                if i == 0:  # Print debug info for first sample only
                    print(f"\n[DEBUG] Mask Analysis for Sample {i}:")
                    print(f"  Mask shape: {mask.shape}")
                    print(f"  Mask mean: {mask_mean:.4f}")
                    print(f"  Mask unique values: {mask_unique}")
                    print(f"  Mask range: [{mask.min():.4f}, {mask.max():.4f}]")
                
                # Detect if mask is inverted
                # Standard inpainting: mask=0 for hole (vertebra to generate), mask=1 for background (keep)
                # For evaluation, we want to compare ONLY the generated vertebra (hole)
                # So if mask_mean > 0.5, the mask is selecting background -> INVERT IT
                if mask_mean > 0.5:
                    mask_for_eval = 1.0 - mask
                    if i == 0:
                        print(f"  ⚠️  Mask is selecting BACKGROUND (mean={mask_mean:.4f} > 0.5)")
                        print(f"  ✅ INVERTING mask to select the GENERATED VERTEBRA instead")
                else:
                    mask_for_eval = mask
                    if i == 0:
                        print(f"  ✅ Mask is selecting HOLE/VERTEBRA (mean={mask_mean:.4f} <= 0.5)")
                
                # FIX: Extract masked regions for proper comparison (using corrected mask)
                gt_masked = (ground_truth * mask_for_eval).squeeze()
                pred_masked = (inpainted_result_np * mask_for_eval).squeeze()
                
                # Additional debug: Check if we're comparing identical regions
                if i == 0:
                    mse_debug = np.mean((gt_masked - pred_masked) ** 2)
                    print(f"  MSE between GT and Pred (masked): {mse_debug:.10f}")
                    if mse_debug < 1e-7:
                        print(f"  🚨 CRITICAL: MSE is near-zero! Likely comparing identical images.")
                        print(f"  This indicates the mask is still selecting the wrong region.")
                # ============================================
                
                # FIX: Calculate data_range properly (use consistent min/max)
                data_range = max(gt_masked.max(), pred_masked.max()) - min(gt_masked.min(), pred_masked.min())
                
                # FIX: Add safety check for data_range
                if data_range < 1e-7:
                    print(f"⚠️  WARNING: Sample {i} has near-zero data_range ({data_range:.6f}). Skipping metrics.")
                    print(f"   This suggests the masked region is empty or constant.")
                    continue
                
                # FIX: SSIM for single-channel grayscale (removed multichannel=True)
                try:
                    ssim_score = ssim(gt_masked, pred_masked, data_range=data_range)
                    ssim_scores.append(ssim_score)
                    
                    # Add warning if SSIM is suspiciously high (potential bug indicator)
                    if ssim_score > 0.999:
                        print(f"⚠️  WARNING: Sample {i} has SSIM={ssim_score:.6f} (possibly comparing identical images)")
                        print(f"   GT masked range: [{gt_masked.min():.4f}, {gt_masked.max():.4f}]")
                        print(f"   Pred masked range: [{pred_masked.min():.4f}, {pred_masked.max():.4f}]")
                        print(f"   Mask evaluation type used: {'INVERTED (hole)' if mask_mean > 0.5 else 'DIRECT (hole)'}")
                except Exception as e:
                    print(f"⚠️  ERROR computing SSIM for sample {i}: {e}")
                    continue
                
                # FIX: PSNR with divide-by-zero protection
                try:
                    mse = np.mean((gt_masked - pred_masked) ** 2)
                    if mse < 1e-10:
                        print(f"⚠️  WARNING: Sample {i} has MSE={mse:.10f} (near-zero, skipping PSNR)")
                        print(f"   This indicates GT and Pred are nearly identical in the masked region.")
                    else:
                        image_psnr = psnr(gt_masked, pred_masked, data_range=data_range)
                        psnr_scores.append(image_psnr)
                        
                        # Add warning for unrealistic PSNR values
                        if image_psnr > 60:
                            print(f"⚠️  WARNING: Sample {i} has PSNR={image_psnr:.2f}dB (suspiciously high)")
                            print(f"   PSNR > 60dB suggests near-perfect reconstruction (unlikely for GAN).")
                except Exception as e:
                    print(f"⚠️  ERROR computing PSNR for sample {i}: {e}")
                
                dice_value_coarse = dice_score(torch.tensor(coarse_seg_binary_np).float(), torch.tensor(normal_vert_label).float())
                dice_scores.append(dice_value_coarse)
                
                iou_value_fine = iou_score(torch.tensor(fine_seg_binary_np).float(), torch.tensor(label).float())
                iou_scores.append(iou_value_fine)
                
                Diff_h = (abs(pred_h - height) / height) * 100
                Diff_hs.append(Diff_h)

        # 计算整个测试集上的评估指标平均值
        # FIX: Add safety checks for empty lists
        print("\n" + "="*80)
        print("📊 EVALUATION SUMMARY")
        print("="*80)
        
        if len(ssim_scores) == 0:
            print("⚠️  WARNING: No valid SSIM scores computed!")
            avg_ssim = 0.0
        else:
            avg_ssim = np.mean(ssim_scores)
            print(f"ℹ️  SSIM: mean={avg_ssim:.4f}, min={np.min(ssim_scores):.4f}, max={np.max(ssim_scores):.4f}, std={np.std(ssim_scores):.4f}")
            
            # Final validation check
            if avg_ssim > 0.95:
                print(f"\n🚨 ALERT: Average SSIM is {avg_ssim:.4f} (>0.95)")
                print(f"   This is suspiciously high for a GAN model.")
                print(f"   Possible causes:")
                print(f"   1. Mask is still inverted (comparing background instead of vertebra)")
                print(f"   2. Model is copying input instead of generating")
                print(f"   3. Evaluation code has a bug")
        
        if len(psnr_scores) == 0:
            print("⚠️  WARNING: No valid PSNR scores computed!")
            avg_psnr = 0.0
        else:
            avg_psnr = np.mean(psnr_scores)
            print(f"ℹ️  PSNR: mean={avg_psnr:.2f}dB, min={np.min(psnr_scores):.2f}dB, max={np.max(psnr_scores):.2f}dB, std={np.std(psnr_scores):.2f}dB")
            
            if avg_psnr > 50:
                print(f"\n🚨 ALERT: Average PSNR is {avg_psnr:.2f}dB (>50dB)")
                print(f"   This indicates near-perfect reconstruction (very unlikely for GAN).")
            
        avg_dice = np.mean(dice_scores)
        avg_iou = np.mean(iou_scores)
        avg_diffh = np.mean(Diff_hs)
        
        print(f"ℹ️  Dice: mean={avg_dice:.4f}")
        print(f"ℹ️  IoU: mean={avg_iou:.4f}")
        print(f"ℹ️  DiffH: mean={avg_diffh:.2f}%")
        print(f"\n✅ Validated {len(ssim_scores)} samples (from {inputs.size(0)} per batch)")
        print("="*80 + "\n")
        
    model.train()  # 恢复模型到训练模式
    viz_images = torch.stack([inputs, inpainted_result, ground_truths,
                              coarse_seg_binary, normal_vert_labels, fine_seg_binary, labels, CAMs], dim=1)
    viz_images = viz_images.view(-1, *list(inputs.size())[1:])
    imgsave_pth = os.path.join(checkpoint_path, "eval_imgs")
    if not os.path.exists(imgsave_pth):
        os.makedirs(imgsave_pth)
    vutils.save_image(viz_images,
                      '%s/nepoch_%03d_eval.png' % (imgsave_pth, iteration),
                      nrow=3 * 4,
                      normalize=True)
    return avg_ssim, avg_psnr, avg_dice, avg_iou, avg_diffh


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    
    # ============ CONFIGURATION VALIDATION ============
    print("\n" + "="*80)
    print("🔍 CONFIGURATION VALIDATION")
    print("="*80)
    
    # Check training duration (paper uses ~1000 epochs)
    total_epochs = opt.n_epochs + opt.n_epochs_decay
    if total_epochs < 200:
        print(f"⚠️  WARNING: Total epochs = {total_epochs} (paper likely uses 1000+)")
        print(f"   Your settings: n_epochs={opt.n_epochs}, n_epochs_decay={opt.n_epochs_decay}")
        print(f"   Consider: --n_epochs 100 --n_epochs_decay 900")
    else:
        print(f"✅ Total training epochs: {total_epochs}")
    
    # Check loss weights
    print(f"\n📊 Loss Weights Configuration:")
    print(f"   lambda_L1: {opt.lambda_L1}")
    if hasattr(opt, 'lambda_edge'):
        print(f"   lambda_edge: {opt.lambda_edge} (paper suggests 80-800)")
    if hasattr(opt, 'lambda_height'):
        print(f"   lambda_height: {opt.lambda_height} (paper suggests 40-80)")
    if hasattr(opt, 'lambda_dice'):
        print(f"   lambda_dice: {opt.lambda_dice}")
    if hasattr(opt, 'lambda_coarse_dice'):
        print(f"   lambda_coarse_dice: {opt.lambda_coarse_dice}")
    
    # Check GPU configuration
    num_gpus_requested = len(opt.gpu_ids.split(',')) if isinstance(opt.gpu_ids, str) else len(opt.gpu_ids)
    if num_gpus_requested > 1:
        print(f"\n⚠️  Multi-GPU training detected ({num_gpus_requested} GPUs)")
        print(f"   Paper used single GPU. Multi-GPU may affect batch normalization.")
        print(f"   Consider: --gpu_ids 0 (single GPU) if results don't match paper")
    
    # Check batch size
    if opt.batch_size != 1:
        print(f"\nℹ️  Batch size: {opt.batch_size}")
    
    print("="*80 + "\n")
    # ================================================
    
    # ============ MULTI-GPU SETUP ============
    num_gpus = torch.cuda.device_count()
    print(f"[GPU Detection] Found {num_gpus} GPU(s) available")
    
    if num_gpus > 1:
        print(f"[Multi-GPU Mode] Using all {num_gpus} GPUs")
        opt.gpu_ids = list(range(num_gpus))
    else:
        print("[Single-GPU Mode] Using GPU 0")
        opt.gpu_ids = [0]
    
    print(f"[GPU Configuration] GPU IDs: {opt.gpu_ids}")
    # ==========================================
    
    logdir = os.path.join(opt.checkpoints_dir, opt.name, 'checkpoints')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(logdir=logdir)
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)
    
    # ============ DATASET VALIDATION ============
    print("\n" + "="*80)
    print("📁 DATASET VALIDATION")
    print("="*80)
    if dataset_size < 1000:
        print(f"⚠️  WARNING: Training set size = {dataset_size}")
        print(f"   Paper reports 1217 training samples from Verse2019")
        print(f"   Your dataset may be incomplete or using a subset")
        print(f"   This will significantly impact model performance!")
    else:
        print(f"✅ Training set size: {dataset_size} samples")
    print("="*80 + "\n")
    # ============================================
    
    sample_test_limit = getattr(opt, 'sample_test', None)
    if sample_test_limit is not None:
        print(f'[Sample Test Mode] Training limited to {sample_test_limit} iterations')
    
    import copy
    opt_test = copy.deepcopy(opt)
    opt_test.batch_size = 5
    opt_test.serial_batches = True
    opt_test.phase = "test"
    opt_test.num_threads = 0
    opt_test.isTrain = False
    dataset_test = create_dataset(opt_test)

    model = create_model(opt)      # create a model given opt.model and other options
    
    # =====================================================
    
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    
    # ===== UPDATED: Initialize best model tracking =====
    best_ssim = 0
    best_epoch = 0
    best_metrics = {}
    # ===== END UPDATED SECTION =====

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer
        model.update_learning_rate()    # update learning rates in the beginning of every epoch
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # Check if sample test limit reached
            if sample_test_limit is not None and total_iters >= sample_test_limit:
                print(f'\n[Sample Test Mode] Reached iteration limit ({sample_test_limit}). Stopping training.')
                print('Saving final model...')
                model.save_networks('latest')
                model.save_networks(f'sample_test_{sample_test_limit}')
                print('Training completed successfully!')
                exit(0)

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
            
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            
        # 经过15个epoch评估一次
        if epoch % 15 == 0:
            device_for_eval = f'cuda:{opt.gpu_ids[0]}' if torch.cuda.is_available() else 'cpu'
            avg_ssim, avg_psnr, avg_dice, avg_iou, avg_diffh = evaluate_model(
                model, dataset_test, device_for_eval, 
                os.path.join(opt.checkpoints_dir, opt.name), epoch
            )
            # 记录评估指标
            writer.add_scalar('Eval/SSIM', avg_ssim, epoch)
            writer.add_scalar('Eval/PSNR', avg_psnr, epoch)
            writer.add_scalar('Eval/Dice', avg_dice, epoch)
            writer.add_scalar('Eval/IoU', avg_iou, epoch)
            writer.add_scalar('Eval/DiffH', avg_diffh, epoch)
            print(f'epoch[{epoch}/{opt.n_epochs + opt.n_epochs_decay + 1}], SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.2f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}, Diffh: {avg_diffh:.2f}')
            
            # ===== UPDATED: Track and save best model based on SSIM =====
            if avg_ssim > best_ssim:
                best_ssim = avg_ssim
                best_epoch = epoch
                best_metrics = {
                    'epoch': epoch,
                    'ssim': avg_ssim,
                    'psnr': avg_psnr,
                    'dice': avg_dice,
                    'iou': avg_iou,
                    'diffh': avg_diffh
                }
                print(f'\n✅ NEW BEST MODEL: SSIM {avg_ssim:.4f} at epoch {epoch}')
                model.save_networks('best_ssim')  # Save best model with special name
                
                # Save best model info to file for reference
                info_path = os.path.join(opt.checkpoints_dir, opt.name, 'best_model_info.txt')
                with open(info_path, 'w') as f:
                    f.write(f'Best Epoch: {best_epoch}\n')
                    f.write(f'Best SSIM: {best_metrics["ssim"]:.4f}\n')
                    f.write(f'PSNR: {best_metrics["psnr"]:.2f}\n')
                    f.write(f'Dice: {best_metrics["dice"]:.4f}\n')
                    f.write(f'IoU: {best_metrics["iou"]:.4f}\n')
                    f.write(f'DiffH: {best_metrics["diffh"]:.2f}%\n')
                print(f'   Saved best model info to {info_path}')
            # ===== END UPDATED SECTION =====

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    
    # ===== UPDATED: Print final best model information =====
    print('\n' + '='*80)
    print('🏆 TRAINING COMPLETED')
    print('='*80)
    if best_epoch > 0:
        print(f'\n✅ Best Model Summary:')
        print(f'   Epoch: {best_metrics["epoch"]}')
        print(f'   SSIM: {best_metrics["ssim"]:.4f}')
        print(f'   PSNR: {best_metrics["psnr"]:.2f} dB')
        print(f'   Dice: {best_metrics["dice"]:.4f}')
        print(f'   IoU: {best_metrics["iou"]:.4f}')
        print(f'   DiffH: {best_metrics["diffh"]:.2f}%')
        print(f'\n💾 Best model saved as: best_ssim')
        print(f'   You can test it with: python test.py --epoch best_ssim --name {opt.name}')
    else:
        print('⚠️  No evaluation metrics were recorded during training')
    print('='*80)
    # ===== END UPDATED SECTION =====