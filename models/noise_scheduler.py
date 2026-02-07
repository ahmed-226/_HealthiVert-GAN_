"""
Noise scheduler for diffusion model training and inference.
Implements DDPM (Denoising Diffusion Probabilistic Models) with linear beta schedule.
"""
import torch
import torch.nn as nn
import numpy as np


class DDPMScheduler:
    """
    DDPM noise scheduler with linear beta schedule.
    
    Args:
        T (int): Number of diffusion timesteps (default: 1000)
        beta_start (float): Starting value of beta schedule (default: 0.0001)
        beta_end (float): Ending value of beta schedule (default: 0.02)
    """
    
    def __init__(self, T=1000, beta_start=0.0001, beta_end=0.02):
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, T)
        
        # Pre-compute alpha values for efficiency
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Pre-compute square roots for noise addition/removal
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def add_noise(self, x_0, t, noise=None):
        """
        Add noise to clean images according to forward diffusion process.
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        
        Args:
            x_0: Clean images, shape (B, C, H, W)
            t: Timesteps, shape (B,)
            noise: Optional pre-generated noise, shape (B, C, H, W)
            
        Returns:
            Noisy images x_t, shape (B, C, H, W)
            Noise epsilon, shape (B, C, H, W)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Move schedule to same device as input
        device = x_0.device
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].to(device).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].to(device).view(-1, 1, 1, 1)
        
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        
        return x_t, noise
    
    def predict_x0_from_eps(self, x_t, eps, t):
        """
        Predict clean image x_0 from noisy image x_t and predicted noise.
        x_0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        
        Args:
            x_t: Noisy images, shape (B, C, H, W)
            eps: Predicted noise, shape (B, C, H, W)
            t: Timesteps, shape (B,)
            
        Returns:
            Predicted clean images, shape (B, C, H, W)
        """
        device = x_t.device
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].to(device).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].to(device).view(-1, 1, 1, 1)
        
        x_0 = (x_t - sqrt_one_minus_alpha_bar * eps) / (sqrt_alpha_bar + 1e-8)
        
        # Clamp to valid range for CT images
        x_0 = torch.clamp(x_0, -1.0, 1.0)
        
        return x_0
    
    def ddim_step(self, x_t, eps, t, t_prev, eta=0.0):
        """
        Single DDIM sampling step (faster inference than DDPM).
        
        Args:
            x_t: Noisy image at timestep t
            eps: Predicted noise
            t: Current timestep
            t_prev: Previous timestep
            eta: DDIM stochasticity parameter (0 = deterministic)
            
        Returns:
            x_{t-1}: Less noisy image
        """
        device = x_t.device
        
        # Get alpha values
        alpha_bar_t = self.alphas_cumprod[t].to(device).view(-1, 1, 1, 1)
        if t_prev >= 0:
            alpha_bar_t_prev = self.alphas_cumprod[t_prev].to(device).view(-1, 1, 1, 1)
        else:
            alpha_bar_t_prev = torch.ones_like(alpha_bar_t)
        
        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t + 1e-8)
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)
        
        # Compute direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_bar_t_prev) * eps
        
        # Compute x_{t-1}
        x_t_prev = torch.sqrt(alpha_bar_t_prev) * x_0_pred + dir_xt
        
        return x_t_prev
