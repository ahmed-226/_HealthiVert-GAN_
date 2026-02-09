"""
Diffusion Model wrapper for HealthiVert.
Wraps the diffusion generator in the model framework.
"""
from .pix2pix_model import Pix2PixModel


class DiffusionModel(Pix2PixModel):
    """
    Diffusion Model for HealthiVert.
    
    This is a wrapper around Pix2PixModel that automatically uses:
    - netG_type = 'diffusion' (instead of 'gan')
    - Diffusion-based training instead of pure GAN
    
    All parameters and behavior are inherited from Pix2PixModel.
    """
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add diffusion-specific options."""
        # Use parent class options
        parser = Pix2PixModel.modify_commandline_options(parser, is_train)
        
        # Override default to diffusion
        parser.set_defaults(netG_type='diffusion')
        
        return parser
