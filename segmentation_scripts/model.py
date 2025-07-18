

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
from torchvision.models.feature_extraction import create_feature_extractor

class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        # Encoder (EfficientNet-B0 backbone)
        backbone = efficientnet_b0(weights='DEFAULT').features
        self.encoder = create_feature_extractor(backbone, {
            '0': 'block0',  # 16 channels
            '2': 'block1',  # 24 channels
            '3': 'block2',  # 40 channels
            '4': 'block3',  # 80 channels
            '6': 'block4'   # 192 channels
        })
        
        # Decoder with corrected channel dimensions
        self.up1 = self._up_block(192, 80)  # Matches block3's 80 channels
        self.up2 = self._up_block(80, 40)   # Matches block2's 40 channels
        self.up3 = self._up_block(40, 24)   # Matches block1's 24 channels
        self.up4 = self._up_block(24, 16)   # Matches block0's 16 channels
        
        # Final upsampling to 512x512
        self.final_upsample = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # Additional convolution
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(8, n_classes, kernel_size=1)  # Final 1x1 conv
        )
        
    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Store original input size for final upsampling
        original_size = x.shape[2:]
        
        # Encoder
        features = self.encoder(x)
        e1 = features['block0']  # 16 channels
        e2 = features['block1']  # 24 channels
        e3 = features['block2']  # 40 channels
        e4 = features['block3']  # 80 channels
        e5 = features['block4']  # 192 channels
        
        # Decoder with spatial adjustment
        d1 = self._add_with_adjustment(self.up1(e5), e4)
        d2 = self._add_with_adjustment(self.up2(d1), e3)
        d3 = self._add_with_adjustment(self.up3(d2), e2)
        d4 = self._add_with_adjustment(self.up4(d3), e1)
        
        # Final upsampling to original input size (512x512)
        output = self.final_upsample(d4)
        
        # Ensure exact size match (sometimes bilinear can be off by 1 pixel)
        if output.shape[2:] != original_size:
            output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
        
        return output
    
    def _add_with_adjustment(self, decoder_out, encoder_out):
        """Handles spatial and channel dimension mismatches"""
        # Channel adjustment
        if decoder_out.size(1) != encoder_out.size(1):
            encoder_out = nn.Conv2d(encoder_out.size(1), decoder_out.size(1), kernel_size=1).to(encoder_out.device)(encoder_out)
        
        # Spatial adjustment
        if decoder_out.shape[2:] != encoder_out.shape[2:]:
            encoder_out = F.interpolate(encoder_out, size=decoder_out.shape[2:], mode='bilinear', align_corners=False)
        
        return decoder_out + encoder_out