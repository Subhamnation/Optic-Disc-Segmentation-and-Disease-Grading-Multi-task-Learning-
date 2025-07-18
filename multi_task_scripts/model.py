


################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
from torchvision.models.feature_extraction import create_feature_extractor

class MultiTaskModel(nn.Module):
    def __init__(self, seg_classes=1, dr_classes=5, dme_classes=3):
        super().__init__()
        
        # Shared backbone with correct feature extraction
        backbone = efficientnet_b0(weights='DEFAULT').features
        self.feature_extractor = create_feature_extractor(
            backbone,
            return_nodes={
                '0': 'block0',
                '1': 'block1',
                '2': 'block2', 
                '3': 'block3',
                '4': 'block4'
            }
        )
        
        # Segmentation head (corrected input channels and upsampling)
        self.seg_head = nn.Sequential(
            nn.Conv2d(80, 128, kernel_size=3, padding=1),  # Fixed input channels
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, seg_classes, kernel_size=1)
        )
        
        # Classification head (corrected input channels)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(80, 256),  # Fixed input channels
            nn.ReLU()
        )
        
        # Task-specific output layers
        self.fc_dr = nn.Linear(256, dr_classes)
        self.fc_dme = nn.Linear(256, dme_classes)

    def forward(self, x, task_type):
        features = self.feature_extractor(x)
        
        if task_type == 'seg':
            # Segmentation path (removed unnecessary interpolation)
            return self.seg_head(features['block4'])
        
        elif task_type == 'cls':
            # Classification path
            x = self.cls_head(features['block4'])
            return self.fc_dr(x), self.fc_dme(x)
        
        else:
            raise ValueError("task_type must be 'seg' or 'cls'")