
##########################################



import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

class IDRiDDataset(Dataset):
    """Optimized Segmentation Dataset"""
    def __init__(self, img_dir, mask_dir, img_size=384, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = (img_size, img_size)
        
        # Get sorted image list
        self.images = sorted([f for f in os.listdir(img_dir) 
                            if f.lower().endswith(('.jpg', '.png', '.tif'))])
        self.masks = [f.replace('.jpg', '_OD.tif').replace('.png', '_OD.tif') 
                     for f in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # Process mask
        mask = np.array(mask) > 0  # Direct binarization
        mask = mask.astype(np.float32)
        
        # Resize
        image = F.resize(image, self.img_size, InterpolationMode.BILINEAR)
        mask = F.resize(Image.fromarray(mask), self.img_size, InterpolationMode.NEAREST)
        
        # Convert to tensor
        image = F.to_tensor(image)
        mask = F.to_tensor(np.array(mask))  # Direct conversion
        
        # Normalize
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
        
        return image, mask

class ClassificationDataset(Dataset):
    """Optimized Classification Dataset"""
    def __init__(self, img_dir, csv_path, img_size=384):
        self.img_dir = img_dir
        self.img_size = (img_size, img_size)
        
        # Load and verify CSV
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        
        # Precompute valid paths and labels
        self.samples = []
        for _, row in self.df.iterrows():
            img_name = str(row['Image name']).strip()
            for ext in ['.jpg', '.jpeg', '.png', '.tif']:
                path = os.path.join(img_dir, img_name + ext)
                if os.path.exists(path):
                    self.samples.append((
                        path,
                        int(row['Retinopathy grade']),
                        int(row['Risk of macular edema'])
                    ))
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, dr_grade, dme_grade = self.samples[idx]
        
        # Load and process image
        image = Image.open(path).convert('RGB')
        image = F.resize(image, self.img_size, InterpolationMode.BILINEAR)
        image = F.to_tensor(image)
        image = F.normalize(image, 
                          mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
        
        return image, torch.tensor(dr_grade), torch.tensor(dme_grade)