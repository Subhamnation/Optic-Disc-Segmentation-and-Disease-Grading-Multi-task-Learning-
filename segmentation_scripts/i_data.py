
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision import transforms  # Added this import
from torchvision.transforms import InterpolationMode
import random
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

class IDRiDDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=512, mask_size=None, augment=True):
        """
        Args:
            img_dir: Directory with training images
            mask_dir: Directory with segmentation masks
            img_size: Size to resize input images (width, height)
            mask_size: Size to resize masks (if None, uses img_size)
            augment: Whether to apply data augmentation
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.mask_size = (mask_size, mask_size) if mask_size else self.img_size
        self.augment = augment

        # Accepted extensions
        exts = ('.jpg', '.JPG', '.png', '.PNG', '.tif')

        # Get sorted image list
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(exts)])

        # Map masks from image base names
        self.masks = [os.path.splitext(f)[0] + "_OD.tif" for f in self.images]

        # Verify that mask files exist
        for mask in self.masks:
            full_path = os.path.join(mask_dir, mask)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Mask file not found: {full_path}")

    def elastic_transform(self, image, mask, alpha=1000, sigma=30):
        """Elastic deformation of images as described in [Simard2003]"""
        random_state = np.random.RandomState(None)
        shape = image.size[::-1]  # (width, height)

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

        # Convert PIL Image to numpy array for transformation
        image_np = np.array(image)
        mask_np = np.array(mask)

        # Apply to each channel of the image
        transformed_image = []
        for i in range(3):  # RGB channels
            transformed_image.append(map_coordinates(image_np[:,:,i], indices, order=1, mode='reflect').reshape(shape))
        transformed_image = np.stack(transformed_image, axis=-1)

        transformed_mask = map_coordinates(mask_np, indices, order=1, mode='reflect').reshape(shape)

        # Convert back to PIL Image
        return Image.fromarray(transformed_image), Image.fromarray(transformed_mask)

    def random_crop(self, image, mask, crop_size=(400, 400)):
        """Random crop with same parameters for image and mask"""
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=crop_size)
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)
        return image, mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Load and binarize mask
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask = (mask > 0).astype(np.float32)  # Binarize
        mask = Image.fromarray(mask)

        # Initial resize (we'll crop later if needed)
        image = F.resize(image, (int(self.img_size[0]*1.2), int(self.img_size[1]*1.2)), 
                       interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, (int(self.mask_size[0]*1.2), int(self.mask_size[1]*1.2)), 
                      interpolation=InterpolationMode.NEAREST)

        # Augmentation (synchronized)
        if self.augment:
            # Random crop
            if random.random() > 0.5:
                image, mask = self.random_crop(image, mask, crop_size=self.img_size)

            # Color jitter (only on image)
            if random.random() > 0.5:
                color_jitter = transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1)
                image = color_jitter(image)

            # Elastic transform
            if random.random() > 0.7:  # 30% chance
                image, mask = self.elastic_transform(image, mask, alpha=500, sigma=25)

            # Geometric transforms
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = F.rotate(image, angle)
                mask = F.rotate(mask, angle)

            if random.random() > 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)

            if random.random() > 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)

        # Final resize to ensure correct size
        image = F.resize(image, self.img_size, interpolation=InterpolationMode.BILINEAR)
        mask = F.resize(mask, self.mask_size, interpolation=InterpolationMode.NEAREST)

        # Convert to tensors
        image = F.to_tensor(image)
        mask = F.to_tensor(mask)

        # Normalize using ImageNet stats
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image, mask