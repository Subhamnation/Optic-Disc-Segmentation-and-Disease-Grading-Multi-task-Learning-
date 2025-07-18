
import torch
import torch.nn.functional as F
import numpy as np
from i_data import IDRiDDataset
from model import UNet
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import glob

def calculate_iou(pred, target):
    """Calculate Intersection over Union"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def calculate_dice(pred, target):
    """Calculate Dice coefficient"""
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)

def visualize_single_result(image, mask, pred):
    """Visualize single image with its true and predicted mask"""
    plt.figure(figsize=(15, 5))
    
    # Original Image
    plt.subplot(1, 3, 1)
    img = image.cpu().permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    plt.imshow(np.clip(img, 0, 1))
    plt.title("Original Image")
    plt.axis('off')
    
    # True Mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask.cpu().squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
    plt.title("True Mask")
    plt.axis('off')
    
    # Predicted Mask
    plt.subplot(1, 3, 3)
    plt.imshow(pred.cpu().squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
    plt.title("Predicted Mask")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Configurations
    MODEL_DIR = "best_models"
    IMG_SIZE = 512
    DATA_DIR = r"E:\courses\Assignement_round_2\A. Segmentation\A. Segmentation"
    TEST_IMG_DIR = os.path.join(DATA_DIR, r"1. Original Images\b. Testing Set")
    TEST_MASK_DIR = os.path.join(DATA_DIR, r"2. All Segmentation Groundtruths\b. Testing Set\5. Optic Disc")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Find latest model
    model_files = glob.glob(os.path.join(MODEL_DIR, "*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {MODEL_DIR}")
    
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Loading model: {latest_model}")

    # Load model
    model = UNet(n_classes=1).to(device)
    checkpoint = torch.load(latest_model, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Initialize dataset - ensure mask_size matches model output
    test_dataset = IDRiDDataset(
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        img_size=IMG_SIZE,
        mask_size=512,  # Changed to match model output
        augment=False
    )
    
    # Get a single sample (first image in the dataset)
    image, mask = test_dataset[2]
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    mask = mask.unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(image)
        
        # Safety check for size mismatch
        if output.shape[-2:] != mask.shape[-2:]:
            print(f"Warning: Resizing outputs from {output.shape[-2:]} to {mask.shape[-2:]}")
            output = F.interpolate(output, size=mask.shape[-2:], mode='bilinear', align_corners=False)
        
        pred = torch.sigmoid(output)

    # Calculate metrics
    iou = calculate_iou(pred[0], mask[0])
    dice = calculate_dice(pred[0], mask[0])
    
    print(f"\nEvaluation Metrics for Single Image:")
    print(f"IoU: {iou.item():.4f}")
    print(f"Dice: {dice.item():.4f}")

    # Visualize results
    visualize_single_result(image[0], mask[0], pred[0])

if __name__ == '__main__':
    main()