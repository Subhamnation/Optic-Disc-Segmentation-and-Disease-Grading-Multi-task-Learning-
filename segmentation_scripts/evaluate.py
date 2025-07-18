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

def visualize_results(images, masks, preds, save_dir="results"):
    """Save visualization of predictions"""
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(5, len(images))):
        plt.figure(figsize=(15, 5))
        
        # Original Image
        plt.subplot(1, 3, 1)
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        plt.imshow(np.clip(img, 0, 1))
        plt.title("Original Image")
        plt.axis('off')
        
        # True Mask
        plt.subplot(1, 3, 2)
        plt.imshow(masks[i].cpu().squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
        plt.title("True Mask")
        plt.axis('off')
        
        # Predicted Mask
        plt.subplot(1, 3, 3)
        plt.imshow(preds[i].cpu().squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
        plt.title("Predicted Mask")
        plt.axis('off')
        
        plt.savefig(os.path.join(save_dir, f"result_{i}.png"), bbox_inches='tight', dpi=300)
        plt.close()

def main():
    # Configurations
    MODEL_DIR = "best_models"
    IMG_SIZE = 512
    BATCH_SIZE = 4  # Reduced for stability
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
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Evaluation
    total_iou = 0.0
    total_dice = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            # Safety check for size mismatch
            if outputs.shape[-2:] != masks.shape[-2:]:
                print(f"Warning: Resizing outputs from {outputs.shape[-2:]} to {masks.shape[-2:]}")
                outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            
            preds = torch.sigmoid(outputs)

            # Calculate metrics
            for i in range(images.size(0)):
                iou = calculate_iou(preds[i], masks[i])
                dice = calculate_dice(preds[i], masks[i])
                total_iou += iou.item()
                total_dice += dice.item()
                num_samples += 1

            # Visualize first batch
            if batch_idx == 0:
                visualize_results(images, masks, preds)

    # Print results
    print(f"\nEvaluation Metrics:")
    print(f"Test Samples: {num_samples}")
    print(f"Mean IoU: {total_iou/num_samples:.4f}")
    print(f"Mean Dice: {total_dice/num_samples:.4f}")
    print(f"Visualizations saved to 'results' directory")

if __name__ == '__main__':
    main()