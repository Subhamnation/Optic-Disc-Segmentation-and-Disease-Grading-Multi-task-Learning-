import torch
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from i_data import IDRiDDataset, ClassificationDataset
from model import MultiTaskModel
import torch.nn.functional as F
from PIL import Image

def calculate_seg_metrics(pred, target):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    return iou.item(), dice.item()

def evaluate_segmentation(model, loader, device):
    model.eval()
    total_iou, total_dice = 0.0, 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            outputs = model(images, 'seg')
            preds = torch.sigmoid(outputs)
            for i in range(images.size(0)):
                iou, dice = calculate_seg_metrics(preds[i], masks[i])
                total_iou += iou
                total_dice += dice
    return total_iou / len(loader.dataset), total_dice / len(loader.dataset)

def evaluate_classification(model, loader, device):
    model.eval()
    dr_correct = dme_correct = total = 0
    with torch.no_grad():
        for images, dr_true, dme_true in loader:
            images = images.to(device, non_blocking=True)
            dr_true, dme_true = dr_true.to(device), dme_true.to(device)
            dr_pred, dme_pred = model(images, 'cls')
            dr_correct += (dr_pred.argmax(1) == dr_true).sum().item()
            dme_correct += (dme_pred.argmax(1) == dme_true).sum().item()
            total += images.size(0)
    return dr_correct / total, dme_correct / total

def find_best_model(model_dir, metric='dice'):
    """Find best model based on saved metric in filename"""
    model_files = glob.glob(os.path.join(model_dir, "*.pth"))
    if not model_files:
        raise FileNotFoundError("No model files found in directory")
    
    best_model = None
    best_score = -1.0
    
    for model_path in model_files:
        filename = os.path.basename(model_path)
        parts = filename.split('_')
        
        # Look for metric score in filename
        for part in parts:
            if metric in part:
                try:
                    score = float(part.replace(metric, ''))
                    if score > best_score:
                        best_score = score
                        best_model = model_path
                except ValueError:
                    continue
                    
    if best_model is None:
        # Fallback to latest model if no metric found
        best_model = max(model_files, key=os.path.getctime)
        print(f"No metric found in filenames. Using latest model: {best_model}")
    else:
        print(f"Selected best model based on {metric}: {best_model} (Score: {best_score:.4f})")
    
    return best_model

def visualize_segmentation(model, loader, device, save_dir="segmentation_visuals", num_samples=5):
    """Generate individual visualizations for each sample"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # Get a batch of data
    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images[:num_samples], 'seg')
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()
    
    # Create individual visualizations
    saved_paths = []
    for i in range(min(num_samples, images.size(0))):
        # Original image (unnormalize)
        img = images[i].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        orig_img = img * std + mean
        orig_img = torch.clamp(orig_img, 0, 1).permute(1, 2, 0).numpy()
        
        # True mask
        true_mask = masks[i].cpu().permute(1, 2, 0).numpy()
        true_mask = np.repeat(true_mask, 3, axis=-1)  # Convert to RGB
        
        # Predicted mask
        pred_mask = preds[i].cpu().permute(1, 2, 0).numpy()
        pred_mask = np.repeat(pred_mask, 3, axis=-1)  # Convert to RGB
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Segmentation Results - Sample {i+1}', fontsize=16)
        
        # Original image
        axes[0].imshow(orig_img)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # True mask
        axes[1].imshow(true_mask, cmap='gray')
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')
        
        # Predicted mask
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title("Predicted Mask")
        axes[2].axis('off')
        
        # Save individual figure
        save_path = os.path.join(save_dir, f"segmentation_sample_{i+1}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        saved_paths.append(save_path)
        print(f"Saved visualization for sample {i+1} to: {save_path}")
    
    return saved_paths

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configuration
    SEG_IMG_DIR = r"E:\courses\Assignement_round_2\A. Segmentation\A. Segmentation\1. Original Images\b. Testing Set"
    SEG_MASK_DIR = r"E:\courses\Assignement_round_2\A. Segmentation\A. Segmentation\2. All Segmentation Groundtruths\b. Testing Set\5. Optic Disc"
    CLS_IMG_DIR = r"E:\courses\Assignement_round_2\B. Disease Grading\B. Disease Grading\1. Original Images\b. Testing Set"
    CLS_TEST_CSV = r"E:\courses\Assignement_round_2\B. Disease Grading\B. Disease Grading\2. Groundtruths\b. IDRiD_Disease Grading_Testing Labels.csv"
    MODEL_DIR = "best_models_multi-task"
    VISUAL_DIR = "segmentation_visuals"

    # Find and load best model
    model_path = find_best_model(MODEL_DIR)
    model = MultiTaskModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model from: {model_path}")

    # Segmentation Evaluation
    seg_dataset = IDRiDDataset(SEG_IMG_DIR, SEG_MASK_DIR)
    seg_loader = DataLoader(
        seg_dataset, 
        batch_size=4, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    seg_iou, seg_dice = evaluate_segmentation(model, seg_loader, device)

    # Generate segmentation visuals
    vis_paths = visualize_segmentation(model, seg_loader, device, save_dir=VISUAL_DIR, num_samples=5)

    # Classification Evaluation
    cls_dataset = ClassificationDataset(CLS_IMG_DIR, CLS_TEST_CSV)
    cls_loader = DataLoader(
        cls_dataset, 
        batch_size=8, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    dr_acc, dme_acc = evaluate_classification(model, cls_loader, device)

    # Print results
    print("\n" + "="*50)
    print(f"{'Evaluation Results':^50}")
    print("="*50)
    print(f"{'Segmentation Metrics':<25} | {'Value':>12}")
    print("-"*50)
    print(f"{'IoU (Optic Disc)':<25} | {seg_iou:.4f}")
    print(f"{'Dice Score':<25} | {seg_dice:.4f}")
    print("\n" + f"{'Classification Metrics':<25} | {'Value':>12}")
    print("-"*50)
    print(f"{'DR Accuracy':<25} | {dr_acc:.4f}")
    print(f"{'DME Accuracy':<25} | {dme_acc:.4f}")
    print("="*50)
    
    # Display visualization paths in results
    print("\nSegmentation visualizations saved at:")
    for path in vis_paths:
        print(f"  - {os.path.abspath(path)}")

if __name__ == '__main__':
    main()