
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from i_data import IDRiDDataset
from model import UNet
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # Weight between BCE and Dice
        self.smooth = smooth

    def forward(self, pred, target):
        # BCEWithLogits
        bce = F.binary_cross_entropy_with_logits(pred, target)
        
        # Dice
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return self.alpha * bce + (1 - self.alpha) * (1 - dice)

def save_model(model, optimizer, epoch, loss, save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f"model_epoch{epoch}_{timestamp}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, model_path)
    print(f"Model saved to {model_path}")

def main():
    # Configurations
    SEED = 42
    IMG_SIZE = 512
    BATCH_SIZE = 4
    EPOCHS = 50
    LR = 1e-4
    DATA_DIR = r"E:\courses\Assignement_round_2\A. Segmentation\A. Segmentation"
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, r"1. Original Images\a. Training Set")
    TRAIN_MASK_DIR = os.path.join(DATA_DIR, r"2. All Segmentation Groundtruths\a. Training Set\5. Optic Disc")

    # Setup
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data loading
    train_dataset = IDRiDDataset(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        img_size=IMG_SIZE,
        mask_size=512,
        augment=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    # Model initialization
    model = UNet(n_classes=1).to(device)
    scaler = GradScaler()
    #scaler = torch.amp.GradScaler(device_type='cuda', enabled=True)
    
    # Loss and optimizer
    criterion = ComboLoss(alpha=0.7)  # 70% BCE, 30% Dice
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    
    # Learning rate schedulers
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss/len(train_loader)
        
        # Update schedulers
        lr_scheduler.step()
        plateau_scheduler.step(avg_loss)
        
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, optimizer, epoch+1, avg_loss, save_dir="best_models")
            print(f"New best model saved with loss {avg_loss:.4f}")
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            save_model(model, optimizer, epoch+1, avg_loss) 

if __name__ == '__main__':
    main()