
##############################

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from i_data import IDRiDDataset, ClassificationDataset
from model import MultiTaskModel

def main():
    # Configuration
    SEG_IMG_DIR = r"E:\courses\Assignement_round_2\A. Segmentation\A. Segmentation\1. Original Images\a. Training Set"
    SEG_MASK_DIR = r"E:\courses\Assignement_round_2\A. Segmentation\A. Segmentation\2. All Segmentation Groundtruths\a. Training Set\5. Optic Disc"
    CLS_IMG_DIR = r"E:\courses\Assignement_round_2\B. Disease Grading\B. Disease Grading\1. Original Images\a. Training Set"
    CLS_CSV_PATH = r"E:\courses\Assignement_round_2\B. Disease Grading\B. Disease Grading\2. Groundtruths\a. IDRiD_Disease Grading_Training Labels.csv"
    SAVE_DIR = "best_models_multi-task" 

    # Create save directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Parameters
    IMG_SIZE = 384
    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    print(f"Models will be saved to: {os.path.abspath(SAVE_DIR)}")
    
    # Initialize datasets
    seg_dataset = IDRiDDataset(SEG_IMG_DIR, SEG_MASK_DIR, IMG_SIZE)
    cls_dataset = ClassificationDataset(CLS_IMG_DIR, CLS_CSV_PATH, IMG_SIZE)

    # DataLoaders with pinned memory
    seg_loader = DataLoader(
        seg_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    cls_loader = DataLoader(
        cls_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Model setup
    model = MultiTaskModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    seg_criterion = nn.BCEWithLogitsLoss()
    cls_criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        
        # Segmentation training
        seg_loss = 0.0
        for images, masks in seg_loader:
            images, masks = images.to(DEVICE, non_blocking=True), masks.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images, 'seg')
                loss = seg_criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            seg_loss += loss.item()
        
        # Classification training
        cls_loss = 0.0
        for images, dr_grades, dme_grades in cls_loader:
            images = images.to(DEVICE, non_blocking=True)
            dr_grades = dr_grades.to(DEVICE, non_blocking=True)
            dme_grades = dme_grades.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast():
                dr_pred, dme_pred = model(images, 'cls')
                loss_dr = cls_criterion(dr_pred, dr_grades)
                loss_dme = cls_criterion(dme_pred, dme_grades)
                loss = (loss_dr + loss_dme) / 2
            
            loss.backward()
            optimizer.step()
            cls_loss += loss.item()
        
        # Print stats
        avg_seg_loss = seg_loss / len(seg_loader)
        avg_cls_loss = cls_loss / len(cls_loader)
        print(f"Seg Loss: {avg_seg_loss:.4f} | Cls Loss: {avg_cls_loss:.4f}")
        
        # Save model every 2 epochs
        if (epoch + 1) % 2 == 0 or (epoch + 1) == NUM_EPOCHS:
            model_name = f"model_epoch_{epoch+1}_seg{avg_seg_loss:.4f}_cls{avg_cls_loss:.4f}.pth"
            save_path = os.path.join(SAVE_DIR, model_name)
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to: {save_path}")

    print("\nTraining complete!")
    print(f"Final models saved in: {os.path.abspath(SAVE_DIR)}")

if __name__ == '__main__':
    main()