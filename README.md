# Optic Disc Segmentation and Multi-task Learning with EfficientNet-B0

## Overview

This repository presents a deep learning framework for **optic disc segmentation** and **multi-task learning (segmentation + disease grading)** using the **IDRiD dataset**.  
The project is organized into two core experiments:

1. **Optic Disc Segmentation (Single-task)**
2. **Multi-task Learning (Segmentation + Disease Grading)**

---

## 1️⃣ Optic Disc Segmentation Task

### 📐 Model Architecture

- **Base Model**: U-Net with **EfficientNet-B0 encoder** (via `torchvision.feature_extraction`).
- **Encoder**: Lightweight pretrained EfficientNet-B0 capturing multi-scale contextual features.
- **Decoder**:
  - Progressive upsampling with skip connections.
  - Final convolutional layers upscale to **512×512**.
- **Justification**: Combines EfficientNet’s parameter efficiency with U-Net’s strong localization capability, making it ideal for medical image segmentation tasks.

---

### 🗂️ Dataset & Preprocessing

- **Dataset**: [IDRiD - Optic Disc Segmentation Subset](https://idrid.grand-challenge.org/)
- **Preprocessing Steps**:
  - Image resizing:
    - **Trial 1**: 256×256
    - **Trial 2**: 512×512
  - Data augmentations applied:
    - Random crop
    - Horizontal and vertical flips
    - Random rotations
    - Color jitter
    - Elastic deformations
  - **Normalization**: Standardized using ImageNet mean and standard deviation.
  - **Mask Binarization**: Applied to obtain clear optic disc regions.

---

### 🏋️ Training Strategy

- **Loss Function**:
  - Combination of **BCEWithLogitsLoss (70%)** and **Dice Loss (30%)**.
- **Optimizer**:
  - **Trial 1**: Adam
  - **Trial 2**: AdamW with weight decay
- **Learning Rate Schedulers**:
  - CosineAnnealingLR
  - ReduceLROnPlateau
- **Mixed Precision Training**: Enabled via `torch.cuda.amp`
- **Epochs**: 50
- **Checkpointing**: Best model saved based on validation loss.

---

### 🧪 Evaluation Metrics & Results

- **Segmentation Metrics**:
  - Dice Score
  - Intersection over Union (IoU)
- **Postprocessing**:
  - Sigmoid activation + thresholding
  - Bilinear resizing to correct dimensional mismatches
- **Inference Protocol**:
  - Batched DataLoader evaluation
  - Metrics averaged over the entire test set

| Trial  | IoU    | Dice  |
|--------|--------|-------|
| 1 (256×256) | 0.2661 | 0.3083 |
| 2 (512×512) | 0.7041 | 0.7780 |

---

## 2️⃣ Multi-task Learning: Optic Disc Segmentation + Disease Grading

### 📐 Model Architecture

- **Shared Backbone**: EfficientNet-B0 pretrained on ImageNet.
- **Segmentation Head**:
  - Multi-stage upsampling with convolutional layers.
  - Output mask size: **384×384**
- **Classification Head**:
  - Adaptive pooling followed by fully connected layers.
  - Outputs:
    - **DR Grade** (5-class classification)
    - **DME Risk** (3-class classification)
- **Justification**: The shared encoder improves efficiency and supports feature reuse across segmentation and classification tasks.

---

### 🗂️ Dataset & Preprocessing

- **Segmentation Data**: IDRiD optic disc masks.
- **Classification Data**: DR and DME labels sourced from IDRiD annotation CSV.
- **Preprocessing**:
  - Resize images and masks to **384×384**
  - Normalize using ImageNet statistics.
  - Binarize masks for segmentation.
  - Parse CSV labels for classification tasks.

---

### 🏋️ Training Strategy

- **Training Loop**: Alternating between segmentation and classification in each epoch.
- **Loss Functions**:
  - **Segmentation**: BCEWithLogitsLoss
  - **Classification**: CrossEntropyLoss for both DR and DME tasks (losses averaged)
- **Optimizer**: Adam with learning rate **1e-4**
- **Mixed Precision**: Enabled via `torch.cuda.amp`
- **Epochs**: 50
- **Checkpointing**: Saved every 2 epochs with dice score recorded in filename.

---

### 🧪 Evaluation Metrics & Performance

- **Segmentation Metrics**:
  - IoU
  - Dice Score
- **Classification Metrics**:
  - Accuracy for DR grading
  - Accuracy for DME risk prediction

| Epochs | IoU   | Dice  | DR Accuracy | DME Accuracy |
|--------|-------|-------|-------------|--------------|
| 10    | 0.6904 | 0.8016 | 0.6019 | 0.7670 |
| 20    | 0.8108 | 0.8934 | 0.6117 | 0.8058 |
| 50    | 0.8063 | 0.8902 | 0.5922 | 0.7184 |

---

### 📊 Visualization & Comparative Analysis

- Visualization of segmentation predictions and comparison with ground truth masks.
- Accuracy and Dice trends analyzed over training epochs.
- Performance comparison with single-task models.

---

## 📝 Conclusion

- The **single-task segmentation model** achieved a Dice score of **0.7780** with higher-resolution input (512×512).
- The **multi-task model** provided competitive segmentation results while achieving reliable DR and DME classification.
- Multi-task learning enhanced encoder efficiency and generalized well across tasks.

---

## 🛠️ Technologies & Frameworks Used

- **PyTorch** for model development
- **torchvision** for feature extraction and data transforms
- **Albumentations** for advanced data augmentations
- **Mixed Precision Training (AMP)** with `torch.cuda.amp`
- **IDRiD Dataset** as the primary data source for both segmentation and classification

---

## 🔥 Future Scope

- Hyperparameter optimization for balancing multi-task losses.
- Exploration of advanced data augmentations (e.g., CutMix, MixUp).
- Testing with more advanced architectures like EfficientNetV2 or ConvNeXt.
- Cross-dataset validation to assess model generalizability.

---

## 📝 Citation & Acknowledgements

If you use this project or dataset, please cite:

```bibtex
@inproceedings{porwal2018idrid,
  title={Indian diabetic retinopathy image dataset (IDRiD): A database for diabetic retinopathy screening research},
  author={Porwal, Prasanna and Pachade, Sachin and Kamble, Rohan and Kokare, Manesh and Deshmukh, Gopal and Sahasrabuddhe, Vinayak and Sardar, Pravin and More, Ashish and Raut, Sharad and et al.},
  booktitle={2018 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  pages={0--0},
  year={2018},
  organization={IEEE}
}
