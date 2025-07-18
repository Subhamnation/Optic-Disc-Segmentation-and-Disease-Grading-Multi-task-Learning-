Optic Disc Segmentation and Multi-task Learning with EfficientNet-B0 Backbone
Overview
This repository presents a deep learning framework for optic disc segmentation and multi-task learning (segmentation + disease grading) using the IDRiD dataset. The project is structured around two core tasks:

Optic Disc Segmentation (Single-task)

Multi-task Learning (Segmentation + DR/DME Classification)

1️⃣ Optic Disc Segmentation Task
📐 Model Architecture
Base Model: U-Net with an EfficientNet-B0 encoder (torchvision.feature_extraction).

Encoder: Pretrained EfficientNet-B0 capturing multi-scale contextual features with high parameter efficiency.

Decoder:

Progressive upsampling with skip connections.

Final convolution layers upscale to 512×512.

Design Rationale:
Merges EfficientNet's feature extraction capacity with U-Net's localization strength, particularly suited for medical image segmentation tasks.

🗂️ Dataset & Preprocessing
Dataset: IDRiD - Optic Disc Segmentation Subset

Preprocessing Techniques:

Image resizing:

Trial 1: 256×256

Trial 2: 512×512

Data augmentations:

Random cropping

Horizontal/vertical flips

Random rotations

Color jittering

Elastic deformations

Normalization: Using ImageNet mean and std.

Mask Processing: Binarization to segment optic disc regions.

🏋️ Training Strategy
Loss Function:

Weighted combination of BCEWithLogitsLoss (70%) and Dice Loss (30%).

Optimizers:

Trial 1: Adam

Trial 2: AdamW (with weight decay)

Learning Rate Schedulers:

CosineAnnealingLR

ReduceLROnPlateau

Mixed Precision Training: Enabled using torch.cuda.amp

Epochs: 50

Checkpointing: Best model saved based on lowest validation loss.

🧪 Evaluation Metrics & Results
Metrics:

Dice Score

Intersection over Union (IoU)

Postprocessing:

Sigmoid activation + thresholding

Bilinear interpolation for resizing during inference

Inference Protocol:

Batched evaluation on the test set

Metrics averaged over all samples

Trial	IoU	Dice
1 (256×256)	0.2661	0.3083
2 (512×512)	0.7041	0.7780

📊 Visualization
Visual comparative analysis was performed between predicted and ground truth masks to qualitatively assess segmentation performance.

2️⃣ Multi-task Learning: Optic Disc Segmentation + Disease Grading
📐 Model Architecture
Backbone: Shared EfficientNet-B0 feature extractor.

Segmentation Head:

Multi-stage upsampling with skip connections.

Outputs 384×384 optic disc masks.

Classification Head:

Adaptive pooling + Fully Connected layers.

Dual output:

DR Grading (5 classes)

DME Risk Prediction (3 classes)

Design Rationale:
Sharing the encoder improves learning efficiency and encourages feature reuse across tasks, benefiting both segmentation and classification performance.

🗂️ Dataset & Preprocessing
Segmentation Data: IDRiD optic disc masks.

Classification Data: DR and DME labels from IDRiD annotation CSVs.

Preprocessing Techniques:

Resize to 384×384.

Normalization with ImageNet statistics.

Masks binarized; classification labels parsed from CSV.

🏋️ Multi-task Training Strategy
Training Loop: Alternating between segmentation and classification tasks each epoch.

Loss Functions:

Segmentation: BCEWithLogitsLoss

Classification: CrossEntropyLoss (averaged for DR & DME tasks)

Optimizer: Adam (LR = 1e-4)

Mixed Precision: Enabled with torch.cuda.amp

Epochs: 50

Checkpointing: Every 2 epochs, saved with Dice score in filename.

🧪 Evaluation Metrics & Performance
Segmentation:

IoU and Dice Score

Classification:

Accuracy for both DR grading and DME risk prediction

Epochs	IoU	Dice	DR Accuracy	DME Accuracy
10	0.6904	0.8016	0.6019	—
20	0.7670	0.8108	0.8934	0.6117
50	0.8058	0.8063	0.8902	0.5922

Note: Final results reflect a trade-off between segmentation precision and classification performance.

📊 Visualizations & Comparative Analysis
Segmentation predictions evaluated against ground truth.

Accuracy trends and Dice scores plotted over training epochs.

Multi-task model performance compared against single-task benchmarks.

📝 Conclusion
The single-task segmentation model achieved strong performance with Dice > 0.77 on higher-resolution trials.

The multi-task model maintained competitive segmentation results while successfully predicting DR grading and DME risk with significant accuracy.

The joint training setup leveraged shared encoder features, validating the multi-task learning hypothesis.

🛠️ Technologies & Frameworks Used
PyTorch (Core framework)

torchvision (Feature extractor and transforms)

Albumentations (Data augmentations)

Mixed Precision (AMP) with torch.cuda.amp

IDRiD Dataset for medical image analysis tasks

🔥 Future Scope
Hyperparameter tuning for balancing segmentation and classification losses.

Incorporating advanced augmentations like CutMix or MixUp.

Exploring EfficientNetV2 or ConvNeXt as potential backbones.

Validation on cross-domain datasets to test generalization.

