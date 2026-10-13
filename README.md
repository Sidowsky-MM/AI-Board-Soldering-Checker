# AI-Board-Soldering-Checker

Automated optical inspection (AOI) pipeline implemented in MATLAB for classifying solder joint defects using deep learning architectures and hybrid classifiers. The workflow includes image preprocessing via localized patch extraction on an open-source dataset, followed by model training and evaluation.

Dataset source: [Kaggle - Solder Joint Defect Detection](https://www.kaggle.com/datasets/akshaydattatraykhare/solder-joint-defect-detection)

## Pipeline & Implementation Details

The core execution logic is implemented in `main.m` and follows a non-destructive, in-memory processing pipeline:

### 1. Data Handling & Preprocessing
* Raw images are loaded directly from the source directory using MATLAB `imageDatastore` with subfolder label parsing.
* The dataset is split into training (70%), validation (15%), and test (15%) subsets in memory.
* On-the-fly data augmentation is applied via `augmentedImageDatastore` (random rotation within ±10° and spatial translations within ±5 px) without altering the original image files on disk.

### 2. Model Architectures & Training
* Custom CNN: Built from scratch with 3 convolutional blocks (32, 64, 128 filters of size 3x3), batch normalization, ReLU activations, max-pooling, and dropout (0.5), trained using Adam.
* SVM (HOG): Features are extracted using Histogram of Oriented Gradients (HOG) after grayscale conversion and resizing to 64x64, classified using an ECOC multiclass SVM with an RBF kernel and evaluated via 5-fold cross-validation.
* ResNet-50: Transfer learning implementation replacing the final fully connected and classification layers to match the target defect classes, fine-tuned with Adam.

### 3. Generated Artifacts
All input images remain unmodified in their original directories. The script produces analytical outputs directly in the workspace directory:
* High-resolution confusion matrices (`confmat_cnn_own.png`, `confmat_svm.png`, `confmat_resnet.png`).
* Comparative performance charts (`bar_accuracies.png`, `bar_f1_score.png`).
* Exported metrics tables (`per_class_metrics.csv`, `accuracy_porownanie.csv`, `f1_macro_porownanie.csv`).

## Performance Benchmark

### Overall Model Performance

| Model | Accuracy | Macro F1-Score |
| :--- | :--- | :--- |
| Custom CNN | 98.23% | 0.92 |
| SVM-CNN | 97.64% | N/A |
| ResNet-50 | 97.83% | 0.91 |
| SVM-ResNet | 98.43% | N/A |

Total training time: 696.65 seconds

### Class-Wise F1-Score Comparison

| Class / Defect Type | Custom CNN | ResNet-50 |
| :--- | :--- | :--- |
| Normal | 1.00 | 1.00 |
| Rotated Component | 0.81 | 0.94 |
| Short | 0.92 | 0.89 |
| Insufficient Solder | 0.95 | 0.91 |
| Excess Solder | 0.90 | 0.83 |
