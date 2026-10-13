# AI-Board-Soldering-Checker

Automated optical inspection (AOI) pipeline implemented in MATLAB for classifying solder joint defects using deep learning architectures and hybrid classifiers. The workflow includes image preprocessing via localized patch extraction on an open-source dataset, followed by model training and evaluation.

Dataset source: [Kaggle - Solder Joint Defect Detection](https://www.kaggle.com/datasets/akshaydattatraykhare/solder-joint-defect-detection)

## Performance Benchmark

### Overall Model Performance

| Architecture | Accuracy | Macro F1-Score |
| :--- | :--- | :--- |
| Custom CNN | 98.23% | 0.92 |
| ResNet-50 | 97.83% | 0.91 |

Total training time: 696.65 seconds

### Class-Wise F1-Score Comparison

| Class / Defect Type | Custom CNN | ResNet-50 |
| :--- | :--- | :--- |
| Normal | 1.00 | 1.00 |
| Rotated Component | 0.81 | 0.94 |
| Solder Bridge / Short | 0.92 | 0.89 |
| Insufficient Solder | 0.95 | 0.91 |
| Excess Solder | 0.90 | 0.83 |
