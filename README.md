# AI-Board-Soldering-Checker

Automated optical inspection (AOI) pipeline implemented in MATLAB for classifying solder joint defects using deep learning architectures and hybrid classifiers. The workflow includes image preprocessing via localized patch extraction on an open-source dataset, followed by model training and evaluation.
Test objects was created by other creator earlier searched on github. 
There is a links to a github repo:
https://www.kaggle.com/datasets/akshaydattatraykhare/solder-joint-defect-detection
https://www.kaggle.com/datasets/mauriziocalabrese/soldef-ai-pcb-dataset-for-defect-detection
## Performance Benchmark

| Model | Accuracy |
| :--- | :--- |
| CNN | 98.62% |
| SVM-CNN | 97.64% |
| ResNet | 97.83% |
| SVM-ResNet | 98.43% |

Total training time: 696.65 seconds
