# PyTorch Image Classification: CIFAR‑10 Baseline + Food‑101 Fine‑Tuning
An end‑to‑end image classification project that spans data preparation, training/fine‑tuning, evaluation/inference, and experiment comparison/visualization. It includes:

- SimpleCNN baseline (trained on CIFAR‑10)
- ResNet18 / ResNet50 fine‑tuning (trained on Food‑101; AMP, Cosine+Warmup, Label Smoothing, augmentation)
Comparison scripts and plots are provided 

## Highlights
- Training pipeline: AMP (mixed precision), backbone freezing/unfreezing, per‑group learning rates, Cosine schedule + Warmup, Label Smoothing
- Augmentation: RandAugment / AutoAugment / ColorJitter + RandomErasing
- Inference tools: single image and directory batch, with CIFAR‑10 class names and ImageNet 1000‑class labels
- Visualization & comparison: compare_results.csv, bar plots (acc/loss), validation accuracy curves
- Data handling: CIFAR‑10 auto‑download; Food‑101 auto or manual download with expected directory layout
## Structure
```
.
├─ models/
│  └─ simple_cnn.py                 
# SimpleCNN baseline
├─ train_simple_cnn.py              
# SimpleCNN train/eval/infer entry
├─ fine-tune_resnet.py              
# ResNet18/50 fine-tuning & 
inference (AMP/aug/freeze)
├─ compare_experiments.py           
# Comparison & plotting (CSV + PNG)
├─ data/
│  ├─ cifar-10-batches-py/          
# CIFAR-10 auto-download
│  └─ food-101/                     
# Food-101 auto/manual download + 
extract
│     ├─ images/                    
# 101 class directories
│     └─ meta/                      
# classes.txt, train.txt, test.txt
└─ outputs/
   ├─ best_model_simple_cnn.pt      
   # SimpleCNN best (CIFAR-10)
   ├─ best_resnet18_food101.pt      
   # ResNet18 best (Food-101)
   ├─ best_resnet50_food101.pt      
   # ResNet50 best (Food-101)
   ├─ checkpoint_*.pt               
   # checkpoints (per arch/dataset)
   ├─ last_*.pt                     
   # last-epoch weights
   ├─ train_resnet18_food101.log    
   # ResNet18 training log
   ├─ train_resnet50_food101.log    
   # ResNet50 training log
   ├─ compare_results.csv           
   # comparison table
   ├─ compare_bar_acc.png           
   # acc bar plot (optional)
   ├─ compare_bar_loss.png          
   # loss bar plot (optional)
   └─ train_curves_food101.png      
   # val_acc curve plot (optional)
```
## Requirements
- Python 3.11+
- PyTorch 2.x (CUDA recommended)
- torchvision
- matplotlib (for plotting, optional)
- Windows or Linux (examples use Windows PowerShell; Git Bash is similar)
Install (example):

```
pip install torch torchvision 
matplotlib
```
## Data
- CIFAR‑10: auto‑download to ./data/cifar-10-batches-py
- Food‑101:
  - auto‑download supported, or manual download/extract under ./data/food-101
  - layout must include:
    - images/ (101 class dirs)
    - meta/ (at least classes.txt , train.txt , test.txt )
  - avoid nested folders like food-101/food-101/images ; ensure meta/meta/* is moved up to meta/ if needed
## Training & Evaluation
### SimpleCNN (CIFAR‑10)
- Train (recommended starter):
```
python train_simple_cnn.py ^
  --dataset CIFAR10 ^
  --epochs 15 ^
  --batch-size 256 ^
  --workers 4 ^
  --lr 0.001 ^
  --weight-decay 0.0005 ^
  --scheduler cosine ^
  --width-mult 1.5 ^
  --amp
```
- Evaluate (best weights):
```
python train_simple_cnn.py ^
  --dataset CIFAR10 ^
  --test-only ^
  --resume 
  outputs\best_model_simple_cnn.pt ^
  --width-mult 1.5 ^
  --amp
```
- Single‑image inference:
```
python train_simple_cnn.py ^
  --dataset CIFAR10 ^
  --predict "D:\path\to\image.jpg" ^
  --resume 
  outputs\best_model_simple_cnn.pt ^
  --width-mult 1.5 ^
  --topk 5 ^
  --amp
```
### ResNet18 (Food‑101)
- Train (freeze 3 epochs → fine‑tune 20 epochs):
```
python fine-tune_resnet.py ^
  --arch resnet18 ^
  --pretrained ^
  --batch-size 128 ^
  --workers 4 ^
  --epochs 20 ^
  --freeze-epochs 3 ^
  --lr-head 0.001 ^
  --lr-backbone 0.0001 ^
  --scheduler cosine ^
  --amp
```
- Evaluate (Food‑101 test):
```
python fine-tune_resnet.py ^
  --arch resnet18 ^
  --test-only ^
  --resume 
  outputs\best_resnet18_food101.pt ^
  --amp
```
### ResNet50 (Food‑101)
- Train (augmentation + warmup + label smoothing):
```
python fine-tune_resnet.py ^
  --arch resnet50 ^
  --pretrained ^
  --batch-size 64 ^
  --workers 4 ^
  --epochs 25 ^
  --freeze-epochs 3 ^
  --lr-head 0.001 ^
  --lr-backbone 0.0001 ^
  --scheduler cosine ^
  --warmup-epochs 2 ^
  --label-smoothing 0.1 ^
  --aug rand ^
  --rand-magnitude 9 ^
  --amp
```
- Evaluate (Food‑101 test):
```
python fine-tune_resnet.py ^
  --arch resnet50 ^
  --test-only ^
  --resume 
  outputs\best_resnet50_food101.pt ^
  --amp
```
## General Inference (ImageNet 1000 classes)
- Pretrained ResNet inference (no training needed):
```
python fine-tune_resnet.py ^
  --predict "D:\path\to\image.jpg" ^
  --arch resnet18 ^
  --pretrained ^
  --imagenet-classes ^
  --topk 5 ^
  --amp
```
## Comparison & Plots
- Run comparison and generate CSV/plots:
```
python compare_experiments.py --plot
```
- Outputs:
  - outputs/compare_results.csv
  - outputs/compare_bar_acc.png
  - outputs/compare_bar_loss.png
  - outputs/train_curves_food101.png (if logs are present)
## Example Results 
- Food‑101 (ResNet18): val_acc ≈ 0.8305, val_loss ≈ 0.6199
- Food‑101 (ResNet50): val_acc ≈ 0.8963, val_loss ≈ 0.4757
- CIFAR‑10 (SimpleCNN): val_acc ≈ 0.8195, val_loss ≈ 0.5174
## Troubleshooting
- Device‑side assert (CUDA): usually caused by mismatch between dataset label count and model output dimension (e.g., using a 5‑class head to evaluate 10‑class data). The comparison script forces the current dataset’s class count when rebuilding SimpleCNN and prunes mismatched weights to avoid crashes.
- OOM: reduce --batch-size , enable --amp , or use --accumulate-steps (supported in ResNet script).
- Empty or slow logs: training scripts default to printing at end of each epoch; ResNet supports --log-interval to print step‑level progress.
- Food‑101 layout errors: ensure data/food-101/images and data/food-101/meta are at the same level; fix nested meta/meta or double food-101/food-101 .
## Acknowledgments
- Datasets: Food‑101 (food classification), CIFAR‑10 (baseline)
- Models & weights: torchvision (ImageNet‑pretrained ResNet family)
