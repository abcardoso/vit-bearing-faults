# Bearing Fault Diagnosis Using Vision Transformers

This project investigates **bearing fault diagnosis** through **spectrogram-based image analysis** using **Vision Transformer (ViT) models** and their variations. It aims to address the challenges of **cross-domain generalization** and **model bias** in industrial datasets.

## Project Overview

- **Spectrogram Generation**:  
  Raw vibration signals from the UORED dataset are converted into spectrogram images.  
  - Preprocessing options: `none`, `RMS normalization`, `Z-score normalization`.
  - Consistent spectrogram settings: `nfft=2048`, `fs=42000 Hz`, `nperseg=1024`, `noverlap=896`, `window='hann'`.

- **Models**:

  This study leverages five state-of-the-art ViT models, specifically chosen for their proven capabilities in image-based tasks and spectrogram classification. Each architecture, detailed below, was selected to evaluate their suitability for bearing fault diagnosis using spectrogram images:

  - **ViT (Vision Transformer)**: [Dosovitskiy et al., 2020]  
    Baseline Vision Transformer model, characterized by global attention on image patches, utilizing Google’s `google/vit-base-patch16-224` model.

  - **DeiT (Data-efficient Image Transformer)**: [Touvron et al., 2021]  
    ViT with knowledge distillation, enhancing data efficiency and reducing model size, utilizing `facebook/deit-base-patch16-224` model.

  - **DINOv2WithRegisters**: [Darcet et al., 2024; Oquab et al., 2023]  
    Leverages self-supervised training for robust feature extraction, facilitating better generalization, utilizing model `facebook/dinov2-with-registers-small`.

  - **SwinV2 (Shifted Window Transformer V2)**: [Liu et al., 2022]  
    Incorporates shifted window attention to handle spatial hierarchies efficiently, utilizing `microsoft/swinv2-tiny-patch4-window8-256`.

  - **MAE (Masked Autoencoder)**: [He et al., 2021]  
    Applies masking strategies in self-supervised pretraining, enabling the model to focus on relevant image features, using model `facebook/vit-mae-base`.

  Each model underwent **transfer learning** from **ImageNet-21k** pretrained weights to fine-tune their feature extraction capabilities specifically for spectrogram-based fault diagnosis.

- **Training Strategies**:
  - **K-Fold Cross-Validation** applied to training and validation splits.
  - **Separate Train/Test Domains**: Ensures no test data leakage into training.
  - **Final Fine-Tuning**: After K-Fold, final model trained on the full training set.

- **Bias Mitigation Focus**:  
  Special emphasis is placed on preventing overfitting and reducing domain similarity bias during model evaluation.

## Results Summary

- Achieved high validation accuracy during K-Fold (up to ~99% for some setups).
- Final independent test accuracy varies significantly depending on preprocessing and model.
- Best performance observed when models are appropriately tuned with **domain-specific preprocessing**.

## Project Structure

```
├── data/
│   └── spectrograms/    # Generated spectrogram images
├── scripts/
│   └── evaluate_model_vitclassifier.py  # Model training, evaluation, K-Fold
├── logs/
│   └── experiment_logs/  # Attention maps, confusion matrices, results
├── main.py               # Experiment runner
├── README.md              # (this file)
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/abcardoso/vit-bearing-faults.git
   cd vit-bearing-faults
   ```
2. Install the environment (recommended):
   ```bash
   conda create -n vit_env python=3.10
   conda vit_env
   pip install -r requirements.txt
   ```

## How to Run an Experiment

```bash
python experiment_runner.py
```

You can customize:
- Model (`model_type`)
- Preprocessing (`preprocessing`)
- Training domains and test domain
- Learning rates, number of epochs, etc.

## References

- Vision Transformer: [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)
- DeiT: [Touvron et al., 2021](https://arxiv.org/abs/2012.12877)
- DINOv2: [Oquab et al., 2023](https://arxiv.org/abs/2304.07193)
- SwinV2: [Liu et al., 2022](https://arxiv.org/abs/2111.09883)
- MAE: [He et al., 2021](https://arxiv.org/abs/2111.06377)
- UORED Dataset: [Sehri et al., 2023](https://doi.org/10.1016/j.ymssp.2023.110055)

