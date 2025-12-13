# Bearing Fault Diagnosis Using Vision Transformers

This project investigates **bearing fault diagnosis** through **spectrogram-based image analysis** using **Vision Transformer (ViT) models** and their variants. It addresses the challenges of **cross-domain generalization** and **model bias** in industrial datasets (UORED-VAFCLS and CWRU).

---

## Author and Scholar

- **Ana Beatriz Cardoso**  
  Google Scholar: https://scholar.google.com/citations?user=vpEq5QIAAAAJ

If this repository or its results are useful in your research, please cite it and the related publications listed on the profile above.

---

## Project Overview

- **Spectrogram Generation**  
  Raw vibration signals are converted into spectrogram images.  
  - Preprocessing options: `none`, `RMS normalization`, `Z-score normalization`  
  - Default STFT settings: `nfft=2048`, `fs=42000 Hz`, `nperseg=1024`, `noverlap=896`, `window='hann'`

- **Models**  
  Five state-of-the-art ViT-family models are evaluated for spectrogram classification:
  - **ViT (Vision Transformer)** (Dosovitskiy et al., 2020), `google/vit-base-patch16-224`
  - **DeiT** (Touvron et al., 2021), `facebook/deit-base-patch16-224`
  - **DINOv2WithRegisters** (Oquab et al., 2023; Darcet et al., 2024), `facebook/dinov2-with-registers-small`
  - **SwinV2** (Liu et al., 2022), `microsoft/swinv2-tiny-patch4-window8-256`
  - **MAE** (He et al., 2021), `facebook/vit-mae-base`  
  All models are fine-tuned from ImageNet-21k pretrained weights.

- **Training Strategies**
  - **K-Fold cross-validation** for train and validation
  - **Strict domain splits** to avoid leakage into test
  - **Final fine-tuning** on the full training set after cross-validation

- **Bias Mitigation Focus**  
  Prevent overfitting and reduce domain similarity bias during evaluation.

---

## Results Summary

- High validation accuracy during K-Fold (up to ~99% in some setups).
- Independent test accuracy varies with preprocessing and backbone.
- Best performance observed with **domain-appropriate preprocessing** and ViT backbones.

---

## Datasets

- **UORED-VAFCLS**: multiple sensors, natural and seeded faults, realistic domain splits.  
- **CWRU**: motor test rig with seeded faults; beware of instance reuse when defining domains.

> Ensure that train, validation, and test partitions are **domain disjoint**. Do not mix segments from the same bearing instance across splits.

---

## Spectrogram Generation

- **Transform**: STFT to 2D spectrograms; amplitudes converted to dB.
- **Defaults**: `fs=42000`, `nfft=2048`, `nperseg=1024`, `noverlap=896`, `window='hann'`.  
- **Normalization**: `none`, `rms`, or `zscore`.
- **Export**: RGB images for compatibility with vision models.

Parameters can be adjusted in the spectrogram creation code to match experimental design.

---

## Project Structure

```
├── data/
│   └── spectrograms/           # Generated spectrogram images
├── scripts/
│   └── evaluate_model_vitclassifier.py  # Training, evaluation, K-Fold
├── logs/
│   └── experiment_logs/        # Attention maps, confusion matrices, results
├── experiment_runner.py        # Experiment runner
├── README.md                   # This file
```

---

## Installation

1) Clone the repository
```bash
git clone https://github.com/abcardoso/vit-bearing-faults.git
cd vit-bearing-faults
```

2) Create and activate the environment (recommended)
```bash
conda create -n vit_env python=3.10
conda activate vit_env
pip install -r requirements.txt
```

> PyTorch with CUDA is recommended for training transformer backbones.

---

## How to Run an Experiment

```bash
python experiment_runner.py
```

You can customize:
- `model_type` (e.g., `vit`, `deit`, `dinov2`, `swinv2`, `mae`)
- `preprocessing` (`none`, `rms`, `zscore`)
- Training domains and held-out test domain
- Learning rate, epochs, batch size, optimizer, etc.

> The training loop resets weights to the same initialization at the start of each fold to ensure independence across folds.

---

## Results and Logging

- Metrics and artifacts are written under `logs/experiment_logs/`.  
- Appendix A results https://github.com/abcardoso/vit-bearing-faults/blob/main/appendix_a.md or https://github.com/abcardoso/vit-bearing-faults/blob/main/appendix_a_results.tsv 
- Recommended reporting:
  - Accuracy, precision, recall, F1
  - Confidence intervals across folds
  - Paired significance tests across fold-matched runs
  - Wall-clock time and memory footprint for training and inference

---

## Reproducibility Checklist

- Fix global and dataloader seeds  
- Record STFT parameters, normalization, and image export settings  
- Store file lists per split to ensure domain disjointness  
- Reset model weights per fold and document pretraining source  
- Report hardware, driver, and library versions

---

## Sustainability Note

Track:
- **Wall-clock** training and inference time  
- **Peak memory** allocation  
- Optional energy estimates, e.g., with `codecarbon`

These signals can support fair trade-offs between accuracy, latency, and resource usage.

---

## References

- Vision Transformer: https://arxiv.org/abs/2010.11929  
- DeiT: https://arxiv.org/abs/2012.12877  
- DINOv2: https://arxiv.org/abs/2304.07193 and Darcet et al., 2024  
- SwinV2: https://arxiv.org/abs/2111.09883  
- MAE: https://arxiv.org/abs/2111.06377  
- UORED dataset paper: https://doi.org/10.1016/j.ymssp.2023.110055

---

## How to Cite

### Software

```bibtex
@misc{cardoso_vit_bearing_faults_2025,
  title        = {Spectrogram-based Vision Transformers for Bearing Fault Diagnosis},
  author       = {Cardoso, Ana Beatriz},
  year         = {2025},
  howpublished = {\url{https://github.com/abcardoso/vit-bearing-faults}},
  note         = {Code and experiments for UORED-VAFCLS and CWRU with domain-split evaluation}
}
```

### Related publications

See the most up-to-date list on Google Scholar:  
https://scholar.google.com/citations?user=vpEq5QIAAAAJ

---

## Acknowledgements

This work builds on public datasets and open-source backbones. Please review dataset and model licenses before redistribution.

---

## License

See `LICENSE` in this repository.
