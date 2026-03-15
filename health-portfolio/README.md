# The Health Portfolio: Severity-Weighted OR-Gate Evaluation of Clinical AI Ensembles

**Paper:** Deobhakta, A. A. (2026). "Concordance Collapse and the Health Portfolio: Why the Metric You Choose Determines Whether AI Ensembles Save Patients or Waste Resources." Working Paper, SSRN.

## Summary

Standard evaluation metrics (AUC) show no ensemble benefit when pairing clinical AI screening models. Severity-weighted OR-gate evaluation reveals that architecturally diverse ensembles reduce the risk of missing sight-threatening diabetic retinopathy by 39-68% across two independent datasets, with zero concordant grade 4 misses achievable.

The evaluation methodology, not the ensemble method, was the bottleneck all along.

## Key Findings

| Finding | EyePACS | APTOS |
|---------|---------|-------|
| Best single model AUC | 0.911 | 0.964 |
| AUC-based ensemble lift | +0.006 (max) | minimal |
| OR-gate cost reduction | -39% | -68% |
| Zero G4 concordant misses | 3 pairs | 2 pairs |
| rho vs concordant miss (r) | 0.73 | 0.33 |

## Repository Structure

```
health-portfolio/
  README.md                    # This file
  requirements.txt             # Python dependencies
  notebooks/
    01_data_preparation.py     # Download and preprocess EyePACS + APTOS
    02_train_models.py         # Train all 11 models
    03_evaluate_eyepacs.py     # Full evaluation on EyePACS test set
    04_evaluate_aptos.py       # External validation on APTOS
    05_or_gate_analysis.py     # OR-gate severity-weighted analysis
  configs/
    severity_weights.json      # Literature-derived QALY-based weights
    model_configs.json         # Architecture and training hyperparameters
  scripts/
    dataset.py                 # Dataset classes (binary and 5-class)
    models.py                  # Model definitions including RETFound wrappers
    evaluation.py              # Evaluation functions (AUC, OR-gate, severity cost)
    utils.py                   # Shared utilities
  results/
    eyepacs_individual.csv     # Individual model results on EyePACS
    eyepacs_pairwise.csv       # All 55 pairwise results on EyePACS
    aptos_individual.csv       # Individual model results on APTOS
    aptos_pairwise.csv         # All 55 pairwise results on APTOS
```

## Reproducing the Results

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- An NVIDIA GPU with at least 16GB VRAM (A100 recommended for RETFound models)
- A HuggingFace account with access to [RETFound](https://huggingface.co/iszt/RETFound_mae_meh)

### Step 1: Environment Setup

```bash
git clone https://github.com/[username]/health-portfolio.git
cd health-portfolio
pip install -r requirements.txt
huggingface-cli login  # Enter your HF token for RETFound access
```

### Step 2: Download Data

Download the following datasets from Kaggle (requires a Kaggle account):

- **EyePACS:** [Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection)
- **APTOS:** [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)

Place the downloaded files in `data/eyepacs/raw/` and `data/aptos/raw/` respectively.

### Step 3: Preprocess Data

```bash
python notebooks/01_data_preparation.py
```

This resizes all images to 256x256 PNG, creates stratified train/val/test splits for EyePACS (70/15/15), and prepares APTOS as a full external test set.

### Step 4: Train Models

```bash
python notebooks/02_train_models.py
```

Trains all 11 models sequentially, saving checkpoints to `models/`. Each model skips training if a checkpoint already exists (safe for interrupted runs). Estimated time: 4-6 hours on a single A100 GPU.

### Step 5: Evaluate

```bash
python notebooks/03_evaluate_eyepacs.py
python notebooks/04_evaluate_aptos.py
python notebooks/05_or_gate_analysis.py
```

Generates all results tables and figures reported in the paper.

## Models

| Model | Architecture | Task | Parameters | EyePACS AUC | APTOS AUC |
|-------|-------------|------|------------|-------------|-----------|
| densenet121_binary | DenseNet121 | Binary | 8M | 0.897 | 0.959 |
| densenet121_5class | DenseNet121 | 5-class | 8M | 0.911 | 0.958 |
| efficientnet_b3_binary | EfficientNet-B3 | Binary | 12M | 0.859 | 0.949 |
| efficientnet_b3_5class | EfficientNet-B3 | 5-class | 12M | 0.867 | 0.954 |
| resnet50_binary | ResNet50 | Binary | 25M | 0.878 | 0.958 |
| resnet50_5class | ResNet50 | 5-class | 25M | 0.872 | 0.958 |
| vit_base_binary | ViT-Base | Binary | 86M | 0.848 | 0.944 |
| vit_base_5class | ViT-Base | 5-class | 86M | 0.848 | 0.950 |
| retfound_binary | RETFound | Binary | 304M | 0.870 | 0.963 |
| retfound_5class | RETFound | 5-class | 304M | 0.881 | 0.959 |
| retfound_adversarial | RETFound | Adversarial | 304M | 0.870 | 0.964 |

## Severity Weights

Derived from clinical economics literature:

| Error Type | Weight | Source |
|-----------|--------|--------|
| Grade 4 miss (PDR) | 500 | DRCR Protocol S; Schmier et al. 2020 |
| Grade 3 miss (Severe NPDR) | 200 | ETDRS (50% progression to PDR in 1 year) |
| Grade 2 miss (Moderate NPDR) | 15 | ETDRS (23% progression in 4 years) |
| Grade 1 miss | 2 | Low immediate risk |
| False positive | 1 | Unnecessary referral ($200-500) |

## Citation

```bibtex
@article{deobhakta2026concordance,
  title={Concordance Collapse and the Health Portfolio: Why the Metric You Choose Determines Whether AI Ensembles Save Patients or Waste Resources},
  author={Deobhakta, Avnish Arvind},
  year={2026},
  journal={Working Paper, SSRN}
}
```

## Related Papers

- Deobhakta, A. A. (2026a). "Within-Task Competence Heterogeneity and the Shape of Technological Displacement." SSRN.
- Deobhakta, A. A. (2026b). "Estimating Within-Task Competence Heterogeneity (kappa)." SSRN.
- Deobhakta, A. A. (2026c). "Vetter's Paradox in Real Data." SSRN.

## License

MIT License. See [LICENSE](LICENSE) for details.
