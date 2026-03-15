# The Health Portfolio: Severity-Weighted OR-Gate Evaluation of Clinical AI Ensembles

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19039636.svg)](https://doi.org/10.5281/zenodo.19039636)

## Papers

**NPJ Digital Medicine (in preparation):**
Deobhakta, A. A. (2026). "Severity-weighted utility and concordant-miss analysis reveal clinically important complementarity in multi-AI diabetic retinopathy screening systems that standard discrimination metrics can miss."

**SSRN Working Paper (posted):**
Deobhakta, A. A. (2026). "Concordance Collapse and the Health Portfolio: Why the Metric You Choose Determines Whether AI Ensembles Save Patients or Waste Resources." Working Paper, SSRN. https://doi.org/10.2139/ssrn.XXXXXXX

## Summary

Standard evaluation metrics (AUC) show no ensemble benefit when pairing clinical AI screening models. Severity-weighted OR-gate evaluation reveals that architecturally diverse ensembles reduce the risk of missing sight-threatening diabetic retinopathy by 39–68% across two independent datasets (EyePACS and APTOS). The key findings:

- **AUC says ensembles don't help.** 51 of 55 model pairs show zero or negative AUC lift.
- **Severity-weighted OR-gate evaluation says they do.** The best pair reduces severity-weighted cost by 39% (EyePACS) and 68% (APTOS).
- **96% of the benefit comes from the OR-gate rule**, not the evaluation metric (2×2 factorial decomposition).
- **OR-gate dominates threshold-optimized single models** at matched operating points.
- **Error correlation (ρ) predicts concordant-miss risk** (r = 0.665 [95% CI: 0.456–0.797]).
- **Results are robust** across 5 clinically plausible severity-weight scenarios and bootstrap resampling (n = 1,000).

## Repository Structure

```
health-portfolio/
├── configs/              # Model training configuration files
├── scripts/
│   ├── dataset.py        # Data loading and preprocessing (EyePACS, APTOS)
│   ├── models.py         # Model definitions (DenseNet121, EfficientNet, ResNet50, ViT, RETFound)
│   └── evaluation.py     # Evaluation framework (AUC, OR-gate, severity-weighted cost)
├── notebooks/            # Colab notebooks for training and analysis
│   ├── notebook1_data_setup.ipynb
│   ├── notebook2_training.ipynb
│   └── notebook3_robustness_analysis.ipynb
├── results/
│   └── npj_robustness_results.json   # Key results for NPJ revision
├── requirements.txt
├── LICENSE
└── README.md
```

## Models

11 deep learning models spanning 5 architectures, each in binary and 5-class formulations:

| Architecture | Binary | 5-Class | Notes |
|---|---|---|---|
| DenseNet121 | ✓ | ✓ | Best individual model (5-class) |
| EfficientNet-B3 | ✓ | ✓ | Top ensemble partner |
| ResNet50 | ✓ | ✓ | |
| ViT-Base | ✓ | ✓ | |
| RETFound | ✓ | ✓ | Retinal foundation model (Zhou et al. 2023) |
| RETFound (adversarial) | ✓ | — | Adversarial decorrelation variant |

## Datasets

- **EyePACS**: 88,702 retinal fundus images, 5-level DR severity grading. [Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection)
- **APTOS 2019**: 3,662 retinal images, same grading scheme (external validation). [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection)

Both datasets require accepting competition rules on Kaggle before download.

## Reproducing Results

### Requirements
```bash
pip install -r requirements.txt
```

### Step 1: Data Setup
Run `notebooks/notebook1_data_setup.ipynb` in Google Colab (GPU required). This downloads and preprocesses both datasets. Requires a Kaggle API token.

### Step 2: Model Training
Run `notebooks/notebook2_training.ipynb`. Trains all 11 models on EyePACS. Approximate time: 48 GPU-hours on NVIDIA T4.

### Step 3: Evaluation and Robustness Analysis
Run `notebooks/notebook3_robustness_analysis.ipynb`. Generates all results including threshold sweep, 2×2 decomposition, weight robustness, and bootstrap CIs. Output saved to `results/npj_robustness_results.json`.

## Severity-Weighted Cost Framework

| Error Type | Weight | Clinical Basis |
|---|---|---|
| Missed Grade 4 (PDR) | 1,000 | QALY loss 0.214/yr; 50% 1-yr progression to blindness |
| Missed Grade 3 (severe NPDR) | 500 | Lower but substantial progression risk |
| False positive | 1 | Cost of unnecessary referral |

Sources: Moshfeghi et al. (2020), ETDRS Report No. 9, Javitt et al. (1996).

## Key Results

### Top Ensemble: DenseNet121 (5-class) + EfficientNet (5-class)

| Metric | EyePACS | APTOS |
|---|---|---|
| Solo severity cost | 6,468 [4,435–8,812] | 1,570 |
| OR-gate severity cost | 3,972 [2,820–5,554] | 499 |
| Cost reduction | 39.2% | 68.2% |
| Concordant severe misses | 6 [2–11] | 1 |
| Error correlation (ρ) | 0.549 [0.497–0.595] | 0.77 |

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{deobhakta2026health_portfolio,
  title={Severity-weighted utility and concordant-miss analysis reveal clinically important complementarity in multi-{AI} diabetic retinopathy screening systems that standard discrimination metrics can miss},
  author={Deobhakta, Avnish Arvind},
  year={2026},
  note={In preparation for NPJ Digital Medicine}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

**Avnish Arvind Deobhakta, M.D.**
Department of Ophthalmology, The New York Eye and Ear Infirmary of Mount Sinai
Icahn School of Medicine at Mount Sinai, New York, NY
