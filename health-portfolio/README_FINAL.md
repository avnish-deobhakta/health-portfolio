# The Health Portfolio: Severity-Weighted OR-Gate Evaluation of Clinical AI Ensembles

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19153932.svg)](https://doi.org/10.5281/zenodo.19153932)

## Papers

**NPJ Digital Medicine (in preparation):**
Deobhakta, A. A., Rosen, R. B., Tai, T. Y. T., Tsai, J. C., Harris, A., & Pasquale, L. (2026). "Hidden safety value in multi-AI screening: a severity-weighted evaluation of diabetic retinopathy ensembles."

**SSRN Working Paper (posted):**
Deobhakta, A. A. (2026). "Concordance Collapse and the Health Portfolio: Why the Metric You Choose Determines Whether AI Ensembles Save Patients or Waste Resources." Working Paper, SSRN. https://doi.org/10.2139/ssrn.XXXXXXX

## Summary

Standard evaluation metrics (AUC) show little or no apparent ensemble benefit across nearly all model pairings. Severity-weighted OR-gate evaluation reveals that architecturally diverse ensembles reduce the risk of missing sight-threatening diabetic retinopathy, with the best observed ensemble reductions ranging from 38.6% on EyePACS to 68.2% on APTOS, depending on model pairing and formulation. The key findings:

- **AUC suggests little apparent ensemble benefit.** 54 of 55 model pairs show zero or negative AUC lift.
- **Severity-weighted OR-gate evaluation detects clinically meaningful ensemble benefit.** The best EyePACS pair reduces severity-weighted cost by 38.6%; the best APTOS pair (a different model combination) achieves a 68.2% reduction.
- **In a 2×2 factorial decomposition, approximately 96% of the observed benefit in the lead analysis was attributable to the OR-gate decision rule** rather than the evaluation metric itself.
- **OR-gate dominates threshold-optimized single models** at matched operating points.
- **Error correlation (ρ) predicts concordant-miss risk** (r = 0.712 [95% CI: 0.456–0.797]).
- **Results are robust** across 5 clinically plausible severity-weight scenarios and bootstrap resampling (n = 1,000).

## Repository Structure

```
health-portfolio/
├── configs/
│   ├── model_configs.json              # Model training configurations (all 11 models)
│   └── severity_weights.json           # Severity-weight derivation, values, and robustness scenarios
├── scripts/
│   ├── dataset.py                      # Data loading and preprocessing (EyePACS, APTOS)
│   ├── models.py                       # Model definitions (DenseNet121, EfficientNet, ResNet50, ViT, RETFound)
│   ├── evaluation.py                   # Evaluation framework (AUC, OR-gate, severity-weighted cost)
│   └── reproduce_results.py            # One-command reproduction from frozen predictions
├── notebooks/
│   ├── notebook1_data_setup.ipynb
│   ├── notebook2_model_training.ipynb
│   └── notebook3_evaluation_robustness.ipynb
├── results/
│   ├── comprehensive_evaluation.json   # Legacy: 55-pair results from in-memory Colab run (superseded by csv/ for both datasets; retained for archival completeness)
│   ├── npj_robustness_results.json     # Threshold sweep, bootstrap CIs
│   ├── exhaustive_pairwise_results.json
│   ├── full_evaluation_results.json
│   ├── or_gate_analysis.json
│   ├── predictions/                    # Frozen per-image predictions (11 models × 2,900 images)
│   └── csv/                            # Human-readable summary tables
├── examples/
│   └── generate_toy_data.py            # Synthetic data for pipeline validation
├── REPRODUCIBILITY_MAP.md              # Every manuscript claim mapped to its data source
├── requirements.txt
├── requirements-lock.txt               # Frozen environment (exact package versions)
├── LICENSE
└── README.md
```

Within `results/`, the authoritative pairwise outputs are the CSV files in `results/csv/`; legacy JSON files are preserved for archival completeness and for artifacts not yet regenerated from frozen predictions. See `results/README.md` for details.

## Reproducing Results

### What is reproducible from included artifacts (no GPU, no Kaggle):

EyePACS analysis results are reproducible from frozen per-image prediction files included in this archive. APTOS external-validation results are provided as frozen summary tables. To regenerate EyePACS results:

```bash
pip install -r requirements-lock.txt
python scripts/reproduce_results.py --predictions-dir results/predictions --dataset eyepacs
```

To validate the pipeline on synthetic data (no real data needed):

```bash
python examples/generate_toy_data.py
python scripts/reproduce_results.py --toy
```

See [`REPRODUCIBILITY_MAP.md`](REPRODUCIBILITY_MAP.md) for a complete mapping of every manuscript claim to its verifiable source file.

### What requires restricted-access data and GPU retraining:

Model fitting from raw retinal images requires Kaggle dataset access and approximately 48 GPU-hours on NVIDIA T4.

#### Step 1: Data Setup
Run `notebooks/notebook1_data_setup.ipynb` in Google Colab (GPU required). Downloads and preprocesses both datasets. Requires a Kaggle API token.

#### Step 2: Model Training
Run `notebooks/notebook2_model_training.ipynb`. Trains all 11 models on EyePACS.

#### Step 3: Evaluation and Robustness Analysis
Run `notebooks/notebook3_evaluation_robustness.ipynb`. Generates all results including threshold sweep, 2×2 decomposition, weight robustness, and bootstrap CIs. Output saved to `results/`.

## Models

11 deep learning models spanning 5 architectures, each in binary and 5-class formulations:

| Architecture | Binary | 5-Class | Notes |
|---|---|---|---|
| DenseNet121 | ✓ | ✓ | Best individual model (5-class, AUC = 0.911) |
| EfficientNet-B3 | ✓ | ✓ | Top ensemble partner |
| ResNet50 | ✓ | ✓ | |
| ViT-Base | ✓ | ✓ | |
| RETFound | ✓ | ✓ | Retinal foundation model (Zhou et al. 2023) |
| RETFound (adversarial) | ✓ | — | Adversarial decorrelation variant (α = 0.08) |

## Datasets

- **EyePACS**: 88,702 retinal fundus images, 5-level DR severity grading. [Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection)
- **APTOS 2019**: 3,662 retinal images, same grading scheme (external validation). [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection)

Both datasets require accepting competition rules on Kaggle before download.

## Severity-Weighted Cost Framework

| Error Type | Weight | Clinical Basis |
|---|---|---|
| Missed Grade 4 (PDR) | 500 | QALY loss 0.214/yr × 30 yrs; ~$320K societal cost. True ratio ~1,600:1, conservatively compressed to 500:1 |
| Missed Grade 3 (severe NPDR) | 200 | 50% 1-yr progression risk, time-discounted expected cost |
| Missed Grade 2 (moderate NPDR) | 15 | 23% progression over 4 years |
| Missed Grade 1 (mild NPDR) | 2 | Low immediate risk, annual monitoring |
| False positive | 1 | Cost of unnecessary referral (~$200–500) |

Sources: Moshfeghi et al. (2020), ETDRS Report No. 9, Javitt et al. (1996). Weights are unitless relative cost indices, not currency.

## Key Results

### Lead Ensemble: DenseNet121 (5-class) + EfficientNet-B3 (5-class), Evaluated on Both Datasets

| Metric | EyePACS | APTOS |
|---|---|---|
| Solo severity cost | 6,490 [4,435–8,812] | 1,570 |
| OR-gate severity cost | 3,983 [2,820–5,554] | 1,232 |
| Cost reduction | 38.6% | 21.5% |
| Concordant severe misses | 6 [2–11] | 2 |
| Error correlation (ρ) | 0.549 [0.497–0.595] | 0.77 |

### Best APTOS Ensemble: DenseNet121 (binary) + EfficientNet-B3 (binary)

| Metric | APTOS |
|---|---|
| OR-gate severity cost | 499 |
| Cost reduction | 68.2% |
| Concordant severe misses | 0 |

Note: The best-performing ensemble differs between datasets. The EyePACS-optimal pair (both 5-class models) achieves a 21.5% reduction on APTOS, while a different pair (both binary models) achieves the maximum 68.2% APTOS reduction. Both results are reported in the manuscript.

## Limitations

This repository does not include raw retinal images or trained model checkpoint files due to dataset access restrictions and package size constraints. Reproducing results from scratch requires Kaggle dataset access and approximately 48 GPU-hours of training compute.

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{deobhakta2026health_portfolio,
  title={Hidden safety value in multi-{AI} screening: a severity-weighted evaluation of diabetic retinopathy ensembles},
  author={Deobhakta, Avnish A. and Rosen, Richard B. and Tai, Tak Yee Tania and Tsai, James C. and Harris, Alon and Pasquale, Louis},
  year={2026},
  note={In preparation for NPJ Digital Medicine}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Authors

**Avnish A. Deobhakta, MD; Richard B. Rosen, MD, FACS; Tak Yee Tania Tai, MD; James C. Tsai, MD; Alon Harris, MS, PhD; Louis Pasquale, MD**

Department of Ophthalmology, New York Eye and Ear Infirmary of Mount Sinai, Icahn School of Medicine at Mount Sinai, New York, NY, USA

Windreich Department of Artificial Intelligence and Human Health, Icahn School of Medicine at Mount Sinai, New York, NY, USA (A.H., L.P.)
