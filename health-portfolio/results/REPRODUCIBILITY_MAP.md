# Reproducibility Map: Manuscript Claims to Data Sources

This document maps every quantitative claim in the manuscript to its verifiable source.

## Ground Truth Files

| File | Description |
|---|---|
| `results/comprehensive_evaluation.json` | All 55 pairwise ensemble results for EyePACS and APTOS |
| `results/npj_robustness_results.json` | Threshold sweep, bootstrap CIs |
| `configs/severity_weights.json` | Weight specification and derivation |
| `configs/model_configs.json` | All 11 model training configurations |
| `results/predictions/` | Per-image prediction files (frozen model outputs) |

## Abstract Claims

| Claim | Value | Source |
|---|---|---|
| "54 of 55 pairs showed zero or negative AUC lift" | 54/55 | Computed from `results/predictions/` (per-pair OR-gate AUC vs best solo AUC) |
| "reduced severity-weighted cost by 39%" | 38.9% | `comprehensive_evaluation.json` → eyepacs → best pair or_cost=3968 vs solo=6490 |
| "solo cost 6,490 [95% CI: 4,435 to 8,812]" | 6,490 | `npj_robustness_results.json` → threshold_sweep_eyepacs at t=0.5; bootstrap CI from same file |
| "OR-gate cost 3,968 [95% CI: 2,820 to 5,554]" | 3,968 | `comprehensive_evaluation.json` → eyepacs → or_gate_results[0] |
| "cost reduction 2,522 [95% CI: 976 to 4,360]" | 2,522 | 6490 - 3968; CI from bootstrap |
| "68% reduction (cost 499 vs. solo 1,570)" | 68.2% | `comprehensive_evaluation.json` → aptos → best pair (DN121_bin+EffNet_bin) |
| "96% of ensemble benefit from OR-gate rule" | 96% | rule_effect=-2410, metric_effect=-112, total=-2522; 2410/2522=95.6%≈96% |
| "rule effect = -2,410" | -2,410 | 2×2 factorial: OR-gate cost minus averaging cost |
| "metric effect = -112" | -112 | 2×2 factorial: severity-weighted minus symmetric |
| "3 severe misses from t=0.05 to t=0.20" | 3 | `npj_robustness_results.json` → threshold_sweep_eyepacs |
| "225 to 603 false positives" | 225–603 | Same file, fp column at t=0.20 and t=0.05 |
| "6 severe misses with 173 false positives" | 6/173 | `comprehensive_evaluation.json` → best pair both_miss=6; FP computed from predictions |
| "r = 0.725 [95% CI: 0.456 to 0.797]" | 0.725 | `comprehensive_evaluation.json` → eyepacs → rho_bothmiss_corr; CI from npj_robustness |
| "rho ≈ 0.70 threshold" | 0.70 | Exploratory, from Figure 2 visual inspection |

## Table 1: Individual Models

| Model | AUC | Severe Misses | G4 | G3 | Cost | Source |
|---|---|---|---|---|---|---|
| DenseNet121 5-class | 0.9109 | 11 | 4 | 7 | 6,490 | Fresh Colab inference (verified) |
| EfficientNet-B3 5-class | 0.8669 | 14 | 5 | 9 | 7,609 | Fresh Colab inference |
| DenseNet121 binary | 0.8974 | 17 | 9 | 8 | 8,583 | Fresh Colab inference |
| RETFound 5-class | 0.8818 | 20 | 5 | 15 | 8,844 | Fresh Colab inference |
| EfficientNet-B3 binary | 0.8591 | 18 | 7 | 11 | 9,459 | Fresh Colab inference |
| ResNet50 5-class | 0.8719 | 17 | 8 | 9 | 9,828 | Fresh Colab inference |
| RETFound adversarial | 0.8697 | 22 | 6 | 16 | 10,314 | Fresh Colab inference |
| RETFound binary | 0.8698 | 24 | 6 | 18 | 10,799 | Fresh Colab inference |
| ResNet50 binary | 0.8782 | 25 | 13 | 12 | 12,595 | Fresh Colab inference |
| ViT-Base 5-class | 0.8477 | 38 | 13 | 25 | 15,838 | Fresh Colab inference |
| ViT-Base binary | 0.8479 | 37 | 14 | 23 | 15,873 | Fresh Colab inference |

## Table 1: Top 10 Ensembles

Source: `comprehensive_evaluation.json` → eyepacs → or_gate_results, sorted by or_cost

## Table 2: Weight Sensitivity

Source: Recomputed from best pair G4/G3 counts × scenario weights + residual. See `results/csv/weight_sensitivity.csv`

## Table 3: Bootstrap CIs

Source: `npj_robustness_results.json` → bootstrap. See `results/csv/bootstrap_summary.csv`

## Figures

| Figure | Content | Generated from |
|---|---|---|
| Figure 1 | AUC lift vs OR-gate cost reduction (top 15) | `results/predictions/` + `comprehensive_evaluation.json` |
| Figure 2 | ρ vs concordant severe misses (55 pairs) | `comprehensive_evaluation.json` → eyepacs |
| Figure 3 | 11×11 concordant severe miss heatmap | `comprehensive_evaluation.json` → eyepacs + individual model predictions |
| Figure 4 | Threshold sweep with OR-gate diamond | `npj_robustness_results.json` + best pair FP count |
| Figure 5 | 2×2 factorial decomposition | Computed from averaging vs OR-gate × symmetric vs severity-weighted |
| Figure 6 | Waterfall cost decomposition | Per-grade miss counts from predictions |

## Results Section Claims

| Claim | Value | Source |
|---|---|---|
| "cost ranges from 6,490 to 15,873" | 6,490–15,873 | Individual model costs (Table 1) |
| "38.9% reduction" | 38.9% | (6490-3968)/6490 |
| "APTOS same pair: cost 1,232 (21.5%)" | 1,232 / 21.5% | `comprehensive_evaluation.json` → aptos, DN121_5c+EffNet_5c pair |
| "One additional pair with zero BM" | 1 | DN121_5c+EffNet_bin on APTOS (BM=0) |
| "coefficient = 35.5, R² = 0.51" | 35.5 / 0.51 | Linear regression on 55 pairs (rho vs concordant_misses) |
| "F(3,50) = 0.25, p = 0.86" | 0.25 / 0.86 | F-test for diversity dummies after controlling for rho |
| "below rho 0.70: BM 6–18" | 6–18 | Pairs with rho < 0.70 from comprehensive_evaluation.json |
| "above rho 0.70: BM 16–31" | 16–31 | Pairs with rho ≥ 0.70 |

## Methods Claims

| Claim | Source |
|---|---|
| Weights: G4=500, G3=200, G2=15, G1=2, FP=1 | `configs/severity_weights.json` |
| "500:1 ratio, compressed from ~1,600:1" | Derivation in severity_weights.json |
| α = 0.08 for adversarial training | `configs/model_configs.json` → retfound_adversarial |
| 80/10/10 train/val/test split | `notebooks/notebook1_data_setup.ipynb` |
| 13,532 training images | notebook1 output |
| Bootstrap n = 1,000 | `npj_robustness_results.json` |

## Numerical Precision Note

The frozen prediction CSVs export model probabilities at full float64 precision. One borderline case (image #1023, grade 2, OR-gate probability 0.497) classifies differently under float32 (in-memory Colab inference) versus float64 (CSV-based reproduction), affecting the lead ensemble cost by ±15 (one w₂ penalty). All shipped CSVs and manuscript numbers are derived from the frozen CSV predictions as processed by `reproduce_results.py`, ensuring exact reproducibility. This precision artifact has no impact on any qualitative conclusion, ranking, or severe-case analysis.
