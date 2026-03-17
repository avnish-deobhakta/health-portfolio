"""
Evaluation functions for the Health Portfolio framework.

Implements three evaluation frameworks:
  1. Standard AUC with ensemble averaging
  2. OR-gate deployment with severity-weighted cost
  3. Concordant miss analysis
"""

import json
import numpy as np
from sklearn.metrics import roc_auc_score


def load_severity_weights(config_path="configs/severity_weights.json"):
    """Load severity weights from config file."""
    with open(config_path) as f:
        config = json.load(f)
    return {
        4: config["severity_weights"]["grade_4_miss"]["weight"],
        3: config["severity_weights"]["grade_3_miss"]["weight"],
        2: config["severity_weights"]["grade_2_miss"]["weight"],
        1: config["severity_weights"]["grade_1_miss"]["weight"],
        "fp": config["severity_weights"]["false_positive"]["weight"],
    }


def _coerce_binary_predictions(preds_or_probs, threshold=0.5):
    """Convert probabilities or binary predictions to {0,1} integers.

    Accepts either:
      - probabilities in [0, 1]
      - already-binary predictions in {0, 1}
    """
    arr = np.asarray(preds_or_probs)
    unique_vals = np.unique(arr)
    if np.all(np.isin(unique_vals, [0, 1])):
        return arr.astype(int)
    return (arr >= threshold).astype(int)


def compute_severity_cost(preds_or_probs, labels, grades, weights, threshold=0.5):
    """Compute total severity-weighted error cost.

    Args:
        preds_or_probs: Array of predicted probabilities or binary predictions.
        labels: Array of binary ground truth labels.
        grades: Array of severity grades (0-4).
        weights: Dict mapping grade -> cost weight.
        threshold: Decision threshold for binary prediction.

    Returns:
        total_cost: Scalar total severity-weighted cost.
        breakdown: Dict with counts per error type.
    """
    binary_preds = _coerce_binary_predictions(preds_or_probs, threshold=threshold)
    total_cost = 0
    breakdown = {
        "grade_4_miss": 0, "grade_3_miss": 0, "grade_2_miss": 0,
        "grade_1_miss": 0, "false_positive": 0
    }

    for i in range(len(binary_preds)):
        if binary_preds[i] != labels[i]:
            if labels[i] == 1:  # false negative
                grade = grades[i]
                cost = weights.get(grade, 2)
                total_cost += cost
                if grade == 4:
                    breakdown["grade_4_miss"] += 1
                elif grade == 3:
                    breakdown["grade_3_miss"] += 1
                elif grade == 2:
                    breakdown["grade_2_miss"] += 1
                else:
                    breakdown["grade_1_miss"] += 1
            else:  # false positive
                total_cost += weights["fp"]
                breakdown["false_positive"] += 1

    return total_cost, breakdown


def summarize_cost_breakdown(breakdown):
    """Map verbose cost breakdown keys to compact reporting keys."""
    return {
        "g4": breakdown["grade_4_miss"],
        "g3": breakdown["grade_3_miss"],
        "g2": breakdown["grade_2_miss"],
        "g1": breakdown["grade_1_miss"],
        "fp": breakdown["false_positive"],
    }


def compute_error_correlation(probs_a, probs_b, labels, threshold=0.5):
    """Compute Pearson correlation of binary error vectors."""
    errors_a = ((probs_a >= threshold).astype(int) != labels).astype(int)
    errors_b = ((probs_b >= threshold).astype(int) != labels).astype(int)
    if errors_a.std() > 0 and errors_b.std() > 0:
        return np.corrcoef(errors_a, errors_b)[0, 1]
    return np.nan


def or_gate_predictions(probs_a, probs_b, threshold=0.5):
    """OR-gate: flag if either model predicts referable."""
    preds_a = (probs_a >= threshold).astype(int)
    preds_b = (probs_b >= threshold).astype(int)
    return np.maximum(preds_a, preds_b)


def concordant_miss_analysis(probs_a, probs_b, labels, grades, threshold=0.5):
    """Identify severe cases that both models miss simultaneously.

    Args:
        probs_a, probs_b: Predicted probabilities from two models.
        labels: Binary ground truth.
        grades: Severity grades (0-4).
        threshold: Decision threshold.

    Returns:
        Dict with concordant miss counts by grade.
    """
    preds_a = (probs_a >= threshold).astype(int)
    preds_b = (probs_b >= threshold).astype(int)

    severe_mask = grades >= 3
    severe_indices = np.where(severe_mask)[0]

    both_miss = set()
    for idx in severe_indices:
        if labels[idx] == 1 and preds_a[idx] == 0 and preds_b[idx] == 0:
            both_miss.add(idx)

    g4_miss = sum(1 for idx in both_miss if grades[idx] == 4)
    g3_miss = sum(1 for idx in both_miss if grades[idx] == 3)

    return {
        "both_miss_total": len(both_miss),
        "g4_both_miss": g4_miss,
        "g3_both_miss": g3_miss,
        "both_miss_indices": sorted(both_miss),
    }


def classify_diversity_type(name_a, name_b):
    """Classify an ensemble pair by diversity type.

    Returns one of: TASK, ARCH, BOTH, SAME.
    """
    arch_a = name_a.split("_")[0] if "retfound" not in name_a else "retfound"
    arch_b = name_b.split("_")[0] if "retfound" not in name_b else "retfound"
    task_a = "5class" if "5class" in name_a else "binary"
    task_b = "5class" if "5class" in name_b else "binary"

    same_arch = (arch_a == arch_b)
    same_task = (task_a == task_b)

    if same_arch and not same_task:
        return "TASK"
    elif not same_arch and same_task:
        return "ARCH"
    elif not same_arch and not same_task:
        return "BOTH"
    else:
        return "SAME"


def full_pairwise_evaluation(all_preds, labels, grades, weights, threshold=0.5):
    """Run complete pairwise evaluation across all models.

    Args:
        all_preds: Dict of model_name -> predicted probabilities.
        labels: Binary ground truth array.
        grades: Severity grade array.
        weights: Severity weight dict.
        threshold: Decision threshold used for binary actions.

    Returns:
        List of dicts, one per pair, with all evaluation metrics.
    """
    model_names = list(all_preds.keys())
    results = []

    for i, name_a in enumerate(model_names):
        for j, name_b in enumerate(model_names):
            if j <= i:
                continue

            probs_a = all_preds[name_a]
            probs_b = all_preds[name_b]

            # AUC-based ensemble averaging
            ens_probs = (probs_a + probs_b) / 2
            ens_auc = roc_auc_score(labels, ens_probs)

            # Error correlation
            rho = compute_error_correlation(probs_a, probs_b, labels, threshold=threshold)

            # OR-gate cost (single clean path)
            or_preds = or_gate_predictions(probs_a, probs_b, threshold=threshold)
            or_cost, or_breakdown = compute_severity_cost(
                or_preds, labels, grades, weights, threshold=threshold
            )
            or_bd = summarize_cost_breakdown(or_breakdown)

            # Concordant misses
            cm = concordant_miss_analysis(
                probs_a, probs_b, labels, grades, threshold=threshold
            )

            # Diversity type
            div_type = classify_diversity_type(name_a, name_b)

            results.append({
                "model_a": name_a,
                "model_b": name_b,
                "diversity_type": div_type,
                "ensemble_auc": float(ens_auc),
                "rho": float(rho),
                "or_gate_cost": int(or_cost),
                "both_miss": cm["both_miss_total"],
                "g4_both_miss": cm["g4_both_miss"],
                "g3_both_miss": cm["g3_both_miss"],
                "or_gate_g4_miss": or_bd["g4"],
                "or_gate_g3_miss": or_bd["g3"],
                "or_gate_g2_miss": or_bd["g2"],
                "or_gate_g1_miss": or_bd["g1"],
                "or_gate_fp": or_bd["fp"],
            })

    return results
