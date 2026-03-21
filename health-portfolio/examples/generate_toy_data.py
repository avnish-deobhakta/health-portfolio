"""
Generate toy synthetic data to exercise the full analysis pipeline.

Usage:
    python examples/generate_toy_data.py
    python scripts/reproduce_results.py --toy

This creates a small synthetic dataset (200 cases, 5 models) with realistic
class distributions and model accuracies. The results are NOT clinically
meaningful — they exist solely to validate that the analysis code runs
correctly end-to-end.
"""

import numpy as np
import pandas as pd
import os

def generate():
    np.random.seed(42)
    n = 200
    
    # Realistic DR grade distribution
    grades = np.random.choice([0,1,2,3,4], size=n, p=[0.70, 0.07, 0.15, 0.04, 0.04])
    labels = (grades >= 2).astype(int)
    image_ids = [f'toy_{i:04d}' for i in range(n)]
    
    print(f'Generated {n} synthetic cases:')
    for g in range(5):
        print(f'  Grade {g}: {(grades == g).sum()}')
    print(f'  Severe (3+4): {(grades >= 3).sum()}')
    print(f'  Referable (2+): {labels.sum()}')
    
    out_dir = os.path.join(os.path.dirname(__file__), 'toy_predictions')
    os.makedirs(out_dir, exist_ok=True)
    
    models = [
        ('model_A_binary', 0.85, 0.30),
        ('model_A_5class', 0.87, 0.28),
        ('model_B_binary', 0.82, 0.35),
        ('model_B_5class', 0.84, 0.32),
        ('model_C_binary', 0.80, 0.38),
    ]
    
    for name, _, noise_std in models:
        noise = np.random.normal(0, noise_std, n)
        logits = labels * 2.0 - 1.0 + noise
        probs = 1 / (1 + np.exp(-logits))
        probs = np.clip(probs, 0.01, 0.99)
        preds = (probs >= 0.5).astype(int)
        
        df = pd.DataFrame({
            'image_id': image_ids,
            'dataset': 'toy',
            'true_grade': grades,
            'true_label': labels,
            'predicted_probability': np.round(probs, 6),
            'predicted_class': preds,
            'model_name': name,
        })
        
        path = os.path.join(out_dir, f'predictions_toy_{name}.csv')
        df.to_csv(path, index=False)
        
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, probs)
        print(f'  {name}: AUC={auc:.3f}, saved to {path}')
    
    print(f'\nToy data ready. Run: python scripts/reproduce_results.py --toy')

if __name__ == '__main__':
    generate()
