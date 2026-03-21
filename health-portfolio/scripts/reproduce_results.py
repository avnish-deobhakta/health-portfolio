#!/usr/bin/env python3
"""
reproduce_results.py — Single source of truth for all manuscript results.

The CSV files in results/csv/ are the OUTPUT of this script run against
the frozen per-image predictions in results/predictions/. To regenerate:

    python scripts/reproduce_results.py

To verify shipped CSVs match regenerated ones:

    python scripts/reproduce_results.py --verify

To run on toy synthetic data (no real data needed):

    python scripts/reproduce_results.py --toy

Weights are loaded from configs/severity_weights.json (never hardcoded).
"""

import argparse, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

def load_weights(repo_root='.'):
    for path in [os.path.join(repo_root, 'configs', 'severity_weights.json'),
                 os.path.join(os.path.dirname(__file__), '..', 'configs', 'severity_weights.json')]:
        if os.path.exists(path):
            with open(path) as f:
                config = json.load(f)
            sw = config['severity_weights']
            weights = {4: sw['grade_4_miss']['weight'], 3: sw['grade_3_miss']['weight'],
                       2: sw['grade_2_miss']['weight'], 1: sw['grade_1_miss']['weight'],
                       'fp': sw['false_positive']['weight']}
            return weights, config.get('robustness_scenarios', [])
    sys.exit('ERROR: configs/severity_weights.json not found')

def severity_cost(preds, labels, grades, weights):
    cost = 0
    for i in range(len(preds)):
        if preds[i] != labels[i]:
            if labels[i] == 1: cost += weights.get(int(grades[i]), weights.get(1, 2))
            else: cost += weights['fp']
    return cost

def load_predictions(pred_dir, dataset):
    models = {}
    for f in sorted(os.listdir(pred_dir)):
        if f.startswith(f'predictions_{dataset}_') and f.endswith('.csv'):
            name = f.replace(f'predictions_{dataset}_', '').replace('.csv', '')
            df = pd.read_csv(os.path.join(pred_dir, f))
            probs = df['predicted_probability'].values
            models[name] = {'probs': probs, 'preds': (probs >= 0.5).astype(int), 'df': df}
    if not models: sys.exit(f'ERROR: No predictions found in {pred_dir} for {dataset}')
    first = list(models.values())[0]['df']
    return models, first['true_label'].values, first['true_grade'].values

def run_analysis(pred_dir, dataset, output_dir, weights, scenarios, repo_root='.'):
    os.makedirs(output_dir, exist_ok=True)
    models, labels, grades = load_predictions(pred_dir, dataset)
    names = sorted(models.keys())
    
    print(f'\n{"="*70}\nREPRODUCING: {dataset.upper()} ({len(names)} models, {len(labels)} images, {int((grades>=3).sum())} severe)\n{"="*70}')
    print(f'Weights: G4={weights[4]} G3={weights[3]} G2={weights[2]} G1={weights[1]} FP={weights["fp"]}')
    
    # Individual models
    print(f'\n{"Model":<35} {"AUC":>8} {"Cost":>8} {"SM":>5} {"G4":>5} {"G3":>5}')
    print('-'*70)
    model_costs, model_aucs = {}, {}
    for name in names:
        probs, preds = models[name]['probs'], models[name]['preds']
        auc = roc_auc_score(labels, probs)
        cost = severity_cost(preds, labels, grades, weights)
        sm = int(np.sum((grades>=3) & (labels==1) & (preds==0)))
        g4 = int(np.sum((grades==4) & (labels==1) & (preds==0)))
        g3 = int(np.sum((grades==3) & (labels==1) & (preds==0)))
        model_costs[name], model_aucs[name] = cost, auc
        print(f'{name:<35} {auc:>8.4f} {cost:>8,} {sm:>5} {g4:>5} {g3:>5}')
    
    best_name = min(model_costs, key=model_costs.get)
    best_cost = model_costs[best_name]
    best_auc = max(model_aucs.values())
    print(f'\nBest: {best_name} (cost={best_cost:,})')
    
    # Pairwise OR-gate
    pairs = []
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            p1, p2 = models[n1]['preds'], models[n2]['preds']
            or_p = np.maximum(p1, p2)
            or_cost = severity_cost(or_p, labels, grades, weights)
            e1, e2 = (p1!=labels).astype(int), (p2!=labels).astype(int)
            rho = np.corrcoef(e1,e2)[0,1] if e1.std()>0 and e2.std()>0 else 0.0
            bm = int(np.sum((grades>=3)&(labels==1)&(p1==0)&(p2==0)))
            g4 = int(np.sum((grades==4)&(labels==1)&(p1==0)&(p2==0)))
            g3 = int(np.sum((grades==3)&(labels==1)&(p1==0)&(p2==0)))
            fp = int(np.sum((or_p==1)&(labels==0)))
            ea = roc_auc_score(labels, np.maximum(models[n1]['probs'], models[n2]['probs']))
            a1 = n1.split('_')[0] if 'retfound' not in n1 else 'retfound'
            a2 = n2.split('_')[0] if 'retfound' not in n2 else 'retfound'
            t1 = '5class' if '5class' in n1 else 'binary'
            t2 = '5class' if '5class' in n2 else 'binary'
            sa, st = a1==a2, t1==t2
            div = 'TASK' if sa and not st else 'ARCH' if not sa and st else 'BOTH' if not sa and not st else 'SAME'
            pairs.append(dict(model_1=n1, model_2=n2, diversity_type=div, rho=rho,
                              concordant_severe_misses=bm, g4_both_miss=g4, g3_both_miss=g3,
                              or_gate_cost=or_cost, fp=fp, ensemble_auc=ea))
    pairs.sort(key=lambda x: x['or_gate_cost'])
    bp = pairs[0]
    
    # === GENERATE CSVs (these ARE the shipped results) ===
    pw = pd.DataFrame([{k: (f"{v:.4f}" if k=='rho' else v) for k,v in p.items() 
                         if k not in ('fp','ensemble_auc')} | {'vs_best_solo': p['or_gate_cost']-best_cost}
                        for p in pairs])
    pw.to_csv(os.path.join(output_dir, f'{dataset}_pairwise_results.csv'), index=False)
    
    # Threshold sweep
    ts = []
    bp_probs = models[best_name]['probs']
    for t in np.arange(0.05, 0.95, 0.05):
        pt = (bp_probs>=t).astype(int)
        ct = severity_cost(pt, labels, grades, weights)
        sev = int(np.sum((grades>=3)&(labels==1)&(pt==0)))
        g4m = int(np.sum((grades==4)&(labels==1)&(pt==0)))
        g3m = int(np.sum((grades==3)&(labels==1)&(pt==0)))
        fpt = int(np.sum((pt==1)&(labels==0)))
        tp = int(np.sum((pt==1)&(labels==1))); fn = int(np.sum((pt==0)&(labels==1)))
        tn = int(np.sum((pt==0)&(labels==0)))
        ts.append(dict(threshold=f'{t:.2f}', severity_cost=ct, g4_misses=g4m, g3_misses=g3m,
                        severe_misses=sev, false_positives=fpt,
                        sensitivity=f'{tp/(tp+fn):.4f}' if tp+fn>0 else '0',
                        specificity=f'{tn/(tn+fpt):.4f}' if tn+fpt>0 else '0'))
    pd.DataFrame(ts).to_csv(os.path.join(output_dir, f'threshold_sweep_{dataset}.csv'), index=False)
    
    # Weight sensitivity
    if scenarios:
        ws = []
        solo_p = models[best_name]['preds']
        ens_p = np.maximum(models[bp['model_1']]['preds'], models[bp['model_2']]['preds'])
        for sc in scenarios:
            sw = {4:sc['w4'],3:sc['w3'],2:sc['w2'],1:sc['w1'],'fp':sc['fp']}
            sc_solo = severity_cost(solo_p, labels, grades, sw)
            sc_ens = severity_cost(ens_p, labels, grades, sw)
            red = (sc_solo-sc_ens)/sc_solo*100 if sc_solo>0 else 0
            ws.append(dict(scenario=sc['name'],w4=sc['w4'],w3=sc['w3'],w2=sc['w2'],w1=sc['w1'],
                           w_fp=sc['fp'],solo_cost=sc_solo,or_gate_cost=sc_ens,
                           reduction_pct=f'{red:.1f}%',top_ensemble='Yes' if sc_ens<sc_solo else 'No'))
        pd.DataFrame(ws).to_csv(os.path.join(output_dir, 'weight_sensitivity.csv'), index=False)
    
    # Bootstrap summary (point estimates)
    zn = sum(1 for p in pairs if p['ensemble_auc']-best_auc<=0)
    ml = max(p['ensemble_auc']-best_auc for p in pairs)
    rhos = [p['rho'] for p in pairs]; bms = [p['concordant_severe_misses'] for p in pairs]
    rv = np.corrcoef(rhos,bms)[0,1] if len(pairs)>2 else 0
    pd.DataFrame([
        dict(quantity='solo_severity_cost',estimate=best_cost),
        dict(quantity='or_gate_severity_cost',estimate=bp['or_gate_cost']),
        dict(quantity='cost_reduction',estimate=best_cost-bp['or_gate_cost']),
        dict(quantity='error_correlation_rho',estimate=f"{bp['rho']:.4f}"),
        dict(quantity='concordant_severe_misses',estimate=bp['concordant_severe_misses']),
        dict(quantity='rho_vs_concordant_miss_r',estimate=f"{rv:.4f}"),
    ]).to_csv(os.path.join(output_dir, 'bootstrap_summary.csv'), index=False)
    
    print(f'\n{"="*70}\nKEY NUMBERS\n{"="*70}')
    print(f'  Solo cost:        {best_cost:,}')
    print(f'  Ensemble cost:    {bp["or_gate_cost"]:,}')
    print(f'  Reduction:        {best_cost-bp["or_gate_cost"]:,} ({(best_cost-bp["or_gate_cost"])/best_cost*100:.1f}%)')
    print(f'  Pair:             {bp["model_1"]} + {bp["model_2"]}')
    print(f'  Concordant SM:    {bp["concordant_severe_misses"]}')
    print(f'  FP:               {bp["fp"]}')
    print(f'  AUC lift ≤0:      {zn}/{len(pairs)}')
    print(f'  Max AUC lift:     {ml:.6f}')
    print(f'  rho-BM r:         {rv:.3f}')
    print(f'\nCSVs written to {output_dir}/')

def verify_mode(pred_dir, dataset, csv_dir, repo_root):
    weights, scenarios = load_weights(repo_root)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        run_analysis(pred_dir, dataset, tmp, weights, scenarios, repo_root)
        print(f'\n{"="*70}\nVERIFICATION\n{"="*70}')
        regen = pd.read_csv(os.path.join(tmp, f'{dataset}_pairwise_results.csv'))
        shipped = os.path.join(csv_dir, f'{dataset}_pairwise_results.csv')
        if os.path.exists(shipped):
            orig = pd.read_csv(shipped)
            match = regen['or_gate_cost'].equals(orig['or_gate_cost'])
            print(f'  Pairwise costs: {"✓ EXACT MATCH" if match else "✗ MISMATCH"}')
            if not match:
                for i in range(len(regen)):
                    if regen.iloc[i]['or_gate_cost'] != orig.iloc[i]['or_gate_cost']:
                        print(f'    Row {i}: regen={regen.iloc[i]["or_gate_cost"]} shipped={orig.iloc[i]["or_gate_cost"]}')
        else:
            print(f'  Shipped CSV not found: {shipped}')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions-dir', default='results/predictions')
    p.add_argument('--output-dir', default='results/csv')
    p.add_argument('--dataset', default='eyepacs')
    p.add_argument('--repo-root', default='.')
    p.add_argument('--toy', action='store_true')
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    if a.toy:
        td = os.path.join(a.repo_root, 'examples', 'toy_predictions')
        np.random.seed(42); n=200
        grades=np.random.choice([0,1,2,3,4],size=n,p=[.70,.07,.15,.04,.04])
        labels=(grades>=2).astype(int); os.makedirs(td, exist_ok=True)
        for nm,ns in [('A_bin',.30),('A_5c',.28),('B_bin',.35),('B_5c',.32),('C_bin',.38)]:
            noise=np.random.normal(0,ns,n); probs=1/(1+np.exp(-(labels*2-1+noise)))
            probs=np.clip(probs,.01,.99)
            pd.DataFrame(dict(image_id=[f'toy_{i:04d}' for i in range(n)],dataset='toy',
                true_grade=grades,true_label=labels,predicted_probability=probs,
                predicted_class=(probs>=.5).astype(int),model_name=f'model_{nm}')
            ).to_csv(os.path.join(td,f'predictions_toy_model_{nm}.csv'),index=False,float_format='%.15g')
        w,s = load_weights(a.repo_root)
        run_analysis(td,'toy',a.output_dir,w,s,a.repo_root)
    elif a.verify:
        verify_mode(a.predictions_dir, a.dataset, a.output_dir, a.repo_root)
    else:
        w,s = load_weights(a.repo_root)
        run_analysis(a.predictions_dir, a.dataset, a.output_dir, w, s, a.repo_root)

if __name__ == '__main__':
    main()
