"""
Stratified K-Fold Cross-Validation for threshold optimization.
More robust than single train/test split.
NO API calls — uses existing results.
"""

import json
import sys
import random
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.complexity_analyzer import ComplexityAnalyzer
from src.router import Router


def main():
    K = 5  # Number of folds
    results_dir = Path("results/20260224_210542")

    # ── Load all existing results ────────────────────
    with open(results_dir / "results_symbolic.json") as f:
        sym_results = json.load(f)
    with open(results_dir / "results_cot_math.json") as f:
        cot_math = json.load(f)
    with open(results_dir / "results_cot_gsm8k.json") as f:
        cot_gsm8k = json.load(f)
    with open(results_dir / "results_hybrid.json") as f:
        hyb_results = json.load(f)

    cot_results = cot_math + cot_gsm8k

    # ── Compute complexity profiles ──────────────────
    analyzer = ComplexityAnalyzer()
    problems_text = [r['problem_text'] for r in sym_results]
    print(f"Analyzing {len(problems_text)} problems...")
    profiles = [analyzer.analyze(p) for p in problems_text]

    # ── Build per-problem data ───────────────────────
    all_data = []
    for i in range(len(profiles)):
        all_data.append({
            'index': i,
            'profile': profiles[i],
            'dataset': 'math' if i < len(cot_math) else 'gsm8k',
            'symbolic': {
                'correct': sym_results[i]['is_correct'],
                'tokens': sym_results[i]['tokens_total'],
            },
            'neural_cot': {
                'correct': cot_results[i]['is_correct'],
                'tokens': cot_results[i]['tokens_total'],
            },
            'hybrid': {
                'correct': hyb_results[i]['is_correct'],
                'tokens': hyb_results[i]['tokens_total'],
            },
        })

    # ── Create STRATIFIED K folds ────────────────────
    # Stratify by dataset (math vs gsm8k) to ensure
    # each fold has proportional representation
    random.seed(42)

    math_indices = [i for i, d in enumerate(all_data) if d['dataset'] == 'math']
    gsm8k_indices = [i for i, d in enumerate(all_data) if d['dataset'] == 'gsm8k']

    random.shuffle(math_indices)
    random.shuffle(gsm8k_indices)

    # Distribute into K folds
    folds = [[] for _ in range(K)]

    for i, idx in enumerate(math_indices):
        folds[i % K].append(idx)
    for i, idx in enumerate(gsm8k_indices):
        folds[i % K].append(idx)

    for fold in folds:
        random.shuffle(fold)

    print(f"\n{K}-Fold split:")
    for i, fold in enumerate(folds):
        n_math = sum(1 for idx in fold if all_data[idx]['dataset'] == 'math')
        n_gsm8k = sum(1 for idx in fold if all_data[idx]['dataset'] == 'gsm8k')
        print(f"  Fold {i+1}: {len(fold)} problems "
              f"(MATH: {n_math}, GSM8K: {n_gsm8k})")

    # ── Run K-Fold ───────────────────────────────────
    fold_results = {
        'symbolic_only': [],
        'cot_only': [],
        'fixed_hybrid': [],
        'random_routing': [],
        'adaptive_original': [],
        'adaptive_optimized': [],
    }

    fold_token_results = {
        'symbolic_only': [],
        'cot_only': [],
        'fixed_hybrid': [],
        'random_routing': [],
        'adaptive_original': [],
        'adaptive_optimized': [],
    }

    # Store per-fold routing details for adaptive optimized
    fold_routing_details = []

    for fold_idx in range(K):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{K}")
        print(f"{'='*60}")

        # Split
        test_indices = set(folds[fold_idx])
        train_indices = []
        for j in range(K):
            if j != fold_idx:
                train_indices.extend(folds[j])

        train_data = [all_data[i] for i in train_indices]
        test_data = [all_data[i] for i in test_indices]

        print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

        # ── Optimize thresholds on TRAIN ─────────────
        best_params = _grid_search(train_data)
        print(f"  Optimal: τ_low={best_params['tau_low']}, "
              f"τ_high={best_params['tau_high']}, "
              f"σ_sym={best_params['sigma_symbolic']}, "
              f"σ_neu={best_params['sigma_neural']}")

        # ── Evaluate ALL methods on TEST ─────────────
        n_test = len(test_data)

        # Baselines (no routing needed)
        sym_acc = sum(1 for d in test_data if d['symbolic']['correct']) / n_test
        sym_tok = sum(d['symbolic']['tokens'] for d in test_data) / n_test
        fold_results['symbolic_only'].append(sym_acc)
        fold_token_results['symbolic_only'].append(sym_tok)

        cot_acc = sum(1 for d in test_data if d['neural_cot']['correct']) / n_test
        cot_tok = sum(d['neural_cot']['tokens'] for d in test_data) / n_test
        fold_results['cot_only'].append(cot_acc)
        fold_token_results['cot_only'].append(cot_tok)

        hyb_acc = sum(1 for d in test_data if d['hybrid']['correct']) / n_test
        hyb_tok = sum(d['hybrid']['tokens'] for d in test_data) / n_test
        fold_results['fixed_hybrid'].append(hyb_acc)
        fold_token_results['fixed_hybrid'].append(hyb_tok)

        # Random routing
        random.seed(123 + fold_idx)
        rand_correct = 0
        rand_tokens = 0
        for d in test_data:
            path = random.choice(['symbolic', 'neural_cot', 'hybrid'])
            rand_correct += int(d[path]['correct'])
            rand_tokens += d[path]['tokens']
        fold_results['random_routing'].append(rand_correct / n_test)
        fold_token_results['random_routing'].append(rand_tokens / n_test)

        # Adaptive (original thresholds)
        router_old = Router(tau_low=0.30, tau_high=0.65,
                           sigma_symbolic=0.60, sigma_neural=0.30)
        old_res = _evaluate_router(router_old, test_data)
        fold_results['adaptive_original'].append(old_res['accuracy'])
        fold_token_results['adaptive_original'].append(old_res['avg_tokens'])

        # Adaptive (optimized thresholds from THIS fold's train set)
        router_opt = Router(
            tau_low=best_params['tau_low'],
            tau_high=best_params['tau_high'],
            sigma_symbolic=best_params['sigma_symbolic'],
            sigma_neural=best_params['sigma_neural'],
        )
        opt_res = _evaluate_router(router_opt, test_data)
        fold_results['adaptive_optimized'].append(opt_res['accuracy'])
        fold_token_results['adaptive_optimized'].append(opt_res['avg_tokens'])
        fold_routing_details.append(opt_res)

        # Print fold summary
        print(f"  Results on test fold:")
        print(f"    Symbolic:           {sym_acc:.4f}")
        print(f"    Random:             {rand_correct/n_test:.4f}")
        print(f"    CoT-only:           {cot_acc:.4f}")
        print(f"    Fixed Hybrid:       {hyb_acc:.4f}")
        print(f"    Adaptive (orig):    {old_res['accuracy']:.4f}")
        print(f"    Adaptive (opt):     {opt_res['accuracy']:.4f}")

    # ── FINAL AGGREGATED RESULTS ─────────────────────
    print("\n" + "=" * 70)
    print(f"FINAL RESULTS: {K}-Fold Stratified Cross-Validation")
    print("=" * 70)
    print(f"{'Method':<25} {'Accuracy':>15} {'Avg Tokens':>18} {'Token Save':>12}")
    print("-" * 70)

    cot_mean_tokens = np.mean(fold_token_results['cot_only'])

    for method in fold_results:
        accs = fold_results[method]
        toks = fold_token_results[method]
        acc_mean = np.mean(accs) * 100
        acc_std = np.std(accs) * 100
        tok_mean = np.mean(toks)
        tok_std = np.std(toks)

        token_save = ((cot_mean_tokens - tok_mean) / cot_mean_tokens) * 100

        name = method.replace('_', ' ').title()

        if method in ['symbolic_only', 'random_routing']:
            save_str = "---"
        else:
            save_str = f"{token_save:+.1f}%"

        print(f"{name:<25} {acc_mean:>6.2f}% ± {acc_std:>4.2f}% "
              f"{tok_mean:>8.1f} ± {tok_std:>5.1f} {save_str:>12}")

    print("=" * 70)

    # ── Routing distribution across all folds ────────
    print("\n--- Adaptive (optimized) Routing Breakdown ---")
    total_routes = defaultdict(int)
    total_correct = defaultdict(int)

    for detail in fold_routing_details:
        for path in detail['route_counts']:
            total_routes[path] += detail['route_counts'][path]
            total_correct[path] += detail['route_correct'][path]

    for path in total_routes:
        n = total_routes[path]
        c = total_correct[path]
        print(f"  {path:<12}: {n:>3} problems, "
              f"accuracy={c/max(1,n):.4f} ({c}/{n})")

    # ── Statistical significance ─────────────────────
    print("\n--- Statistical Tests ---")

    # Paired comparison: Adaptive optimized vs CoT-only
    adaptive_accs = np.array(fold_results['adaptive_optimized'])
    cot_accs = np.array(fold_results['cot_only'])
    diff = adaptive_accs - cot_accs

    print(f"Adaptive - CoT (per fold): {[f'{d:.4f}' for d in diff]}")
    print(f"Mean difference: {np.mean(diff)*100:.2f}pp ± {np.std(diff)*100:.2f}pp")

    # Adaptive vs Random
    random_accs = np.array(fold_results['random_routing'])
    diff_rand = adaptive_accs - random_accs
    print(f"Adaptive - Random (per fold): {[f'{d:.4f}' for d in diff_rand]}")
    print(f"Mean difference: {np.mean(diff_rand)*100:.2f}pp ± {np.std(diff_rand)*100:.2f}pp")

    # Token comparison
    adaptive_toks = np.array(fold_token_results['adaptive_optimized'])
    cot_toks = np.array(fold_token_results['cot_only'])
    tok_savings = (cot_toks - adaptive_toks) / cot_toks * 100
    print(f"\nToken savings per fold: {[f'{s:.1f}%' for s in tok_savings]}")
    print(f"Mean token saving: {np.mean(tok_savings):.1f}% ± {np.std(tok_savings):.1f}%")

    # ── Save everything ──────────────────────────────
    output = {
        'k_folds': K,
        'total_problems': len(all_data),
        'fold_sizes': [len(f) for f in folds],
        'results_per_fold': {
            method: {
                'accuracy_per_fold': [round(a, 4) for a in accs],
                'tokens_per_fold': [round(t, 1) for t in fold_token_results[method]],
            }
            for method, accs in fold_results.items()
        },
        'aggregated': {
            method: {
                'accuracy_mean': round(float(np.mean(accs)), 4),
                'accuracy_std': round(float(np.std(accs)), 4),
                'avg_tokens_mean': round(float(np.mean(fold_token_results[method])), 1),
                'avg_tokens_std': round(float(np.std(fold_token_results[method])), 1),
            }
            for method, accs in fold_results.items()
        },
        'routing_totals': {
            path: {
                'count': total_routes[path],
                'correct': total_correct[path],
                'accuracy': round(total_correct[path] / max(1, total_routes[path]), 4),
            }
            for path in total_routes
        },
    }

    save_path = results_dir / "kfold_results.json"
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {save_path}")


def _grid_search(train_data):
    """Find optimal thresholds on training data."""
    best_score = -float('inf')
    best_params = {}

    max_tokens = max(d['neural_cot']['tokens'] for d in train_data) or 1

    for tau_low in np.arange(0.10, 0.50, 0.05):
        for tau_high in np.arange(0.40, 0.90, 0.05):
            if tau_low >= tau_high:
                continue
            for sigma_sym in np.arange(0.40, 0.80, 0.10):
                for sigma_neu in np.arange(0.15, 0.50, 0.10):
                    if sigma_neu >= sigma_sym:
                        continue

                    router = Router(
                        tau_low=tau_low, tau_high=tau_high,
                        sigma_symbolic=sigma_sym, sigma_neural=sigma_neu,
                    )

                    total_correct = 0
                    total_tokens = 0

                    for d in train_data:
                        decision = router.route(d['profile'])
                        path = decision.path.value
                        result = d[path]
                        total_correct += int(result['correct'])
                        total_tokens += result['tokens']

                    router.reset_history()

                    n = len(train_data)
                    accuracy = total_correct / n
                    token_eff = 1 - (total_tokens / (n * max_tokens))
                    score = 0.7 * accuracy + 0.3 * token_eff

                    if score > best_score:
                        best_score = score
                        best_params = {
                            'tau_low': round(float(tau_low), 2),
                            'tau_high': round(float(tau_high), 2),
                            'sigma_symbolic': round(float(sigma_sym), 2),
                            'sigma_neural': round(float(sigma_neu), 2),
                        }

    return best_params


def _evaluate_router(router, test_data):
    """Evaluate a router configuration on test data."""
    correct = 0
    total_tokens = 0
    route_counts = defaultdict(int)
    route_correct = defaultdict(int)

    for d in test_data:
        decision = router.route(d['profile'])
        path = decision.path.value
        result = d[path]

        route_counts[path] += 1
        if result['correct']:
            correct += 1
            route_correct[path] += 1
        total_tokens += result['tokens']

    router.reset_history()
    n = len(test_data)

    return {
        'accuracy': correct / n,
        'avg_tokens': total_tokens / n,
        'total_tokens': total_tokens,
        'route_counts': dict(route_counts),
        'route_correct': dict(route_correct),
    }


if __name__ == "__main__":
    main()
