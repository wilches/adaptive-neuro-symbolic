"""
Threshold optimization with proper train/test split.
NO API calls — uses existing results.
"""

import json
import sys
import random
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.complexity_analyzer import ComplexityAnalyzer
from src.router import Router, RoutePath


def main():
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
    # For each problem i, we know the result of all 3 paths
    all_data = []
    for i in range(len(profiles)):
        all_data.append({
            'index': i,
            'profile': profiles[i],
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

    # ── Train/Test split (70/30, stratified by dataset) ──
    random.seed(42)

    # Separate MATH and GSM8K to stratify
    math_indices = list(range(len(cot_math)))           # 0..298
    gsm8k_indices = list(range(len(cot_math), len(cot_math) + len(cot_gsm8k)))  # 299..398

    random.shuffle(math_indices)
    random.shuffle(gsm8k_indices)

    math_split = int(0.7 * len(math_indices))
    gsm8k_split = int(0.7 * len(gsm8k_indices))

    train_indices = math_indices[:math_split] + gsm8k_indices[:gsm8k_split]
    test_indices = math_indices[math_split:] + gsm8k_indices[gsm8k_split:]

    random.shuffle(train_indices)
    random.shuffle(test_indices)

    train_data = [all_data[i] for i in train_indices]
    test_data = [all_data[i] for i in test_indices]

    print(f"\nTrain set: {len(train_data)} problems")
    print(f"Test set:  {len(test_data)} problems")

    # ── Grid search on TRAIN set only ────────────────
    print("\nRunning grid search on TRAIN set...")

    best_score = -float('inf')
    best_params = {}

    # Max tokens for normalization (from CoT baseline)
    max_tokens = max(d['neural_cot']['tokens'] for d in train_data) or 1

    for tau_low in np.arange(0.10, 0.50, 0.05):
        for tau_high in np.arange(0.40, 0.90, 0.05):
            if tau_low >= tau_high:
                continue
            for sigma_sym in np.arange(0.40, 0.80, 0.10):
                for sigma_neu in np.arange(0.15, 0.50, 0.10):
                    if sigma_neu >= sigma_sym:
                        continue

                    # Create router with these thresholds
                    router = Router(
                        tau_low=tau_low,
                        tau_high=tau_high,
                        sigma_symbolic=sigma_sym,
                        sigma_neural=sigma_neu,
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
                            'train_score': round(score, 4),
                            'train_accuracy': round(accuracy, 4),
                            'train_token_eff': round(token_eff, 4),
                        }

    print("\n" + "=" * 60)
    print("OPTIMAL THRESHOLDS (found on TRAIN set):")
    print("=" * 60)
    print(json.dumps(best_params, indent=2))

    # ── Evaluate on TEST set (never seen during optimization) ──
    print("\n" + "=" * 60)
    print("EVALUATION ON HELD-OUT TEST SET:")
    print("=" * 60)

    # Evaluate with OPTIMIZED thresholds
    router_opt = Router(
        tau_low=best_params['tau_low'],
        tau_high=best_params['tau_high'],
        sigma_symbolic=best_params['sigma_symbolic'],
        sigma_neural=best_params['sigma_neural'],
    )

    opt_results = _evaluate(router_opt, test_data, "Adaptive (optimized)")

    # Evaluate with ORIGINAL thresholds for comparison
    router_old = Router(
        tau_low=0.30,
        tau_high=0.65,
        sigma_symbolic=0.60,
        sigma_neural=0.30,
    )

    old_results = _evaluate(router_old, test_data, "Adaptive (original)")

    # Evaluate baselines on same test set
    print("\n--- Baselines on same test set ---")

    # CoT-only on test set
    cot_correct = sum(1 for d in test_data if d['neural_cot']['correct'])
    cot_tokens = sum(d['neural_cot']['tokens'] for d in test_data)
    n_test = len(test_data)
    print(f"CoT-only:          accuracy={cot_correct/n_test:.4f} "
          f"({cot_correct}/{n_test}), "
          f"avg_tokens={cot_tokens/n_test:.1f}")

    # Symbolic-only on test set
    sym_correct = sum(1 for d in test_data if d['symbolic']['correct'])
    sym_tokens = sum(d['symbolic']['tokens'] for d in test_data)
    print(f"Symbolic-only:     accuracy={sym_correct/n_test:.4f} "
          f"({sym_correct}/{n_test}), "
          f"avg_tokens={sym_tokens/n_test:.1f}")

    # Hybrid on test set
    hyb_correct = sum(1 for d in test_data if d['hybrid']['correct'])
    hyb_tokens = sum(d['hybrid']['tokens'] for d in test_data)
    print(f"Fixed Hybrid:      accuracy={hyb_correct/n_test:.4f} "
          f"({hyb_correct}/{n_test}), "
          f"avg_tokens={hyb_tokens/n_test:.1f}")

    # Random routing baseline (ablation)
    random.seed(123)
    rand_correct = 0
    rand_tokens = 0
    for d in test_data:
        path = random.choice(['symbolic', 'neural_cot', 'hybrid'])
        rand_correct += int(d[path]['correct'])
        rand_tokens += d[path]['tokens']
    print(f"Random routing:    accuracy={rand_correct/n_test:.4f} "
          f"({rand_correct}/{n_test}), "
          f"avg_tokens={rand_tokens/n_test:.1f}")

    # ── Final comparison table ───────────────────────
    print("\n" + "=" * 60)
    print("FINAL COMPARISON TABLE (on held-out test set)")
    print("=" * 60)
    print(f"{'Method':<25} {'Accuracy':>10} {'Avg Tokens':>12} {'Token Save':>12}")
    print("-" * 62)
    print(f"{'Symbolic-only':<25} {sym_correct/n_test*100:>9.2f}% {sym_tokens/n_test:>11.1f} {'---':>12}")
    print(f"{'Random routing':<25} {rand_correct/n_test*100:>9.2f}% {rand_tokens/n_test:>11.1f} {'---':>12}")
    print(f"{'CoT-only':<25} {cot_correct/n_test*100:>9.2f}% {cot_tokens/n_test:>11.1f} {'baseline':>12}")
    print(f"{'Fixed Hybrid':<25} {hyb_correct/n_test*100:>9.2f}% {hyb_tokens/n_test:>11.1f} "
          f"{(1-hyb_tokens/max(1,cot_tokens))*100:>+10.1f}%")
    print(f"{'Adaptive (original)':<25} {old_results['accuracy']*100:>9.2f}% "
          f"{old_results['avg_tokens']:>11.1f} "
          f"{(1-old_results['total_tokens']/max(1,cot_tokens))*100:>+10.1f}%")
    print(f"{'Adaptive (optimized)':<25} {opt_results['accuracy']*100:>9.2f}% "
          f"{opt_results['avg_tokens']:>11.1f} "
          f"{(1-opt_results['total_tokens']/max(1,cot_tokens))*100:>+10.1f}%")
    print("=" * 62)

    # ── Save everything ──────────────────────────────
    output = {
        'optimal_thresholds': best_params,
        'old_thresholds': {
            'tau_low': 0.30, 'tau_high': 0.65,
            'sigma_symbolic': 0.60, 'sigma_neural': 0.30,
        },
        'train_size': len(train_data),
        'test_size': len(test_data),
        'test_results': {
            'symbolic_only': {'accuracy': sym_correct/n_test, 'avg_tokens': sym_tokens/n_test},
            'cot_only': {'accuracy': cot_correct/n_test, 'avg_tokens': cot_tokens/n_test},
            'fixed_hybrid': {'accuracy': hyb_correct/n_test, 'avg_tokens': hyb_tokens/n_test},
            'random_routing': {'accuracy': rand_correct/n_test, 'avg_tokens': rand_tokens/n_test},
            'adaptive_original': old_results,
            'adaptive_optimized': opt_results,
        },
    }

    save_path = results_dir / "optimization_results_v2.json"
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {save_path}")


def _evaluate(router, test_data, label):
    """Evaluate a router on test data."""
    correct = 0
    total_tokens = 0
    route_counts = {'symbolic': 0, 'neural_cot': 0, 'hybrid': 0}
    route_correct = {'symbolic': 0, 'neural_cot': 0, 'hybrid': 0}

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

    print(f"\n{label}:")
    print(f"  Accuracy: {correct}/{n} = {correct/n:.4f} ({correct/n*100:.2f}%)")
    print(f"  Avg Tokens: {total_tokens/n:.1f}")
    print(f"  Routing: {route_counts}")
    for path in route_counts:
        c = route_counts[path]
        if c > 0:
            acc = route_correct[path] / c
            print(f"    {path}: {route_correct[path]}/{c} = {acc:.4f}")

    return {
        'accuracy': correct / n,
        'total_tokens': total_tokens,
        'avg_tokens': round(total_tokens / n, 1),
        'route_counts': route_counts,
        'route_correct': route_correct,
    }


if __name__ == "__main__":
    main()
