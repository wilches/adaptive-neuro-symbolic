"""
Find optimal thresholds using existing experimental data.
NO API calls needed — uses results we already have.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.complexity_analyzer import ComplexityAnalyzer
from src.router import Router


def main():
    # ── Load existing results ────────────────────────
    results_dir = Path("results/20260224_210542")

    with open(results_dir / "results_symbolic.json") as f:
        sym_results = json.load(f)
    with open(results_dir / "results_cot_math.json") as f:
        cot_math = json.load(f)
    with open(results_dir / "results_cot_gsm8k.json") as f:
        cot_gsm8k = json.load(f)
    with open(results_dir / "results_hybrid.json") as f:
        hyb_results = json.load(f)

    cot_results = cot_math + cot_gsm8k

    # ── Re-analyze each problem to get complexity profiles ──
    analyzer = ComplexityAnalyzer()

    # Use symbolic results as source of problem texts (all 399)
    problems_text = [r['problem_text'] for r in sym_results]

    print(f"Analyzing {len(problems_text)} problems...")
    profiles = [analyzer.analyze(p) for p in problems_text]
    print("Done analyzing.")

    # ── Build path_results ───────────────────────────
    path_results = {
        'symbolic': [
            {'correct': r['is_correct'], 'tokens': r['tokens_total']}
            for r in sym_results
        ],
        'neural_cot': [
            {'correct': r['is_correct'], 'tokens': r['tokens_total']}
            for r in cot_results
        ],
        'hybrid': [
            {'correct': r['is_correct'], 'tokens': r['tokens_total']}
            for r in hyb_results
        ],
    }

    # Verify sizes match
    print(f"Symbolic results: {len(path_results['symbolic'])}")
    print(f"CoT results: {len(path_results['neural_cot'])}")
    print(f"Hybrid results: {len(path_results['hybrid'])}")
    print(f"Profiles: {len(profiles)}")

    # ── Optimize ─────────────────────────────────────
    print("\nRunning grid search (this takes ~1 minute)...")
    router = Router()
    best = router.optimize_thresholds(
        profiles, path_results,
        alpha=0.7, beta=0.3
    )

    print("\n" + "=" * 60)
    print("OPTIMAL THRESHOLDS FOUND:")
    print("=" * 60)
    print(json.dumps(best, indent=2))

    # ── Simulate with optimal thresholds ─────────────
    print("\n" + "=" * 60)
    print("SIMULATED RESULTS WITH OPTIMAL THRESHOLDS:")
    print("=" * 60)

    router_opt = Router(
        tau_low=best['tau_low'],
        tau_high=best['tau_high'],
        sigma_symbolic=best['sigma_symbolic'],
        sigma_neural=best['sigma_neural'],
    )

    correct = 0
    total_tokens = 0
    route_counts = {'symbolic': 0, 'neural_cot': 0, 'hybrid': 0}
    route_correct = {'symbolic': 0, 'neural_cot': 0, 'hybrid': 0}
    route_tokens = {'symbolic': 0, 'neural_cot': 0, 'hybrid': 0}

    for i, profile in enumerate(profiles):
        decision = router_opt.route(profile)
        path = decision.path.value
        route_counts[path] += 1

        if i < len(path_results.get(path, [])):
            r = path_results[path][i]
            if r['correct']:
                correct += 1
                route_correct[path] += 1
            total_tokens += r['tokens']
            route_tokens[path] += r['tokens']

    n = len(profiles)
    print(f"\nOverall Accuracy: {correct}/{n} = {correct/n:.4f} ({correct/n*100:.2f}%)")
    print(f"Total Tokens: {total_tokens}")
    print(f"Avg Tokens: {total_tokens/n:.1f}")

    print(f"\nRouting Distribution:")
    for path in route_counts:
        c = route_counts[path]
        acc = route_correct[path] / max(1, c)
        avg_t = route_tokens[path] / max(1, c)
        print(f"  {path:12s}: {c:3d} problems, "
              f"accuracy={acc:.4f}, avg_tokens={avg_t:.1f}")

    # ── Compare with current results ─────────────────
    print("\n" + "=" * 60)
    print("COMPARISON:")
    print("=" * 60)
    print(f"{'Method':<25} {'Accuracy':>10} {'Avg Tokens':>12}")
    print("-" * 50)
    print(f"{'CoT-only (baseline)':<25} {'63.91%':>10} {'1363.7':>12}")
    print(f"{'Adaptive (old τ)':<25} {'62.16%':>10} {'1262.4':>12}")
    print(f"{'Adaptive (optimized τ)':<25} "
          f"{correct/n*100:.2f}%".rjust(10) +
          f"{total_tokens/n:.1f}".rjust(13))

    # ── Save optimal config ──────────────────────────
    config = {
        'optimal_thresholds': best,
        'old_thresholds': {
            'tau_low': 0.30,
            'tau_high': 0.65,
            'sigma_symbolic': 0.60,
            'sigma_neural': 0.30,
        },
    }

    with open(results_dir / "optimal_thresholds.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved to {results_dir / 'optimal_thresholds.json'}")
    print("\nUse these in run_experiment.py:")
    print(f"  Router(tau_low={best['tau_low']}, "
          f"tau_high={best['tau_high']}, "
          f"sigma_symbolic={best['sigma_symbolic']}, "
          f"sigma_neural={best['sigma_neural']})")


if __name__ == "__main__":
    main()
