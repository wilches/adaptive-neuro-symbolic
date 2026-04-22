"""Quick test to verify datasets load correctly."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

from experiments.load_data import load_math_dataset, load_gsm8k_dataset


def test_datasets():
    print("=" * 60)
    print("DATASET LOADING TEST")
    print("=" * 60)

    # ── Test MATH ────────────────────────────────
    print("\n--- Loading MATH dataset ---")
    math_data = load_math_dataset(sample_per_level=3, levels=[1, 2, 3, 4, 5])
    print(f"Loaded: {len(math_data)} problems")

    # Show distribution
    from collections import Counter
    levels = Counter(p['level'] for p in math_data)
    types = Counter(p['type'] for p in math_data)

    print(f"By level: {dict(sorted(levels.items()))}")
    print(f"By type:  {dict(types)}")

    # Show samples
    for level in [1, 3, 5]:
        sample = next((p for p in math_data if p['level'] == level), None)
        if sample:
            print(f"\n  Level {level} example:")
            print(f"    Problem: {sample['problem'][:80]}...")
            print(f"    Answer:  {sample['answer'][:50]}")
            print(f"    Type:    {sample['type']}")

    # ── Test GSM8K ───────────────────────────────
    print("\n--- Loading GSM8K dataset ---")
    gsm8k_data = load_gsm8k_dataset(sample_size=5)
    print(f"Loaded: {len(gsm8k_data)} problems")

    for p in gsm8k_data[:2]:
        print(f"\n  GSM8K example:")
        print(f"    Problem: {p['problem'][:80]}...")
        print(f"    Answer:  {p['answer']}")

    # ── Verify answers exist ─────────────────────
    print("\n--- Quality Check ---")
    math_with_answers = sum(1 for p in math_data if p['answer'])
    gsm8k_with_answers = sum(1 for p in gsm8k_data if p['answer'])
    print(f"MATH problems with answers:  {math_with_answers}/{len(math_data)}")
    print(f"GSM8K problems with answers: {gsm8k_with_answers}/{len(gsm8k_data)}")

    if math_with_answers == len(math_data) and gsm8k_with_answers == len(gsm8k_data):
        print("\n✅ All datasets loaded correctly!")
        print(f"\nReady for full experiment. Recommended command:")
        print(f"  python experiments/run_experiment.py \\")
        print(f"    --provider siliconflow \\")
        print(f"    --math-per-level 60 \\")
        print(f"    --gsm8k-size 100")
    else:
        print("\n⚠️ Some problems missing answers. Check dataset loader.")


def analyze_sample_for_experiment():
    """
    Analyze the exact sample we'll use in the experiment.
    Shows what the Complexity Analyzer thinks about each problem.
    """
    from src.complexity_analyzer import ComplexityAnalyzer
    from collections import Counter

    print("\n" + "=" * 60)
    print("PRE-EXPERIMENT ANALYSIS (60 per level + 100 GSM8K)")
    print("=" * 60)

    math_data = load_math_dataset(sample_per_level=60, levels=[1, 2, 3, 4, 5])
    gsm8k_data = load_gsm8k_dataset(sample_size=100)
    all_data = math_data + gsm8k_data

    analyzer = ComplexityAnalyzer()

    # Analyze all problems
    route_counts = Counter()
    route_by_level = {}
    route_by_dataset = {'MATH': Counter(), 'GSM8K': Counter()}
    scores = []

    for p in all_data:
        profile = analyzer.analyze(p['problem'])
        route = profile.routing_suggestion
        route_counts[route] += 1
        scores.append(profile.composite_score)

        level = p.get('level', 0)
        if level not in route_by_level:
            route_by_level[level] = Counter()
        route_by_level[level][route] += 1

        route_by_dataset[p['dataset']][route] += 1

    print(f"\nTotal problems: {len(all_data)}")
    print(f"  MATH:  {len(math_data)}")
    print(f"  GSM8K: {len(gsm8k_data)}")

    # Answers quality
    with_answers = sum(1 for p in all_data if p['answer'])
    print(f"  With answers: {with_answers}/{len(all_data)}")

    # Route distribution
    print(f"\nPredicted routing distribution:")
    for route, count in sorted(route_counts.items()):
        pct = count / len(all_data) * 100
        print(f"  {route:12s}: {count:4d} ({pct:.1f}%)")

    # Route by level
    print(f"\nRouting by difficulty level:")
    print(f"  {'Level':<8} {'Symbolic':>10} {'Hybrid':>10} {'Neural':>10} {'Total':>8}")
    print(f"  {'-'*48}")
    for level in sorted(route_by_level.keys()):
        counts = route_by_level[level]
        total = sum(counts.values())
        sym = counts.get('symbolic', 0)
        hyb = counts.get('hybrid', 0)
        neu = counts.get('neural_cot', 0)
        print(f"  {level:<8} {sym:>10} {hyb:>10} {neu:>10} {total:>8}")

    # Route by dataset
    print(f"\nRouting by dataset:")
    for ds, counts in route_by_dataset.items():
        print(f"  {ds}: {dict(counts)}")

    # Complexity score distribution
    import numpy as np
    scores = np.array(scores)
    print(f"\nComplexity score distribution:")
    print(f"  Mean:   {scores.mean():.4f}")
    print(f"  Std:    {scores.std():.4f}")
    print(f"  Min:    {scores.min():.4f}")
    print(f"  Max:    {scores.max():.4f}")
    print(f"  Median: {np.median(scores):.4f}")

    # Estimate token savings
    symbolic_count = route_counts.get('symbolic', 0)
    estimated_savings_pct = symbolic_count / len(all_data) * 100
    print(f"\nEstimated token savings from symbolic routing: {estimated_savings_pct:.1f}%")
    print(f"  ({symbolic_count} problems × ~1000 tokens/problem = ~{symbolic_count * 1000:,} tokens saved)")

    # Cost estimate
    non_symbolic = len(all_data) - symbolic_count
    est_cost = non_symbolic * 0.0003 * 4  # 4 methods, ~\$0.0003 per problem per method
    print(f"\nEstimated experiment cost: ~${est_cost:.2f}")

    print(f"\n✅ Sample looks good for experiment!" if with_answers == len(all_data) else "\n⚠️ Some problems missing answers!")


# Add to main
if __name__ == "__main__":
    test_datasets()
    analyze_sample_for_experiment()
