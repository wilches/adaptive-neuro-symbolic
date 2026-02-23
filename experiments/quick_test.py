"""
Quick test to verify the pipeline works end-to-end.
Uses a few hardcoded problems — no API costs for symbolic tests.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.complexity_analyzer import ComplexityAnalyzer
from src.router import Router
from src.engines.symbolic_engine import SymbolicEngine


def test_analyzer():
    """Test the complexity analyzer with diverse problems."""
    analyzer = ComplexityAnalyzer()

    problems = [
        {
            'text': (
                "Natalia sold clips to 48 of her friends in April, "
                "and then she sold half as many clips in May. "
                "How many clips did Natalia sell altogether in April and May?"
            ),
            'expected_route': 'neural_cot',
            'label': 'GSM8K verbal',
        },
        {
            'text': r"Find all real solutions to $\frac{x^2 - 4}{x - 2} = 3$",
            'expected_route': 'hybrid',
            'label': 'MATH algebraic',
        },
        {
            'text': r"Simplify $\frac{24}{36}$",
            'expected_route': 'symbolic',
            'label': 'Simple fraction',
        },
        {
            'text': (
                "Prove that for all integers n >= 1, "
                r"$\sum_{k=1}^{n} k = \frac{n(n+1)}{2}$"
            ),
            'expected_route': 'hybrid',
            'label': 'Induction proof',
        },
        {
            'text': (
                "For every epsilon > 0, there exists a delta > 0 "
                "such that for all x, if |x - a| < delta then |f(x) - f(a)| < epsilon. "
                "Prove this for f(x) = 2x + 1."
            ),
            'expected_route': 'hybrid',
            'label': 'Epsilon-delta',
        },
        {
            'text': "What is 15 * 23 + 7?",
            'expected_route': 'symbolic',
            'label': 'Pure arithmetic',
        },
    ]

    print("=" * 80)
    print("COMPLEXITY ANALYZER TEST")
    print("=" * 80)

    for p in problems:
        profile = analyzer.analyze(p['text'])
        match = "✓" if profile.routing_suggestion == p['expected_route'] else "✗"

        print(f"\n{match} [{p['label']}]")
        print(f"   Problem: {p['text'][:80]}...")
        print(f"   d_L={profile.logical_depth:.2f}  "
              f"n_V={profile.variable_count}  "
              f"sigma_S={profile.symbolic_structure:.3f}  "
              f"omega_N={profile.nesting_complexity}  "
              f"kappa_Q={profile.quantifier_alternation}")
        print(f"   C(p)={profile.composite_score:.4f}  "
              f"-> Route: {profile.routing_suggestion}  "
              f"(expected: {p['expected_route']})")


def test_symbolic_engine():
    """Test the symbolic engine with solvable problems."""
    engine = SymbolicEngine()

    problems = [
        {
            'text': "$2x + 3 = 7$",
            'expected': "2",
            'label': "Linear equation",
        },
        {
            'text': "$x^2 - 4 = 0$",
            'expected': "[-2, 2]",
            'label': "Quadratic equation",
        },
        {
            'text': "Calculate $\\frac{24}{36}$",
            'expected': "2/3",
            'label': "Fraction simplification",
        },
        {
            'text': "What is 15 * 23 + 7?",
            'expected': "352",
            'label': "Arithmetic",
        },
        {
            'text': "$\\sqrt{144}$",
            'expected': "12",
            'label': "Square root",
        },
        {
            'text': (
                "Natalia sold clips to 48 of her friends in April, "
                "and then she sold half as many clips in May."
            ),
            'expected': "",
            'label': "Word problem (should FAIL — not symbolic)",
        },
    ]

    print("\n" + "=" * 80)
    print("SYMBOLIC ENGINE TEST")
    print("=" * 80)

    for p in problems:
        result = engine.solve(p['text'])
        status = "OK" if result.success else "FAIL"

        print(f"\n[{status}] {p['label']}")
        print(f"   Input:    {p['text'][:70]}")
        print(f"   Answer:   {result.answer}")
        print(f"   Expected: {p['expected']}")
        print(f"   Method:   {result.method_used}")
        print(f"   Steps:    {result.steps}")
        print(f"   Latency:  {result.latency_seconds:.4f}s")


def test_router():
    """Test the router with pre-computed profiles."""
    analyzer = ComplexityAnalyzer()
    router = Router()

    problems = [
        "What is 5 + 3?",
        "$\\frac{24}{36}$",
        (
            "Natalia sold clips to 48 of her friends in April, "
            "and then she sold half as many clips in May. "
            "How many clips did Natalia sell altogether in April and May?"
        ),
        r"Solve for x: $\frac{x^2 - 4}{x - 2} = \sqrt{x + 2} + 2$",
        (
            "Prove that for all n >= 1, there exists a prime p "
            "such that p divides n! + 1."
        ),
        (
            "For every epsilon > 0, there exists N such that "
            "for all n > N, |a_n - L| < epsilon. "
            "Prove that the sequence a_n = 1/n converges to 0."
        ),
    ]

    print("\n" + "=" * 80)
    print("ROUTER TEST")
    print("=" * 80)

    for problem in problems:
        profile = analyzer.analyze(problem)
        decision = router.route(profile)

        print(f"\nProblem: {problem[:70]}...")
        print(f"   C(p)={profile.composite_score:.4f}  "
              f"sigma={profile.symbolic_structure:.3f}")
        print(f"   -> {decision.path.value} "
              f"(confidence={decision.confidence:.3f})")
        print(f"   Reasoning: {decision.reasoning}")

    # Print distribution
    dist = router.get_routing_distribution()
    print(f"\nRouting distribution: {dist}")
    stats = router.get_confidence_stats()
    print(f"Confidence stats: {stats}")


def test_full_pipeline_local():
    """
    Test the full pipeline using ONLY the symbolic engine.
    No API calls — completely free.
    """
    from src.pipeline import AdaptivePipeline
    from src.config import ProviderConfig

    # Dummy config — won't actually call API for symbolic-only
    dummy_config = ProviderConfig(
        name="dummy",
        api_key="not-needed",
        base_url="http://localhost",
        model="dummy",
    )

    pipeline = AdaptivePipeline(cot_config=dummy_config)

    test_problems = [
        {'id': 'test_1', 'problem': '$2x + 6 = 10$', 'answer': '2'},
        {'id': 'test_2', 'problem': '$x^2 = 25$', 'answer': '5'},
        {'id': 'test_3', 'problem': 'Calculate $\\frac{100}{4}$', 'answer': '25'},
        {'id': 'test_4', 'problem': 'What is 12 * 8 + 4?', 'answer': '100'},
    ]

    print("\n" + "=" * 80)
    print("FULL PIPELINE TEST (Symbolic-only, no API cost)")
    print("=" * 80)

    results = pipeline.batch_solve(
        test_problems,
        prompt_type="math",
        force_route="symbolic",
    )

    for r in results:
        status = "CORRECT" if r.is_correct else "WRONG"
        print(f"\n[{status}] {r.problem_id}")
        print(f"   Problem:   {r.problem_text}")
        print(f"   Predicted: {r.predicted_answer}")
        print(f"   Expected:  {r.expected_answer}")
        print(f"   Tokens:    {r.tokens_total}")
        print(f"   Latency:   {r.latency_seconds:.3f}s")

    summary = pipeline.get_summary()
    print(f"\nSummary: {json.dumps(summary, indent=2, default=str)}")


import json

if __name__ == "__main__":
    print("\n>>> RUNNING ALL TESTS <<<\n")

    print("\n--- Test 1: Complexity Analyzer ---")
    test_analyzer()

    print("\n--- Test 2: Symbolic Engine ---")
    test_symbolic_engine()

    print("\n--- Test 3: Router ---")
    test_router()

    print("\n--- Test 4: Full Pipeline (local) ---")
    test_full_pipeline_local()

    print("\n>>> ALL TESTS COMPLETE <<<")
