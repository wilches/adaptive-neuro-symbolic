import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_siliconflow_config
from src.engines.cot_engine import CoTEngine


def test_siliconflow(api_key: str):
    """Test basic API connectivity with SiliconFlow."""

    print("=" * 60)
    print("TESTING SILICONFLOW API CONNECTION")
    print("=" * 60)

    config = get_siliconflow_config(api_key)
    print(f"Provider: {config.name}")
    print(f"Model:    {config.model}")
    print(f"Base URL: {config.base_url}")

    engine = CoTEngine(config)

    # ── Test 1: Simple arithmetic (GSM8K style) ─────
    print("\n--- Test 1: Simple word problem ---")
    result = engine.solve(
        "Maria has 5 apples. She buys 3 more. How many apples does she have?",
        prompt_type="gsm8k"
    )
    print(f"Success:  {result.success}")
    print(f"Answer:   {result.answer}")
    print(f"Tokens:   {result.tokens_total}")
    print(f"Latency:  {result.latency_seconds}s")
    print(f"Cost:     ${engine.estimate_cost(result.tokens_input, result.tokens_output):.6f}")
    if result.error_message:
        print(f"Error:    {result.error_message}")
    print(f"Response preview: {result.full_response[:200]}...")

    if not result.success:
        print("\n⚠️  API call failed. Check your API key and model availability.")
        print("Common fixes:")
        print("  1. Verify API key is correct")
        print("  2. Try a different model:")
        print("     - deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        print("     - deepseek-ai/DeepSeek-V2.5")
        print("     - Qwen/Qwen2.5-7B-Instruct")
        return False

    # ── Test 2: MATH style problem ──────────────────
    print("\n--- Test 2: MATH style problem ---")
    result2 = engine.solve(
        r"Find the value of $x$ if $2x + 5 = 13$.",
        prompt_type="math"
    )
    print(f"Success:  {result2.success}")
    print(f"Answer:   {result2.answer}")
    print(f"Tokens:   {result2.tokens_total}")
    print(f"Latency:  {result2.latency_seconds}s")

    # ── Test 3: Full pipeline with adaptive routing ──
    print("\n--- Test 3: Full adaptive pipeline (3 problems) ---")
    from src.pipeline import AdaptivePipeline

    pipeline = AdaptivePipeline(cot_config=config)

    test_problems = [
        {
            'id': 'api_test_1',
            'problem': 'Simplify $\\frac{48}{64}$',
            'answer': '3/4',
        },
        {
            'id': 'api_test_2',
            'problem': (
                'Tom has 15 marbles. He gives 1/3 of them to his friend. '
                'How many marbles does Tom have left?'
            ),
            'answer': '10',
        },
        {
            'id': 'api_test_3',
            'problem': r'Solve for $x$: $3x - 7 = 2x + 5$',
            'answer': '12',
        },
    ]

    results = pipeline.batch_solve(
        test_problems,
        prompt_type="math",
    )

    print("\nResults:")
    for r in results:
        status = "✓" if r.is_correct else "✗"
        print(f"  {status} [{r.problem_id}] route={r.route_taken:12s} "
              f"answer={r.predicted_answer:10s} "
              f"expected={r.expected_answer:10s} "
              f"tokens={r.tokens_total}")

    summary = pipeline.get_summary()
    print(f"\nAccuracy:    {summary['accuracy']:.2%}")
    print(f"Total tokens: {summary['total_tokens']}")
    print(f"Total cost:   ${summary['total_cost_usd']:.6f}")
    print(f"Routes used:  {summary['route_distribution']}")

    print("\n" + "=" * 60)
    print("API CONNECTION TEST COMPLETE")
    print("=" * 60)

    return True


if __name__ == "__main__":
    import os

    # Cargar .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Obtener key
    api_key = os.getenv("SILICONFLOW_API_KEY", "")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    if args.api_key:
        api_key = args.api_key

    if not api_key:
        print("No API key. Set SILICONFLOW_API_KEY in .env")
        sys.exit(1)

    success = test_siliconflow(api_key)

    if success:
        print("\n✅ Ready for experiment! Run:")
        print("  python experiments/run_experiment.py --provider siliconflow --math-per-level 5 --gsm8k-size 10")
