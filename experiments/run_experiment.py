"""
Main experiment runner.
Executes all baselines and the adaptive system, collects results.
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import AdaptivePipeline
from src.config import (
    get_openai_config,
    get_siliconflow_config,
    get_gemini_config,
    ExperimentConfig,
)
from experiments.load_data import load_math_dataset, load_gsm8k_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def run_full_experiment(
    provider: str = "siliconflow",
    api_key: str = "",
    math_per_level: int = 20,       # Start small for testing!
    gsm8k_size: int = 40,
    output_dir: str = "results",
):
    """
    Run the complete experiment suite.

    Phases:
        1. Load datasets
        2. Run baseline: Symbolic-only
        3. Run baseline: CoT-only
        4. Run baseline: Fixed-Hybrid
        5. Run adaptive system
        6. Generate summary
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Setup provider ───────────────────────────────
    if provider == "siliconflow":
        config = get_siliconflow_config(api_key)
    elif provider == "openai":
        config = get_openai_config(api_key)
    elif provider == "gemini":
        config = get_gemini_config(api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    logger.info(f"Provider: {config.name}/{config.model}")

    # ── Phase 1: Load data ───────────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 1: Loading datasets")
    logger.info("=" * 60)

    math_data = load_math_dataset(sample_per_level=math_per_level)
    gsm8k_data = load_gsm8k_dataset(sample_size=gsm8k_size)
    all_data = math_data + gsm8k_data

    logger.info(f"Total problems: {len(all_data)} "
                f"(MATH: {len(math_data)}, GSM8K: {len(gsm8k_data)})")

    # Save dataset info
    with open(out_dir / "dataset_info.json", 'w') as f:
        json.dump({
            'math_count': len(math_data),
            'gsm8k_count': len(gsm8k_data),
            'total': len(all_data),
            'math_per_level': math_per_level,
        }, f, indent=2)

    # ── Phase 2: Symbolic-only baseline ──────────────
    logger.info("=" * 60)
    logger.info("PHASE 2: Symbolic-only baseline")
    logger.info("=" * 60)

    pipeline_sym = AdaptivePipeline(cot_config=config)
    results_sym = pipeline_sym.batch_solve(
        all_data,
        prompt_type="math",
        force_route="symbolic",
        save_path=str(out_dir / "results_symbolic.json"),
    )
    summary_sym = pipeline_sym.get_summary()
    logger.info(f"Symbolic-only: Accuracy={summary_sym['accuracy']:.4f}, "
                f"Tokens={summary_sym['total_tokens']}")

    # ── Phase 3: CoT-only baseline ───────────────────
    logger.info("=" * 60)
    logger.info("PHASE 3: CoT-only baseline")
    logger.info("=" * 60)

    pipeline_cot = AdaptivePipeline(cot_config=config)

    # Use appropriate prompt type for each dataset
    math_results_cot = pipeline_cot.batch_solve(
        math_data,
        prompt_type="math",
        force_route="neural_cot",
        save_path=str(out_dir / "results_cot_math.json"),
    )
    gsm8k_results_cot = pipeline_cot.batch_solve(
        gsm8k_data,
        prompt_type="gsm8k",
        force_route="neural_cot",
        save_path=str(out_dir / "results_cot_gsm8k.json"),
    )
    summary_cot = pipeline_cot.get_summary()
    logger.info(f"CoT-only: Accuracy={summary_cot['accuracy']:.4f}, "
                f"Tokens={summary_cot['total_tokens']}")

    # ── Phase 4: Fixed-Hybrid baseline ───────────────
    logger.info("=" * 60)
    logger.info("PHASE 4: Fixed-Hybrid baseline")
    logger.info("=" * 60)

    pipeline_hyb = AdaptivePipeline(cot_config=config)
    results_hyb = pipeline_hyb.batch_solve(
        all_data,
        prompt_type="math",
        force_route="hybrid",
        save_path=str(out_dir / "results_hybrid.json"),
    )
    summary_hyb = pipeline_hyb.get_summary()
    logger.info(f"Fixed-Hybrid: Accuracy={summary_hyb['accuracy']:.4f}, "
                f"Tokens={summary_hyb['total_tokens']}")

    # ── Phase 5: Adaptive system ─────────────────────
    logger.info("=" * 60)
    logger.info("PHASE 5: Adaptive system (OURS)")
    logger.info("=" * 60)

    pipeline_adaptive = AdaptivePipeline(cot_config=config)

    math_results_adapt = pipeline_adaptive.batch_solve(
        math_data,
        prompt_type="math",
        save_path=str(out_dir / "results_adaptive_math.json"),
    )
    gsm8k_results_adapt = pipeline_adaptive.batch_solve(
        gsm8k_data,
        prompt_type="gsm8k",
        save_path=str(out_dir / "results_adaptive_gsm8k.json"),
    )
    summary_adaptive = pipeline_adaptive.get_summary()
    logger.info(f"Adaptive: Accuracy={summary_adaptive['accuracy']:.4f}, "
                f"Tokens={summary_adaptive['total_tokens']}")

    # ── Phase 6: Generate final summary ──────────────
    logger.info("=" * 60)
    logger.info("PHASE 6: Final Summary")
    logger.info("=" * 60)

    final_summary = {
        'experiment_timestamp': timestamp,
        'provider': f"{config.name}/{config.model}",
        'dataset_sizes': {
            'math': len(math_data),
            'gsm8k': len(gsm8k_data),
            'total': len(all_data),
        },
        'results': {
            'symbolic_only': summary_sym,
            'cot_only': summary_cot,
            'fixed_hybrid': summary_hyb,
            'adaptive': summary_adaptive,
        },
    }

    with open(out_dir / "final_summary.json", 'w') as f:
        json.dump(final_summary, f, indent=2, default=str)

    # ── Print comparison table ───────────────────────
    _print_comparison_table(final_summary)

    return final_summary


def _print_comparison_table(summary: dict):
    """Print a formatted comparison table."""
    results = summary['results']

    print("\n" + "=" * 75)
    print("RESULTS COMPARISON")
    print("=" * 75)
    print(f"{'Method':<20} {'Accuracy':>10} {'Tokens':>12} {'Cost (USD)':>12}")
    print("-" * 75)

    for method, data in results.items():
        name = method.replace('_', ' ').title()
        acc = f"{data.get('accuracy', 0):.4f}"
        tokens = f"{data.get('total_tokens', 0):,}"
        cost = f"${data.get('total_cost_usd', 0):.6f}"
        print(f"{name:<20} {acc:>10} {tokens:>12} {cost:>12}")

    print("=" * 75)


# ── Entry Point ──────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run adaptive reasoning experiment")
    parser.add_argument("--provider", default="siliconflow",
                        choices=["siliconflow", "openai", "gemini"])
    parser.add_argument("--api-key", required=True, help="API key for the provider")
    parser.add_argument("--math-per-level", type=int, default=20,
                        help="MATH problems per difficulty level")
    parser.add_argument("--gsm8k-size", type=int, default=40,
                        help="Number of GSM8K problems")
    parser.add_argument("--output-dir", default="results")

    args = parser.parse_args()

    run_full_experiment(
        provider=args.provider,
        api_key=args.api_key,
        math_per_level=args.math_per_level,
        gsm8k_size=args.gsm8k_size,
        output_dir=args.output_dir,
    )
