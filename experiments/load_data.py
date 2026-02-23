"""
Dataset loading and sampling for experiments.
Handles MATH and GSM8K datasets from HuggingFace.
"""

import random
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def load_math_dataset(
    sample_per_level: int = 100,
    levels: List[int] = [1, 2, 3, 4, 5],
    seed: int = 42,
) -> List[Dict]:
    """
    Load and sample from the MATH dataset (competition math).

    Args:
        sample_per_level: Number of problems per difficulty level.
        levels: Which difficulty levels to include (1-5).
        seed: Random seed.

    Returns:
        List of dicts with keys: 'id', 'problem', 'answer', 'level', 'type'.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    logger.info("Loading MATH dataset from HuggingFace...")
    dataset = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)

    random.seed(seed)
    sampled = []
    idx = 0

    # Group by level
    by_level = {}
    for item in dataset:
        level_str = item.get('level', 'Level 1')
        # Extract level number
        level_num = int(level_str.replace('Level ', '')) if 'Level' in str(level_str) else 1
        if level_num not in by_level:
            by_level[level_num] = []
        by_level[level_num].append(item)

    for level in levels:
        if level not in by_level:
            logger.warning(f"Level {level} not found in dataset")
            continue

        pool = by_level[level]
        n = min(sample_per_level, len(pool))
        selected = random.sample(pool, n)

        for item in selected:
            # Extract answer from solution
            answer = _extract_math_answer(item.get('solution', ''))
            sampled.append({
                'id': f"math_{idx}",
                'problem': item['problem'],
                'answer': answer,
                'level': level,
                'type': item.get('type', 'unknown'),
                'dataset': 'MATH',
            })
            idx += 1

    logger.info(
        f"MATH: Loaded {len(sampled)} problems "
        f"(levels {levels}, {sample_per_level} per level)"
    )
    return sampled


def load_gsm8k_dataset(
    sample_size: int = 200,
    seed: int = 42,
) -> List[Dict]:
    """
    Load and sample from the GSM8K dataset (grade school math).

    Args:
        sample_size: Total number of problems to sample.
        seed: Random seed.

    Returns:
        List of dicts with keys: 'id', 'problem', 'answer'.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")

    logger.info("Loading GSM8K dataset from HuggingFace...")
    dataset = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)

    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))

    sampled = []
    for i, idx in enumerate(indices):
        item = dataset[idx]
        answer = _extract_gsm8k_answer(item.get('answer', ''))
        sampled.append({
            'id': f"gsm8k_{i}",
            'problem': item['question'],
            'answer': answer,
            'level': 0,  # GSM8K doesn't have difficulty levels
            'type': 'arithmetic',
            'dataset': 'GSM8K',
        })

    logger.info(f"GSM8K: Loaded {len(sampled)} problems")
    return sampled


def _extract_math_answer(solution: str) -> str:
    """Extract the final answer from a MATH dataset solution."""
    import re

    # Look for \boxed{...}
    patterns = [
        r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}',
        r'\\boxed\{(.+?)\}',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, solution, re.DOTALL)
        if matches:
            return matches[-1].strip()

    return ""


def _extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the numerical answer from GSM8K format (#### NUMBER)."""
    import re

    match = re.search(r'####\s*(.+?)$', answer_text, re.MULTILINE)
    if match:
        # Clean: remove commas, whitespace
        answer = match.group(1).strip().replace(',', '')
        return answer

    # Fallback: last number
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer_text)
    if numbers:
        return numbers[-1].replace(',', '')

    return ""
