# experiments/load_data.py
"""
Dataset loading and sampling for experiments.
Handles MATH and GSM8K datasets from HuggingFace.
Includes multiple fallback sources for robustness.
"""

import re
import random
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# MATH DATASET
# ═══════════════════════════════════════════════════════════

def load_math_dataset(
    sample_per_level: int = 100,
    levels: List[int] = [1, 2, 3, 4, 5],
    seed: int = 42,
) -> List[Dict]:
    """Load and sample from the MATH dataset."""
    loaders = [
        ("qwedsacf/competition_math", _load_math_qwedsacf),   # ← NUEVO: Tu hallazgo
        ("lighteval/MATH", _load_math_lighteval),
        ("hendrycks/competition_math", _load_math_hendrycks),
        ("manual_download", _load_math_manual),
    ]

    dataset = None
    for source_name, loader_func in loaders:
        try:
            logger.info(f"Trying MATH dataset source: {source_name}")
            dataset = loader_func()
            if dataset and len(dataset) > 0:
                logger.info(f"Successfully loaded from {source_name}: {len(dataset)} problems")
                break
        except Exception as e:
            logger.warning(f"Failed to load from {source_name}: {e}")
            continue

    if dataset is None or len(dataset) == 0:
        logger.error("Could not load MATH dataset from any source.")
        logger.info("Falling back to built-in sample problems.")
        return _get_builtin_math_problems(sample_per_level, levels, seed)

    return _sample_math(dataset, sample_per_level, levels, seed)

def _load_math_qwedsacf():
    """Load from qwedsacf/competition_math (community mirror)."""
    from datasets import load_dataset
    # Este dataset solo tiene split 'train' (contiene los 12,500 problemas)
    ds = load_dataset("qwedsacf/competition_math", split="train")
    all_problems = []
    for item in ds:
        all_problems.append({
            'problem': item.get('problem', ''),
            'solution': item.get('solution', ''),
            'level': item.get('level', 'Level 1'),
            'type': item.get('type', 'unknown'),
        })
    return all_problems


def _load_math_lighteval():
    """Load from lighteval/MATH (most reliable source)."""
    from datasets import load_dataset

    # lighteval/MATH has splits by subject
    subjects = [
        "algebra", "counting_and_probability",
        "geometry", "intermediate_algebra",
        "number_theory", "prealgebra", "precalculus"
    ]

    all_problems = []
    for subject in subjects:
        try:
            ds = load_dataset("lighteval/MATH", subject, split="test")
            for item in ds:
                all_problems.append({
                    'problem': item.get('problem', ''),
                    'solution': item.get('solution', ''),
                    'level': item.get('level', 'Level 1'),
                    'type': subject,
                })
        except Exception as e:
            logger.warning(f"Could not load subject {subject}: {e}")
            continue

    return all_problems


def _load_math_hendrycks():
    """Load from hendrycks/competition_math (may require auth)."""
    from datasets import load_dataset
    ds = load_dataset("hendrycks/competition_math", split="test")
    return [dict(item) for item in ds]


def _load_math_manual():
    """Download MATH dataset directly from GitHub."""
    import urllib.request
    import json
    import tempfile
    import zipfile
    import os

    url = "https://people.eecs.berkeley.edu/~hendrycks/MATH.tar"
    logger.info(f"Downloading MATH dataset from {url}...")

    # This is a large download, skip if not needed
    raise NotImplementedError("Manual download not implemented - use lighteval/MATH instead")


def _sample_math(
    raw_dataset: list,
    sample_per_level: int,
    levels: List[int],
    seed: int,
) -> List[Dict]:
    """
    Sample from MATH dataset, stratified by LEVEL and TYPE.
    Ensures representative coverage of mathematical domains.
    """
    random.seed(seed)
    sampled = []
    idx = 0

    # Group by level
    by_level = {}
    for item in raw_dataset:
        level_str = str(item.get('level', 'Level 1'))
        match = re.search(r'(\d+)', level_str)
        level_num = int(match.group(1)) if match else 1
        if level_num not in by_level:
            by_level[level_num] = []
        by_level[level_num].append(item)

    logger.info(f"MATH levels found: { {k: len(v) for k, v in by_level.items()} }")

    for level in levels:
        if level not in by_level:
            logger.warning(f"Level {level} not found in dataset")
            continue

        pool = by_level[level]

        # ══════════════════════════════════════════
        # Stratify by TYPE within each level
        # This ensures we get algebra, geometry,
        # number theory, etc. proportionally
        # ══════════════════════════════════════════
        by_type = {}
        for item in pool:
            t = item.get('type', 'unknown')
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(item)

        # Calculate proportional allocation per type
        n_total = min(sample_per_level, len(pool))
        type_counts = {}
        remaining = n_total

        for t, items in sorted(by_type.items()):
            proportion = len(items) / len(pool)
            count = max(1, round(proportion * n_total))
            count = min(count, len(items), remaining)
            type_counts[t] = count
            remaining -= count

        # Distribute any remaining slots
        while remaining > 0:
            for t in by_type:
                if remaining <= 0:
                    break
                if type_counts.get(t, 0) < len(by_type[t]):
                    type_counts[t] = type_counts.get(t, 0) + 1
                    remaining -= 1

        # Sample from each type
        level_samples = []
        for t, count in type_counts.items():
            available = by_type[t]
            selected = random.sample(available, min(count, len(available)))
            level_samples.extend(selected)

        # Shuffle within level
        random.shuffle(level_samples)

        for item in level_samples:
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

    # Log sampling distribution
    from collections import Counter
    type_dist = Counter(p['type'] for p in sampled)
    level_dist = Counter(p['level'] for p in sampled)
    logger.info(f"MATH: Sampled {len(sampled)} problems")
    logger.info(f"  By level: {dict(sorted(level_dist.items()))}")
    logger.info(f"  By type:  {dict(sorted(type_dist.items()))}")

    return sampled


# ═══════════════════════════════════════════════════════════
# GSM8K DATASET
# ═══════════════════════════════════════════════════════════

def load_gsm8k_dataset(
    sample_size: int = 200,
    seed: int = 42,
) -> List[Dict]:
    """Load and sample from GSM8K."""
    loaders = [
        ("openai/gsm8k", _load_gsm8k_openai),
        ("gsm8k", _load_gsm8k_alt),
    ]

    dataset = None
    for source_name, loader_func in loaders:
        try:
            logger.info(f"Trying GSM8K source: {source_name}")
            dataset = loader_func()
            if dataset and len(dataset) > 0:
                logger.info(f"Loaded from {source_name}: {len(dataset)} problems")
                break
        except Exception as e:
            logger.warning(f"Failed {source_name}: {e}")
            continue

    if dataset is None or len(dataset) == 0:
        logger.error("Could not load GSM8K from any source.")
        logger.info("Falling back to built-in sample problems.")
        return _get_builtin_gsm8k_problems(sample_size, seed)

    return _sample_gsm8k(dataset, sample_size, seed)


def _load_gsm8k_openai():
    """Load from openai/gsm8k."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [dict(item) for item in ds]


def _load_gsm8k_alt():
    """Alternative GSM8K source."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    return [dict(item) for item in ds]


def _sample_gsm8k(
    raw_dataset: list,
    sample_size: int,
    seed: int,
) -> List[Dict]:
    """Sample from loaded GSM8K dataset."""
    random.seed(seed)
    indices = random.sample(range(len(raw_dataset)), min(sample_size, len(raw_dataset)))

    sampled = []
    for i, idx in enumerate(indices):
        item = raw_dataset[idx]
        answer = _extract_gsm8k_answer(item.get('answer', ''))
        sampled.append({
            'id': f"gsm8k_{i}",
            'problem': item.get('question', item.get('problem', '')),
            'answer': answer,
            'level': 0,
            'type': 'arithmetic',
            'dataset': 'GSM8K',
        })

    logger.info(f"GSM8K: Sampled {len(sampled)} problems")
    return sampled


# ═══════════════════════════════════════════════════════════
# ANSWER EXTRACTION
# ═══════════════════════════════════════════════════════════

def _extract_math_answer(solution: str) -> str:
    """Extract the final answer from a MATH dataset solution."""
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
    """Extract numerical answer from GSM8K format (#### NUMBER)."""
    match = re.search(r'####\s*(.+?)$', answer_text, re.MULTILINE)
    if match:
        return match.group(1).strip().replace(',', '')

    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer_text)
    if numbers:
        return numbers[-1].replace(',', '')

    return ""


# ═══════════════════════════════════════════════════════════
# BUILT-IN FALLBACK PROBLEMS
# (If HuggingFace downloads fail, use these)
# ═══════════════════════════════════════════════════════════

def _get_builtin_math_problems(
    sample_per_level: int,
    levels: List[int],
    seed: int,
) -> List[Dict]:
    """
    Built-in math problems as fallback.
    Covers all 5 difficulty levels.
    """
    problems = [
        # ── Level 1: Basic ───────────────────────────
        {"problem": "What is the value of $2 + 3 \\times 4$?",
         "answer": "14", "level": 1, "type": "prealgebra"},
        {"problem": "Simplify $\\frac{12}{18}$.",
         "answer": "\\frac{2}{3}", "level": 1, "type": "prealgebra"},
        {"problem": "What is $25\\%$ of $80$?",
         "answer": "20", "level": 1, "type": "prealgebra"},
        {"problem": "If $x = 3$, what is $2x + 5$?",
         "answer": "11", "level": 1, "type": "prealgebra"},
        {"problem": "What is the area of a rectangle with length $5$ and width $3$?",
         "answer": "15", "level": 1, "type": "geometry"},
        {"problem": "Compute $7^2$.",
         "answer": "49", "level": 1, "type": "prealgebra"},
        {"problem": "What is $\\frac{3}{4} + \\frac{1}{4}$?",
         "answer": "1", "level": 1, "type": "prealgebra"},
        {"problem": "Solve for $x$: $x + 7 = 15$.",
         "answer": "8", "level": 1, "type": "algebra"},
        {"problem": "What is the perimeter of a square with side length $6$?",
         "answer": "24", "level": 1, "type": "geometry"},
        {"problem": "Compute $\\sqrt{81}$.",
         "answer": "9", "level": 1, "type": "prealgebra"},

        # ── Level 2: Elementary ──────────────────────
        {"problem": "Solve for $x$: $3x - 5 = 16$.",
         "answer": "7", "level": 2, "type": "algebra"},
        {"problem": "What is the value of $\\frac{2^4 \\cdot 3}{6}$?",
         "answer": "8", "level": 2, "type": "prealgebra"},
        {"problem": "If $f(x) = 2x + 1$, what is $f(5)$?",
         "answer": "11", "level": 2, "type": "algebra"},
        {"problem": "How many ways can you arrange 3 books on a shelf?",
         "answer": "6", "level": 2, "type": "counting_and_probability"},
        {"problem": "Find the slope of the line passing through $(1, 2)$ and $(3, 8)$.",
         "answer": "3", "level": 2, "type": "algebra"},
        {"problem": "What is $\\gcd(12, 18)$?",
         "answer": "6", "level": 2, "type": "number_theory"},
        {"problem": "Simplify $(x + 2)(x - 2)$.",
         "answer": "x^2 - 4", "level": 2, "type": "algebra"},
        {"problem": "A triangle has sides of length 3, 4, and 5. What is its area?",
         "answer": "6", "level": 2, "type": "geometry"},
        {"problem": "What is $2^{10}$?",
         "answer": "1024", "level": 2, "type": "prealgebra"},
        {"problem": "Solve for $x$: $\\frac{x}{3} = 7$.",
         "answer": "21", "level": 2, "type": "algebra"},

        # ── Level 3: Intermediate ────────────────────
        {"problem": "Solve for $x$: $x^2 - 5x + 6 = 0$.",
         "answer": "2", "level": 3, "type": "algebra"},
        {"problem": "What is the sum of the first 20 positive integers?",
         "answer": "210", "level": 3, "type": "algebra"},
        {"problem": "In how many ways can 5 people be seated in a row?",
         "answer": "120", "level": 3, "type": "counting_and_probability"},
        {"problem": "Find the value of $\\log_2 32$.",
         "answer": "5", "level": 3, "type": "algebra"},
        {"problem": "If $\\sin\\theta = \\frac{3}{5}$, what is $\\cos\\theta$ for acute $\\theta$?",
         "answer": "\\frac{4}{5}", "level": 3, "type": "precalculus"},
        {"problem": "What is the remainder when $2^{100}$ is divided by $3$?",
         "answer": "1", "level": 3, "type": "number_theory"},
        {"problem": "Solve the system: $x + y = 10$, $x - y = 4$.",
         "answer": "7", "level": 3, "type": "algebra"},
        {"problem": "What is $\\binom{7}{3}$?",
         "answer": "35", "level": 3, "type": "counting_and_probability"},
        {"problem": "Find the distance between points $(1, 2)$ and $(4, 6)$.",
         "answer": "5", "level": 3, "type": "geometry"},
        {"problem": "Simplify $\\frac{x^2 - 9}{x + 3}$.",
         "answer": "x - 3", "level": 3, "type": "algebra"},

        # ── Level 4: Advanced ────────────────────────
        {"problem": "Find the sum of all positive divisors of $72$.",
         "answer": "195", "level": 4, "type": "number_theory"},
        {"problem": "How many 4-digit numbers have digits that sum to 9?",
         "answer": "165", "level": 4, "type": "counting_and_probability"},
        {"problem": "If $x + \\frac{1}{x} = 5$, what is $x^2 + \\frac{1}{x^2}$?",
         "answer": "23", "level": 4, "type": "algebra"},
        {"problem": "Find the area of a triangle with vertices $(0,0)$, $(4,0)$, and $(2,3)$.",
         "answer": "6", "level": 4, "type": "geometry"},
        {"problem": "What is the coefficient of $x^3$ in the expansion of $(1+x)^7$?",
         "answer": "35", "level": 4, "type": "counting_and_probability"},
        {"problem": "Compute $\\sum_{k=1}^{10} k^2$.",
         "answer": "385", "level": 4, "type": "algebra"},
        {"problem": "Find $\\phi(100)$ (Euler's totient function).",
         "answer": "40", "level": 4, "type": "number_theory"},
        {"problem": "Solve $|2x - 3| = 7$.",
         "answer": "5", "level": 4, "type": "algebra"},
        {"problem": "What is the period of $\\sin(3x)$?",
         "answer": "\\frac{2\\pi}{3}", "level": 4, "type": "precalculus"},
        {"problem": "Find the inverse of $f(x) = \\frac{2x+1}{x-3}$.",
         "answer": "\\frac{3x+1}{x-2}", "level": 4, "type": "intermediate_algebra"},

        # ── Level 5: Competition ─────────────────────
        {"problem": "Find the number of integers $n$ with $1 \\le n \\le 1000$ such that $n^2 + 1$ is divisible by $5$.",
         "answer": "400", "level": 5, "type": "number_theory"},
        {"problem": "Compute $\\sum_{n=1}^{\\infty} \\frac{1}{n(n+1)}$.",
         "answer": "1", "level": 5, "type": "intermediate_algebra"},
        {"problem": "How many subsets of $\\{1, 2, 3, ..., 10\\}$ contain no two consecutive integers?",
         "answer": "144", "level": 5, "type": "counting_and_probability"},
        {"problem": "If $a + b + c = 6$ and $a^2 + b^2 + c^2 = 14$, find $ab + bc + ca$.",
         "answer": "11", "level": 5, "type": "algebra"},
        {"problem": "Find the last two digits of $7^{2023}$.",
         "answer": "43", "level": 5, "type": "number_theory"},
        {"problem": "Compute $\\int_0^1 x \\ln(x) \\, dx$.",
         "answer": "-\\frac{1}{4}", "level": 5, "type": "precalculus"},
        {"problem": "Find the minimum value of $x^2 + \\frac{1}{x}$ for $x > 0$.",
         "answer": "\\frac{3}{2^{2/3}}", "level": 5, "type": "intermediate_algebra"},
        {"problem": "In how many ways can you tile a $2 \\times 10$ board with $1 \\times 2$ dominoes?",
         "answer": "89", "level": 5, "type": "counting_and_probability"},
        {"problem": "Find $\\gcd(2^{30} - 1, 2^{20} - 1)$.",
         "answer": "1023", "level": 5, "type": "number_theory"},
        {"problem": "What is the radius of the inscribed circle of a triangle with sides 3, 4, 5?",
         "answer": "1", "level": 5, "type": "geometry"},
    ]

    random.seed(seed)
    sampled = []
    idx = 0

    by_level = {}
    for p in problems:
        lvl = p['level']
        if lvl not in by_level:
            by_level[lvl] = []
        by_level[lvl].append(p)

    for level in levels:
        if level not in by_level:
            continue
        pool = by_level[level]
        n = min(sample_per_level, len(pool))
        selected = random.sample(pool, n)
        for item in selected:
            sampled.append({
                'id': f"math_{idx}",
                'problem': item['problem'],
                'answer': item['answer'],
                'level': level,
                'type': item['type'],
                'dataset': 'MATH',
            })
            idx += 1

    logger.info(f"MATH (built-in): Loaded {len(sampled)} problems")
    return sampled


def _get_builtin_gsm8k_problems(
    sample_size: int,
    seed: int,
) -> List[Dict]:
    """Built-in GSM8K-style problems as fallback."""
    problems = [
        {"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
         "answer": "72"},
        {"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
         "answer": "10"},
        {"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to make up the remaining amount?",
         "answer": "5"},
        {"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
         "answer": "42"},
        {"question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
         "answer": "624"},
        {"question": "Mark has a garden with flowers. He planted plants of three colors in it. Ten of them are yellow, and there are 80% more of those in red. Blue flowers make up only 25% of red flowers. How many flowers does Mark have in his garden?",
         "answer": "35"},
        {"question": "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?",
         "answer": "48"},
        {"question": "Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds?",
         "answer": "16"},
        {"question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 less than her budget. If she spent $20 on a shirt and the rest on pants, how much did the pants cost?",
         "answer": "150"},
        {"question": "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?",
         "answer": "990"},
        {"question": "A farmer has 52 cows. Each cow gives 5 liters of milk per day. How many liters of milk does the farmer get in a week?",
         "answer": "1820"},
        {"question": "Sarah has 100 stickers. She gives 1/4 to her sister and 1/5 of the remaining to her friend. How many stickers does Sarah have left?",
         "answer": "60"},
        {"question": "A store sells notebooks for $3 each. If Tom buys 7 notebooks and pays with a $50 bill, how much change does he receive?",
         "answer": "29"},
        {"question": "Maria baked 60 cookies. She ate 5, gave 12 to her neighbor, and split the rest equally among 4 friends. How many cookies did each friend get?",
         "answer": "10.75"},
        {"question": "A train travels at 80 km/h. How far does it travel in 2 hours and 30 minutes?",
         "answer": "200"},
        {"question": "Jake has 3 times as many marbles as Tom. Tom has 15 marbles. They combine their marbles and split them equally. How many marbles does each person get?",
         "answer": "30"},
        {"question": "A pool is being filled at a rate of 20 gallons per minute. If the pool holds 2400 gallons, how many hours will it take to fill?",
         "answer": "2"},
        {"question": "Lisa earns $15 per hour. She worked 6 hours on Monday, 8 hours on Tuesday, and 4 hours on Wednesday. How much did she earn in total?",
         "answer": "270"},
        {"question": "A box contains 24 red balls and 36 blue balls. What fraction of the balls are red? Express as a simplified fraction.",
         "answer": "2/5"},
        {"question": "Peter has $250. He spends 40% on books and 25% of the remainder on food. How much money does he have left?",
         "answer": "112.5"},
    ]

    random.seed(seed)
    n = min(sample_size, len(problems))
    selected = random.sample(problems, n)

    sampled = []
    for i, item in enumerate(selected):
        sampled.append({
            'id': f"gsm8k_{i}",
            'problem': item['question'],
            'answer': item['answer'],
            'level': 0,
            'type': 'arithmetic',
            'dataset': 'GSM8K',
        })

    logger.info(f"GSM8K (built-in): Loaded {len(sampled)} problems")
    return sampled
