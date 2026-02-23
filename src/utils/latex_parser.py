"""
Lightweight LaTeX Math Parser
Extracts and processes mathematical expressions from LaTeX text.
No dependency on antlr4 — pure regex + heuristic parsing.
"""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MathExpression:
    """A mathematical expression extracted from text."""
    raw: str           # Original LaTeX string
    cleaned: str       # Cleaned / normalized version
    start_pos: int     # Position in original text
    end_pos: int
    is_display: bool   # True if display math ($$, \[)


def extract_math_expressions(text: str) -> List[MathExpression]:
    """
    Extract all mathematical expressions from LaTeX text.

    Handles: $...$,
$$
...
$$
, \\(...\\), \\[...\$$, and bare expressions.
    """
    expressions: List[MathExpression] = []

    # ── Display math:$$$$───────────────────────
    for match in re.finditer(r'\$\$(.+?)\$\$', text, re.DOTALL):
        expressions.append(MathExpression(
            raw=match.group(1),
            cleaned=_clean_latex(match.group(1)),
            start_pos=match.start(),
            end_pos=match.end(),
            is_display=True
        ))

    # ── Display math: $$...$$ ───────────────────────
    for match in re.finditer(r'\\$$(.+?)\\$$', text, re.DOTALL):
        expressions.append(MathExpression(
            raw=match.group(1),
            cleaned=_clean_latex(match.group(1)),
            start_pos=match.start(),
            end_pos=match.end(),
            is_display=True
        ))

    # ── Inline math: $...$ (not$$)─────────────────
    for match in re.finditer(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', text):
        # Skip if already captured as display math
        if not any(e.start_pos <= match.start() < e.end_pos for e in expressions):
            expressions.append(MathExpression(
                raw=match.group(1),
                cleaned=_clean_latex(match.group(1)),
                start_pos=match.start(),
                end_pos=match.end(),
                is_display=False
            ))

    # ── Inline math: \(...\) ────────────────────────
    for match in re.finditer(r'\\\((.+?)\\\)', text):
        if not any(e.start_pos <= match.start() < e.end_pos for e in expressions):
            expressions.append(MathExpression(
                raw=match.group(1),
                cleaned=_clean_latex(match.group(1)),
                start_pos=match.start(),
                end_pos=match.end(),
                is_display=False
            ))

    # ── Sort by position ────────────────────────────
    expressions.sort(key=lambda e: e.start_pos)
    return expressions


def compute_expression_tree_depth(latex_expr: str) -> int:
    """
    Estimate the depth of a mathematical expression tree from LaTeX.

    Uses a heuristic approach:
    - Each nested operation (frac, sqrt, sum, etc.) adds depth.
    - Brace nesting is tracked.
    - Operator composition is estimated.
    """
    depth_score = 0

    # ── Nesting-increasing constructs ───────────────
    construct_depths = {
        r'\\frac': 2.0,       # Fraction: numerator + denominator = 2 branches
        r'\\dfrac': 2.0,
        r'\\tfrac': 2.0,
        r'\\sqrt': 1.5,       # Square root: 1 branch + index
        r'\\sum': 2.0,        # Sum: bounds + body
        r'\\prod': 2.0,
        r'\\int': 2.0,        # Integral: bounds + integrand
        r'\\iint': 2.5,
        r'\\iiint': 3.0,
        r'\\lim': 1.5,
        r'\\binom': 2.0,      # Binomial: n, k
        r'\\overset': 1.5,
        r'\\underset': 1.5,
    }

    for pattern, d in construct_depths.items():
        count = len(re.findall(pattern, latex_expr))
        depth_score += count * d

    # ── Function composition ────────────────────────
    functions = [
        r'\\sin', r'\\cos', r'\\tan', r'\\log', r'\\ln', r'\\exp',
        r'\\arcsin', r'\\arccos', r'\\arctan',
    ]
    func_count = sum(len(re.findall(f, latex_expr)) for f in functions)
    depth_score += func_count * 1.0

    # ── Exponentiation chains ───────────────────────
    exp_count = len(re.findall(r'\^', latex_expr))
    depth_score += exp_count * 1.0

    # ── Subscript chains ────────────────────────────
    sub_count = len(re.findall(r'_', latex_expr))
    depth_score += sub_count * 0.5

    return int(round(depth_score))


def compute_max_brace_depth(text: str) -> int:
    """Compute maximum brace/bracket nesting depth."""
    max_depth = 0
    current = 0
    openers = {'(', '[', '{'}
    closers = {')', ']', '}'}

    for char in text:
        if char in openers:
            current += 1
            max_depth = max(max_depth, current)
        elif char in closers:
            current = max(0, current - 1)

    return max_depth


def extract_variables_from_latex(text: str) -> set:
    """
    Extract mathematical variable names from LaTeX text.

    Identifies:
    - Single-letter variables (x, y, z, n, k, ...)
    - Greek-letter variables (\\alpha, \\beta, ...)
    - Subscripted variables (x_1, a_{ij}, ...)
    """
    variables = set()

    # ── Greek letters used as variables ─────────────
    greek_pattern = (
        r'\\(alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|'
        r'theta|vartheta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|'
        r'tau|upsilon|phi|varphi|chi|psi|omega|'
        r'Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega)'
    )
    for match in re.finditer(greek_pattern, text):
        variables.add('\\' + match.group(1))

    # ── Remove known non-variable commands ──────────
    exclusions = {
        'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
        'arcsin', 'arccos', 'arctan',
        'sinh', 'cosh', 'tanh',
        'log', 'ln', 'exp', 'lim', 'max', 'min',
        'sup', 'inf', 'det', 'dim', 'ker', 'deg',
        'gcd', 'lcm', 'mod', 'arg', 'hom',
        'frac', 'dfrac', 'tfrac', 'sqrt', 'text',
        'left', 'right', 'big', 'Big', 'bigg', 'Bigg',
        'cdot', 'times', 'div', 'pm', 'mp',
        'leq', 'geq', 'neq', 'le', 'ge', 'ne',
        'equiv', 'approx', 'sim', 'simeq',
        'subset', 'supset', 'subseteq', 'supseteq',
        'in', 'notin', 'ni', 'mid', 'nmid',
        'forall', 'exists', 'nexists',
        'sum', 'prod', 'coprod', 'int', 'iint', 'iiint', 'oint',
        'cup', 'cap', 'bigcup', 'bigcap',
        'wedge', 'vee', 'neg', 'land', 'lor', 'lnot',
        'implies', 'iff', 'Rightarrow', 'Leftrightarrow',
        'mathbb', 'mathcal', 'mathfrak', 'mathrm', 'mathbf',
        'binom', 'choose', 'pmod',
        'infty', 'emptyset', 'varnothing',
        'quad', 'qquad', 'hspace', 'vspace',
        'begin', 'end', 'item', 'label', 'ref',
        'overline', 'underline', 'hat', 'bar', 'tilde', 'vec', 'dot',
    }

    # ── Remove all math content delimiters ──────────
    cleaned = text
    cleaned = re.sub(r'\$\$(.+?)\$\$', r'\1', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'\$(.+?)\$', r'\1', cleaned)
    cleaned = re.sub(r'\\\((.+?)\\\)', r'\1', cleaned)
    cleaned = re.sub(r'\\$$(.+?)\\$$', r'\1', cleaned, flags=re.DOTALL)

    # ── Remove LaTeX commands (keep arguments) ──────
    cleaned_for_vars = re.sub(r'\\[a-zA-Z]+', ' ', cleaned)

    # ── Single-letter variables ─────────────────────
    # Context-aware: look for isolated single letters
    words = re.findall(r'\b[a-zA-Z]\b', cleaned_for_vars)
    for w in words:
        if w.lower() not in {'a', 'i'}:
            # 'a' and 'I' could be articles/pronouns in English,
            # but in math context they're usually variables.
            # We keep them but could refine with context.
            variables.add(w)

    # ── Subscripted variables: x_1, x_{ij}, a_n ────
    for match in re.finditer(r'([a-zA-Z])_\{?([^}\s,;.]+)\}?', cleaned):
        base = match.group(1)
        sub = match.group(2)
        variables.add(f"{base}_{sub}")
        # Also add the base as a "family" variable
        variables.add(base)

    return variables


def _clean_latex(expr: str) -> str:
    """Clean / normalize a LaTeX expression."""
    cleaned = expr.strip()
    # Remove display style commands
    cleaned = re.sub(r'\\(displaystyle|textstyle|scriptstyle)', '', cleaned)
    # Normalize spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned
