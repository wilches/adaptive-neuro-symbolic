"""
Mathematical Token Analyzer
Identifies and classifies tokens in mathematical text with LaTeX.
"""

import re
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass, field
from enum import Enum


class TokenType(Enum):
    """Classification of mathematical tokens."""
    VARIABLE = "variable"
    OPERATOR = "operator"
    NUMBER = "number"
    FUNCTION = "function"
    QUANTIFIER = "quantifier"
    RELATION = "relation"
    BRACKET = "bracket"
    LATEX_CMD = "latex_command"
    NATURAL_LANG = "natural_language"
    GREEK = "greek_letter"
    UNKNOWN = "unknown"


@dataclass
class MathToken:
    """A classified token from mathematical text."""
    text: str
    token_type: TokenType
    position: int


# ── Canonical sets ──────────────────────────────────────────

LATEX_FUNCTIONS: Set[str] = {
    'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
    'arcsin', 'arccos', 'arctan',
    'sinh', 'cosh', 'tanh',
    'log', 'ln', 'exp',
    'lim', 'limsup', 'liminf',
    'max', 'min', 'sup', 'inf',
    'det', 'dim', 'ker', 'deg',
    'gcd', 'lcm', 'mod',
    'arg', 'hom', 'Pr',
}

GREEK_LETTERS: Set[str] = {
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'varepsilon',
    'zeta', 'eta', 'theta', 'vartheta', 'iota', 'kappa',
    'lambda', 'mu', 'nu', 'xi', 'pi', 'varpi',
    'rho', 'varrho', 'sigma', 'varsigma', 'tau', 'upsilon',
    'phi', 'varphi', 'chi', 'psi', 'omega',
    'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi',
    'Sigma', 'Upsilon', 'Phi', 'Psi', 'Omega',
}

RELATION_OPERATORS: Set[str] = {
    '=', '<', '>', '\\leq', '\\geq', '\\neq',
    '\\le', '\\ge', '\\ne',
    '\\equiv', '\\approx', '\\sim', '\\simeq',
    '\\subset', '\\supset', '\\subseteq', '\\supseteq',
    '\\in', '\\notin', '\\ni',
    '\\mid', '\\nmid', '\\parallel', '\\perp',
    '\\propto', '\\cong',
    '≤', '≥', '≠', '∈', '∉', '⊂', '⊃', '⊆', '⊇',
}

ARITHMETIC_OPERATORS: Set[str] = {
    '+', '-', '*', '/', '^', '_',
    '\\cdot', '\\times', '\\div', '\\pm', '\\mp',
    '\\oplus', '\\otimes', '\\circ',
    '\\cup', '\\cap', '\\setminus',
    '\\wedge', '\\vee', '\\neg',
    '\\land', '\\lor', '\\lnot',
    '∧', '∨', '¬', '∪', '∩',
}

BIG_OPERATORS: Set[str] = {
    '\\sum', '\\prod', '\\coprod',
    '\\int', '\\iint', '\\iiint', '\\oint',
    '\\bigcup', '\\bigcap',
    '\\bigoplus', '\\bigotimes',
    '∑', '∏', '∫',
}

QUANTIFIER_WORDS_UNIVERSAL: List[str] = [
    r'\\forall', r'∀',
    r'\bfor\s+all\b', r'\bfor\s+every\b', r'\bfor\s+each\b',
    r'\bfor\s+any\b', r'\ball\b.*\bsatisfy\b', r'\bevery\b',
    r'\bwhenever\b', r'\balways\b',
]

QUANTIFIER_WORDS_EXISTENTIAL: List[str] = [
    r'\\exists', r'∃',
    r'\bthere\s+exist(s)?\b', r'\bthere\s+is\b', r'\bthere\s+are\b',
    r'\bsome\b', r'\bat\s+least\s+one\b',
    r'\bfind\s+(a|an|the)\b', r'\bexist(s)?\b',
]

LOGICAL_CONNECTIVES: List[str] = [
    r'\bif\b', r'\bthen\b', r'\bimplies\b', r'\btherefore\b',
    r'\bhence\b', r'\bthus\b', r'\bsince\b', r'\bbecause\b',
    r'\bsuppose\b', r'\bassume\b', r'\bgiven\s+that\b',
    r'\bit\s+follows\b', r'\bconsequently\b',
    r'\bif\s+and\s+only\s+if\b', r'\biff\b', r'\bsuch\s+that\b',
    r'\bprove\s+that\b', r'\bshow\s+that\b', r'\bdemonstrate\b',
    r'\blet\b', r'\bdefine\b', r'\bwhere\b',
    r'\\implies', r'\\iff', r'\\Rightarrow', r'\\Leftrightarrow',
    r'→', r'↔', r'⟹', r'⟺',
]


def tokenize_math_text(text: str) -> List[MathToken]:
    """
    Tokenize mathematical text into classified MathTokens.

    Handles LaTeX, Unicode math symbols, and natural language.
    """
    tokens: List[MathToken] = []
    pos = 0

    # ── Pass 1: identify LaTeX commands ─────────────────
    for match in re.finditer(r'\\([a-zA-Z]+)', text):
        cmd = match.group(1)
        if cmd in GREEK_LETTERS:
            tokens.append(MathToken(match.group(0), TokenType.GREEK, match.start()))
        elif cmd in LATEX_FUNCTIONS:
            tokens.append(MathToken(match.group(0), TokenType.FUNCTION, match.start()))
        elif match.group(0) in BIG_OPERATORS:
            tokens.append(MathToken(match.group(0), TokenType.OPERATOR, match.start()))
        elif match.group(0) in RELATION_OPERATORS:
            tokens.append(MathToken(match.group(0), TokenType.RELATION, match.start()))
        elif match.group(0) in ARITHMETIC_OPERATORS:
            tokens.append(MathToken(match.group(0), TokenType.OPERATOR, match.start()))
        else:
            tokens.append(MathToken(match.group(0), TokenType.LATEX_CMD, match.start()))

    # ── Pass 2: identify non-LaTeX math tokens ─────────
    for match in re.finditer(r'[+\-*/=<>^_≤≥≠∈∉⊂⊃∪∩∧∨¬→↔∀∃∑∏∫]', text):
        tokens.append(MathToken(match.group(0), TokenType.OPERATOR, match.start()))

    # ── Pass 3: identify numbers ────────────────────────
    for match in re.finditer(r'\b\d+(\.\d+)?\b', text):
        tokens.append(MathToken(match.group(0), TokenType.NUMBER, match.start()))

    # ── Pass 4: brackets ────────────────────────────────
    for match in re.finditer(r'[(){}$$$$]', text):
        tokens.append(MathToken(match.group(0), TokenType.BRACKET, match.start()))

    tokens.sort(key=lambda t: t.position)
    return tokens


def count_math_vs_nl_tokens(text: str) -> Tuple[int, int]:
    """
    Count mathematical tokens vs. natural language tokens.

    Returns:
        (math_count, nl_count)
    """
    math_tokens = tokenize_math_text(text)
    math_count = len([
        t for t in math_tokens
        if t.token_type != TokenType.NATURAL_LANG
    ])

    # Remove math content, count remaining words
    cleaned = text
    # Remove LaTeX
    cleaned = re.sub(r'\$[^$]+\$', ' ', cleaned)
    cleaned = re.sub(r'\\\([^)]+\\\)', ' ', cleaned)
    cleaned = re.sub(r'\\$$[^$$]+\\\]', ' ', cleaned)
    cleaned = re.sub(r'\$$a-zA-Z]+', ' ', cleaned)
    cleaned = re.sub(r'[+\-*/=<>^_{}()\\≤≥≠∈∉⊂⊃∪∩∧∨¬→↔∀∃∑∏∫\d]', ' ', cleaned)
    nl_words = [w for w in cleaned.split() if len(w) > 1]
    nl_count = len(nl_words)

    return math_count, nl_count


def extract_quantifier_sequence(text: str) -> List[str]:
    """
    Extract the ordered sequence of quantifiers (universal / existential)
    from mathematical text.

    Returns:
        List of 'universal' or 'existential' strings in order of appearance.
    """
    text_lower = text.lower()
    quantifier_positions: List[Tuple[int, str]] = []

    for pattern in QUANTIFIER_WORDS_UNIVERSAL:
        for match in re.finditer(pattern, text_lower):
            quantifier_positions.append((match.start(), 'universal'))

    for pattern in QUANTIFIER_WORDS_EXISTENTIAL:
        for match in re.finditer(pattern, text_lower):
            quantifier_positions.append((match.start(), 'existential'))

    quantifier_positions.sort(key=lambda x: x[0])
    return [q[1] for q in quantifier_positions]


def count_logical_connectives(text: str) -> int:
    """Count the number of logical connectives / reasoning keywords."""
    text_lower = text.lower()
    count = 0
    for pattern in LOGICAL_CONNECTIVES:
        count += len(re.findall(pattern, text_lower))
    return count
