"""
Structural Complexity Analyzer for Mathematical Prompts
========================================================

Core module of the Adaptive Neuro-Symbolic Reasoning system.
Computes five structural complexity metrics for each mathematical prompt and produces a composite score used by the Router for path selection.

Metrics:
    Φ(p) = ⟨d_L(p), n_V(p), σ_S(p), ω_N(p), κ_Q(p)⟩

    d_L : Logical Depth — depth of the mathematical expression tree
    n_V : Variable Count — number of distinct mathematical variables
    σ_S : Symbolic Structure — ratio of symbolic to natural-language content
    ω_N : Nesting Complexity — max depth of nested sub-expressions
    κ_Q : Quantifier Alternation — alternation count of ∀/∃ quantifiers

Author: Juan Camilo Herrera Wilches
Date: February 2025
"""

import json
import re
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from src.utils.math_tokenizer import (
    count_math_vs_nl_tokens,
    extract_quantifier_sequence,
    count_logical_connectives,
)
from src.utils.latex_parser import (
    extract_math_expressions,
    compute_expression_tree_depth,
    compute_max_brace_depth,
    extract_variables_from_latex,
)


# ─── Data Classes ────────────────────────────────────────────

@dataclass
class ComplexityProfile:
    """
    Complete structural complexity profile for a mathematical prompt.

    Attributes:
        logical_depth (float):
            Estimated depth of the expression/reasoning tree.
        variable_count (int):
            Number of distinct mathematical variables identified.
        symbolic_structure (float):
            Ratio ∈ [0,1] of symbolic content to total content.
        nesting_complexity (int):
            Maximum nesting depth of brackets/constructs.
        quantifier_alternation (int):
            Number of ∀↔∃ alternations in quantifier sequence.
        composite_score (float):
            Weighted combination ∈ [0,1] of all normalized metrics.
        routing_suggestion (str):
            Preliminary routing suggestion based on thresholds.
        raw_metrics (dict):
            Dictionary with raw (un-normalized) metric values.
    """
    logical_depth: float
    variable_count: int
    symbolic_structure: float
    nesting_complexity: int
    quantifier_alternation: int
    composite_score: float
    routing_suggestion: str
    raw_metrics: dict

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ─── Main Analyzer Class ────────────────────────────────────

class ComplexityAnalyzer:
    """
    Analyzes structural complexity of mathematical prompts.

    The analyzer computes five metrics that together characterize
    the structural difficulty and symbolic density of a prompt.
    These metrics drive the Router's path-selection logic.

    Usage:
        analyzer = ComplexityAnalyzer()
        profile = analyzer.analyze("Prove that for all n ≥ 1, $\\sum_{k=1}^{n} k = \\frac{n(n+1)}{2}$")
        print(profile.composite_score)
        print(profile.routing_suggestion)
    """

    # ── Default metric weights for composite score ──
    DEFAULT_WEIGHTS: Dict[str, float] = {
        'logical_depth':          0.25,
        'variable_count':         0.15,
        'symbolic_structure':     0.25,
        'nesting_complexity':     0.15,
        'quantifier_alternation': 0.20,
    }

    # ── Default routing thresholds ──────────────────
    DEFAULT_THRESHOLDS: Dict[str, float] = {
        'tau_low':  0.30,   # Below this → SYMBOLIC path
        'tau_high': 0.65,   # Above this → NEURAL CoT path
        'sigma_symbolic_threshold': 0.60,  # σ_S above this favors symbolic
        'sigma_neural_threshold':   0.30,  # σ_S below this favors neural
    }

    # ── Normalization bounds (calibrated on MATH dataset) ──
    DEFAULT_NORM_PARAMS: Dict[str, Dict[str, float]] = {
        'logical_depth':          {'min': 0.0, 'max': 25.0},
        'variable_count':         {'min': 0.0, 'max': 15.0},
        'symbolic_structure':     {'min': 0.0, 'max': 1.0},
        'nesting_complexity':     {'min': 0.0, 'max': 12.0},
        'quantifier_alternation': {'min': 0.0, 'max': 5.0},
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        norm_params: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS.copy()
        self.norm_params = norm_params or {
            k: v.copy() for k, v in self.DEFAULT_NORM_PARAMS.items()
        }

        # Validate weights sum to 1
        w_sum = sum(self.weights.values())
        if abs(w_sum - 1.0) > 1e-6:
            # Auto-normalize
            self.weights = {k: v / w_sum for k, v in self.weights.items()}

    # ─── PUBLIC API ──────────────────────────────────

    def analyze(self, prompt: str) -> ComplexityProfile:
        """
        Compute the full complexity profile for a mathematical prompt.

        Args:
            prompt: Mathematical problem text (may contain LaTeX).

        Returns:
            ComplexityProfile with all metrics and routing suggestion.
        """
        # ── Step 1: Compute raw metrics ─────────────
        d_L = self._compute_logical_depth(prompt)
        n_V = self._compute_variable_count(prompt)
        sigma_S = self._compute_symbolic_structure(prompt)
        omega_N = self._compute_nesting_complexity(prompt)
        kappa_Q = self._compute_quantifier_alternation(prompt)

        raw_metrics = {
            'logical_depth': d_L,
            'variable_count': n_V,
            'symbolic_structure': sigma_S,
            'nesting_complexity': omega_N,
            'quantifier_alternation': kappa_Q,
        }

        # ── Step 2: Compute composite score ─────────
        composite = self._compute_composite_score(
            d_L, n_V, sigma_S, omega_N, kappa_Q
        )

        # ── Step 3: Determine routing suggestion ────
        routing = self._suggest_route(composite, sigma_S, kappa_Q)

        return ComplexityProfile(
            logical_depth=d_L,
            variable_count=n_V,
            symbolic_structure=sigma_S,
            nesting_complexity=omega_N,
            quantifier_alternation=kappa_Q,
            composite_score=round(composite, 4),
            routing_suggestion=routing,
            raw_metrics=raw_metrics,
        )

    def batch_analyze(
        self, prompts: List[str], calibrate: bool = True
    ) -> List[ComplexityProfile]:
        """
        Analyze a batch of prompts. Optionally calibrate normalization
        parameters from the observed distribution (two-pass).

        Args:
            prompts: List of mathematical prompt strings.
            calibrate: If True, run a first pass to set norm bounds.

        Returns:
            List of ComplexityProfile objects.
        """
        if calibrate and len(prompts) > 10:
            self._calibrate_normalization(prompts)

        return [self.analyze(p) for p in prompts]

    def get_config(self) -> dict:
        """Return current analyzer configuration."""
        return {
            'weights': self.weights,
            'thresholds': self.thresholds,
            'norm_params': self.norm_params,
        }

    def set_thresholds(self, tau_low: float, tau_high: float):
        """Update routing thresholds."""
        assert 0 <= tau_low < tau_high <= 1, \
            f"Need 0 ≤ τ_low < τ_high ≤ 1, got {tau_low}, {tau_high}"
        self.thresholds['tau_low'] = tau_low
        self.thresholds['tau_high'] = tau_high

    def set_weights(self, weights: Dict[str, float]):
        """Update metric weights (auto-normalized to sum to 1)."""
        w_sum = sum(weights.values())
        self.weights = {k: v / w_sum for k, v in weights.items()}

    # ─── METRIC COMPUTATIONS ─────────────────────────

    def _compute_logical_depth(self, prompt: str) -> float:
        """
        Metric d_L: Logical Depth

        Combines:
        - Expression tree depth from LaTeX constructs
        - Count of logical connectives in natural language
        - Proof-step indicators (implies, therefore, hence...)

        Higher d_L → more complex reasoning chain required.
        """
        # ── Part A: Expression tree depth from LaTeX ──
        math_exprs = extract_math_expressions(prompt)
        max_expr_depth = 0
        for expr in math_exprs:
            depth = compute_expression_tree_depth(expr.raw)
            max_expr_depth = max(max_expr_depth, depth)

        # ── Part B: Logical connective count ──────────
        connective_count = count_logical_connectives(prompt)

        # ── Part C: Multi-step indicators ─────────────
        step_patterns = [
            r'\bstep\s+\d+\b',
            r'\bfirst\b.*\bthen\b',
            r'\b(i+|1|2|3|4|5)\)',    # numbered steps
            r'\bcase\s+\d+\b',
            r'\bby\s+induction\b',
            r'\bbase\s+case\b',
            r'\binductive\s+step\b',
            r'\bcontradiction\b',
            r'\bcontrapositive\b',
            r'\bwithout\s+loss\s+of\s+generality\b',
            r'\bWLOG\b',
        ]
        step_count = sum(
            len(re.findall(p, prompt, re.IGNORECASE))
            for p in step_patterns
        )

        # Combine: weighted sum
        d_L = max_expr_depth + 0.8 * connective_count + 1.2 * step_count
        return round(d_L, 2)

    def _compute_variable_count(self, prompt: str) -> int:
        """
        Metric n_V: Variable Count

        Counts distinct mathematical variables including:
        - Single-letter variables (x, y, z, n, k)
        - Greek letters (α, β, θ, ...)
        - Subscripted variants (x₁, aᵢⱼ, ...)

        Higher n_V → more complex variable tracking needed.
        """
        variables = extract_variables_from_latex(prompt)
        return len(variables)

    def _compute_symbolic_structure(self, prompt: str) -> float:
        """
        Metric σ_S: Symbolic Structure Score

        Ratio of mathematical/symbolic tokens to total content.
        ∈ [0, 1] where:
        - σ_S → 1: heavily symbolic (candidate for symbolic solver)
        - σ_S → 0: mostly natural language (candidate for CoT)
        """
        math_count, nl_count = count_math_vs_nl_tokens(prompt)
        total = math_count + nl_count

        if total == 0:
            return 0.0

        sigma = math_count / total
        return round(min(1.0, sigma), 4)

    def _compute_nesting_complexity(self, prompt: str) -> int:
        """
        Metric ω_N: Nesting Complexity

        Maximum depth of nested mathematical constructs:
        - Parentheses / brackets / braces
        - LaTeX environments (\frac{\frac{...}{...}}{...})

        Higher ω_N → deeper sub-expression structure.
        """
        return compute_max_brace_depth(prompt)

    def _compute_quantifier_alternation(self, prompt: str) -> int:
        """
        Metric κ_Q: Quantifier Alternation

        Counts alternations between universal (∀) and existential (∃)
        quantifiers in order of appearance.

        Relates to the arithmetic hierarchy:
        - 0 alternations: Σ₁/Π₁ (simpler)
        - k alternations: relates to Σₖ/Πₖ complexity

        Higher κ_Q → more complex logical structure → harder for solvers.
        """
        quantifier_seq = extract_quantifier_sequence(prompt)

        if len(quantifier_seq) <= 1:
            return 0

        alternations = sum(
            1 for i in range(1, len(quantifier_seq))
            if quantifier_seq[i] != quantifier_seq[i - 1]
        )
        return alternations

    # ─── COMPOSITE SCORE ─────────────────────────────

    def _compute_composite_score(
        self,
        d_L: float,
        n_V: int,
        sigma_S: float,
        omega_N: int,
        kappa_Q: int,
    ) -> float:
        """
        Compute weighted composite complexity score.

            C(p) = Σᵢ wᵢ · φ̂ᵢ(p)

        where φ̂ᵢ ∈ [0,1] are min-max normalized metrics.
        """
        normalized = {
            'logical_depth':          self._normalize(d_L, 'logical_depth'),
            'variable_count':         self._normalize(float(n_V), 'variable_count'),
            'symbolic_structure':     sigma_S,  # Already in [0,1]
            'nesting_complexity':     self._normalize(float(omega_N), 'nesting_complexity'),
            'quantifier_alternation': self._normalize(float(kappa_Q), 'quantifier_alternation'),
        }

        composite = sum(
            self.weights[metric] * normalized[metric]
            for metric in self.weights
        )
        return max(0.0, min(1.0, composite))

    def _normalize(self, value: float, metric_name: str) -> float:
        """Min-max normalization to [0, 1]."""
        params = self.norm_params.get(metric_name, {'min': 0, 'max': 1})
        min_val = params['min']
        max_val = params['max']
        if max_val <= min_val:
            return 0.0
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))

    # ─── ROUTING SUGGESTION ──────────────────────────

    def _suggest_route(
        self, composite: float, sigma_S: float, kappa_Q: int
    ) -> str:
        """Preliminary routing suggestion based on thresholds."""
        tau_low = self.thresholds['tau_low']
        tau_high = self.thresholds['tau_high']
        sigma_sym = self.thresholds['sigma_symbolic_threshold']
        sigma_neu = self.thresholds['sigma_neural_threshold']

        if composite < tau_low and sigma_S > sigma_sym:
            return "symbolic"
        elif composite > tau_high or sigma_S < sigma_neu:
            return "neural_cot"
        else:
            return "hybrid"

    # ─── CALIBRATION ─────────────────────────────────

    def _calibrate_normalization(self, prompts: list):
        """
        Two-pass calibration: compute raw metrics on all prompts,
        then set min/max from observed percentiles (5th/95th).
        """
        raw_values = {
            'logical_depth': [],
            'variable_count': [],
            'nesting_complexity': [],
            'quantifier_alternation': [],
        }

        for prompt in prompts:
            raw_values['logical_depth'].append(
                self._compute_logical_depth(prompt)
            )
            raw_values['variable_count'].append(
                float(self._compute_variable_count(prompt))
            )
            raw_values['nesting_complexity'].append(
                float(self._compute_nesting_complexity(prompt))
            )
            raw_values['quantifier_alternation'].append(
                float(self._compute_quantifier_alternation(prompt))
            )

        for metric, values in raw_values.items():
            if values:
                arr = np.array(values)
                self.norm_params[metric] = {
                    'min': float(np.percentile(arr, 5)),
                    'max': float(np.percentile(arr, 95)),
                }
