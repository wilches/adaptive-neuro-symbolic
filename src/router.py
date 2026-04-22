"""
Adaptive Router for Neuro-Symbolic Reasoning
=============================================

Selects the optimal reasoning path (Symbolic, Neural CoT, or Hybrid)
for each mathematical prompt based on structural complexity metrics.

The Router implements a threshold-based policy that can be optimized
via grid search over validation data.
"""

import json
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from src.complexity_analyzer import ComplexityProfile


class RoutePath(Enum):
    SYMBOLIC = "symbolic"
    NEURAL_COT = "neural_cot"
    HYBRID = "hybrid"


@dataclass
class RoutingDecision:
    """Complete record of a routing decision."""
    path: RoutePath
    confidence: float
    reasoning: str
    composite_score: float
    symbolic_structure: float
    quantifier_alternation: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d['path'] = self.path.value
        return d


class Router:
    """
    Threshold-based router that maps ComplexityProfiles to reasoning paths.

    Routing Rules:
        SYMBOLIC:   C(p) < τ_low  AND  σ_S > σ_sym
        NEURAL_CoT: C(p) > τ_high  OR  σ_S < σ_neu
        HYBRID:     otherwise (middle zone)

    Special rules:
        - High quantifier alternation (κ_Q ≥ 2) boosts toward HYBRID
        - Very high nesting (ω_N ≥ 8) boosts toward SYMBOLIC
    """

    def __init__(
        self,
        tau_low: float = 0.30,
        tau_high: float = 0.65,
        sigma_symbolic: float = 0.60,
        sigma_neural: float = 0.30,
    ):
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.sigma_symbolic = sigma_symbolic
        self.sigma_neural = sigma_neural
        self.history: List[RoutingDecision] = []
        self._decision_counts = {p: 0 for p in RoutePath}

    def route(self, profile: ComplexityProfile) -> RoutingDecision:
        """
        Determine the optimal reasoning path for a given complexity profile.
        """
        C = profile.composite_score
        sigma = profile.symbolic_structure
        kappa = profile.quantifier_alternation
        omega = profile.nesting_complexity

        # ══════════════════════════════════════════════
        # FIX: Extract variable count from profile
        # Problems with variables + equations benefit from
        # at least symbolic solving (not just simplification)
        # ══════════════════════════════════════════════
        n_V = profile.variable_count

        path, confidence, reasoning = self._apply_rules(
            C, sigma, kappa, omega, n_V
        )

        decision = RoutingDecision(
            path=path,
            confidence=round(confidence, 4),
            reasoning=reasoning,
            composite_score=C,
            symbolic_structure=sigma,
            quantifier_alternation=kappa,
        )

        self.history.append(decision)
        self._decision_counts[path] += 1
        return decision

    def _apply_rules(
    self, C: float, sigma: float, kappa: int, omega: int, n_V: int = 0
    ) -> Tuple[RoutePath, float, str]:
        """Core routing logic with variable-awareness."""

        # ── Rule 0 (NEW): Variable-aware upgrade ────
        # If there are variables AND high symbolic content,
        # the problem likely needs equation SOLVING.
        # Symbolic engine CAN solve equations, so keep it
        # in symbolic. But if nesting is also high, go hybrid.
        # This handles cases like $\frac{x^2-4}{x-2} = 3$

        # ── Rule 1: SYMBOLIC ────────────────────────
        if C < self.tau_low and sigma > self.sigma_symbolic:
            # NEW: If variables + moderate complexity → upgrade to HYBRID
            if n_V >= 1 and (omega >= 2 or C > self.tau_low * 0.7):
                confidence = 0.65
                reasoning = (
                    f"HYBRID (upgraded from Symbolic): "
                    f"C={C:.3f} is low but n_V={n_V} variables "
                    f"with ω={omega} nesting suggests equation solving"
                )
                return RoutePath.HYBRID, confidence, reasoning

            confidence = self._calc_confidence_symbolic(C, sigma)
            reasoning = (
                f"SYMBOLIC: Low complexity (C={C:.3f} < τ_low={self.tau_low}) "
                f"+ high symbolic content (σ={sigma:.3f} > {self.sigma_symbolic})"
            )
            return RoutePath.SYMBOLIC, confidence, reasoning

        # ── Rule 2: NEURAL CoT ─────────────────────
        if C > self.tau_high or sigma < self.sigma_neural:
            if kappa >= 2:
                confidence = 0.70
                reasoning = (
                    f"HYBRID (upgraded from CoT): complexity C={C:.3f}, "
                    f"but κ_Q={kappa} >= 2 suggests formal structure"
                )
                return RoutePath.HYBRID, confidence, reasoning

            confidence = self._calc_confidence_neural(C, sigma)
            reasoning = (
                f"NEURAL_COT: High complexity (C={C:.3f}) "
                f"or low symbolic content (σ={sigma:.3f} < {self.sigma_neural})"
            )
            return RoutePath.NEURAL_COT, confidence, reasoning

        # ── Rule 3: HYBRID ──────────────────────────
        confidence = self._calc_confidence_hybrid(C, sigma, omega)
        reasoning = (
            f"HYBRID: Medium complexity zone "
            f"(C={C:.3f}, σ={sigma:.3f}, ω={omega})"
        )

        if omega >= 8:
            confidence = min(1.0, confidence + 0.1)
            reasoning += f" [High nesting ω={omega} boosts confidence]"

        return RoutePath.HYBRID, confidence, reasoning

    def _calc_confidence_symbolic(self, C: float, sigma: float) -> float:
        """Higher confidence when C is much lower than threshold AND sigma is high."""
        distance = (self.tau_low - C) / self.tau_low  # How far below threshold
        return min(1.0, 0.5 + 0.3 * distance + 0.2 * sigma)

    def _calc_confidence_neural(self, C: float, sigma: float) -> float:
        """Higher confidence when clearly in neural territory."""
        if sigma < self.sigma_neural:
            return min(1.0, 0.6 + 0.4 * (1 - sigma / self.sigma_neural))
        return min(1.0, 0.5 + 0.5 * (C / self.tau_high))

    def _calc_confidence_hybrid(self, C: float, sigma: float, omega: int) -> float:
        """Hybrid confidence is moderate; boosted by high nesting."""
        base = 0.55
        nesting_bonus = min(0.2, omega * 0.025)
        return min(1.0, base + nesting_bonus)

    # ─── THRESHOLD OPTIMIZATION ──────────────────────

    def optimize_thresholds(
        self,
        profiles: List[ComplexityProfile],
        path_results: Dict[str, List[dict]],
        alpha: float = 0.7,    # Weight for accuracy
        beta: float = 0.3,     # Weight for token efficiency
    ) -> dict:
        """
        Grid search for optimal routing thresholds.

        Args:
            profiles: ComplexityProfile for each problem.
            path_results: Dict with keys 'symbolic', 'neural_cot', 'hybrid',
                          each containing list of dicts with 'correct' (bool)
                          and 'tokens' (int).
            alpha: Weight for accuracy in objective.
            beta: Weight for token savings in objective.

        Returns:
            Dict with optimal thresholds and achieved score.
        """
        best_score = -float('inf')
        best_params = {}

        # Token normalization (max tokens from CoT baseline)
        max_tokens = max(
            (r['tokens'] for r in path_results.get('neural_cot', [{'tokens': 1}])),
            default=1
        )

        for tau_low in np.arange(0.15, 0.50, 0.05):
            for tau_high in np.arange(0.45, 0.85, 0.05):
                if tau_low >= tau_high:
                    continue
                for sigma_sym in np.arange(0.40, 0.80, 0.10):
                    for sigma_neu in np.arange(0.15, 0.45, 0.10):
                        if sigma_neu >= sigma_sym:
                            continue

                        # Temporarily set thresholds
                        self.tau_low = tau_low
                        self.tau_high = tau_high
                        self.sigma_symbolic = sigma_sym
                        self.sigma_neural = sigma_neu

                        # Simulate routing
                        total_correct = 0
                        total_tokens = 0
                        n = len(profiles)

                        for i, profile in enumerate(profiles):
                            decision = self.route(profile)
                            path_key = decision.path.value

                            if i < len(path_results.get(path_key, [])):
                                result = path_results[path_key][i]
                                total_correct += int(result.get('correct', False))
                                total_tokens += result.get('tokens', 0)

                        # Clear history from simulation
                        self.history = []
                        self._decision_counts = {p: 0 for p in RoutePath}

                        accuracy = total_correct / n if n > 0 else 0
                        token_efficiency = 1 - (total_tokens / (n * max_tokens)) if max_tokens > 0 else 0

                        score = alpha * accuracy + beta * token_efficiency

                        if score > best_score:
                            best_score = score
                            best_params = {
                                'tau_low': tau_low,
                                'tau_high': tau_high,
                                'sigma_symbolic': sigma_sym,
                                'sigma_neural': sigma_neu,
                                'score': score,
                                'accuracy': accuracy,
                                'token_efficiency': token_efficiency,
                            }

        # Apply best thresholds
        if best_params:
            self.tau_low = best_params['tau_low']
            self.tau_high = best_params['tau_high']
            self.sigma_symbolic = best_params['sigma_symbolic']
            self.sigma_neural = best_params['sigma_neural']

        return best_params

    # ─── ANALYSIS ────────────────────────────────────

    def get_routing_distribution(self) -> Dict[str, int]:
        """Get counts of routing decisions by path."""
        return {p.value: c for p, c in self._decision_counts.items()}

    def get_confidence_stats(self) -> Dict[str, float]:
        """Get confidence statistics across all decisions."""
        if not self.history:
            return {}
        confidences = [d.confidence for d in self.history]
        return {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences)),
        }

    def reset_history(self):
        """Clear routing history."""
        self.history = []
        self._decision_counts = {p: 0 for p in RoutePath}
