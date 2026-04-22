"""
Adaptive Neuro-Symbolic Reasoning Pipeline
============================================

Orchestrates the full flow:
    Prompt → Complexity Analysis → Routing → Engine Execution → Result

This is the main entry point for running experiments.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path

from src.complexity_analyzer import ComplexityAnalyzer, ComplexityProfile
from src.router import Router, RoutePath, RoutingDecision
from src.engines.cot_engine import CoTEngine, CoTResult
from src.engines.symbolic_engine import SymbolicEngine, SymbolicResult
from src.config import ProviderConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result from processing one problem through the pipeline."""
    problem_id: str
    problem_text: str
    expected_answer: str
    predicted_answer: str
    is_correct: bool

    # Pipeline components
    complexity_profile: dict
    routing_decision: dict
    route_taken: str

    # Performance metrics
    tokens_input: int
    tokens_output: int
    tokens_total: int
    latency_seconds: float
    api_cost_usd: float

    # Execution details
    engine_used: str
    engine_details: dict

    # Hybrid path specifics
    cot_answer: str = ""
    symbolic_verification: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class AdaptivePipeline:
    """
    Main pipeline for adaptive neuro-symbolic mathematical reasoning.

    Usage:
        pipeline = AdaptivePipeline(cot_config=my_provider_config)
        result = pipeline.solve("Find x: 2x + 3 = 7", expected="2")
        print(result.is_correct, result.route_taken, result.tokens_total)
    """

    def __init__(
        self,
        cot_config: ProviderConfig,
        analyzer_kwargs: Optional[dict] = None,
        router_kwargs: Optional[dict] = None,
    ):
        # ── Initialize components ────────────────────
        self.analyzer = ComplexityAnalyzer(**(analyzer_kwargs or {}))
        self.router = Router(**(router_kwargs or {}))
        self.cot_engine = CoTEngine(cot_config)
        self.symbolic_engine = SymbolicEngine()

        self.cot_config = cot_config
        self.results: List[PipelineResult] = []

        logger.info(
            f"Pipeline initialized with CoT provider: "
            f"{cot_config.name}/{cot_config.model}"
        )

    def solve(
        self,
        problem: str,
        expected_answer: str = "",
        problem_id: str = "",
        prompt_type: str = "general",
        force_route: Optional[str] = None,
    ) -> PipelineResult:
        """
        Process a single mathematical problem through the adaptive pipeline.

        Args:
            problem: The mathematical problem text.
            expected_answer: Ground truth answer for evaluation.
            problem_id: Identifier for this problem.
            prompt_type: 'math', 'gsm8k', or 'general'.
            force_route: If set, override router ('symbolic', 'neural_cot', 'hybrid').

        Returns:
            PipelineResult with complete execution record.
        """
        start_time = time.time()

        # ── Step 1: Analyze complexity ───────────────
        profile = self.analyzer.analyze(problem)

        # ── Step 2: Route ────────────────────────────
        if force_route:
            route = RoutePath(force_route)
            decision = RoutingDecision(
                path=route,
                confidence=1.0,
                reasoning=f"Forced route: {force_route}",
                composite_score=profile.composite_score,
                symbolic_structure=profile.symbolic_structure,
                quantifier_alternation=profile.quantifier_alternation,
            )
        else:
            decision = self.router.route(profile)
            route = decision.path

        # ── Step 3: Execute selected path ────────────
        if route == RoutePath.SYMBOLIC:
            result = self._execute_symbolic(problem, prompt_type)
        elif route == RoutePath.NEURAL_COT:
            result = self._execute_cot(problem, prompt_type)
        else:  # HYBRID
            result = self._execute_hybrid(problem, prompt_type)

        # ── Step 4: Evaluate correctness ─────────────
        predicted = result['answer']
        is_correct = self._check_answer(predicted, expected_answer)

        total_latency = round(time.time() - start_time, 3)

        # ── Build pipeline result ────────────────────
        pipeline_result = PipelineResult(
            problem_id=problem_id,
            problem_text=problem,
            expected_answer=expected_answer,
            predicted_answer=predicted,
            is_correct=is_correct,
            complexity_profile=profile.to_dict(),
            routing_decision=decision.to_dict(),
            route_taken=route.value,
            tokens_input=result.get('tokens_input', 0),
            tokens_output=result.get('tokens_output', 0),
            tokens_total=result.get('tokens_total', 0),
            latency_seconds=total_latency,
            api_cost_usd=result.get('cost', 0.0),
            engine_used=result.get('engine', ''),
            engine_details=result.get('details', {}),
            cot_answer=result.get('cot_answer', ''),
            symbolic_verification=result.get('symbolic_verification', ''),
        )

        self.results.append(pipeline_result)
        return pipeline_result

    def batch_solve(
        self,
        problems: List[Dict],
        prompt_type: str = "general",
        force_route: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> List[PipelineResult]:
        """
        Process a batch of problems.

        Args:
            problems: List of dicts with keys 'problem', 'answer', 'id'.
            prompt_type: 'math', 'gsm8k', or 'general'.
            force_route: If set, override router for all problems.
            save_path: If set, save results to this JSON file.

        Returns:
            List of PipelineResults.
        """
        from tqdm import tqdm

        results = []
        for item in tqdm(problems, desc=f"Solving ({force_route or 'adaptive'})"):
            result = self.solve(
                problem=item.get('problem', ''),
                expected_answer=str(item.get('answer', '')),
                problem_id=str(item.get('id', '')),
                prompt_type=prompt_type,
                force_route=force_route,
            )
            results.append(result)

            # Rate limiting for API calls
            if result.route_taken != 'symbolic':
                time.sleep(0.5)  # Avoid hitting rate limits

        if save_path:
            self._save_results(results, save_path)

        return results

    # ─── EXECUTION PATHS ─────────────────────────────

    def _execute_symbolic(self, problem: str, prompt_type: str = "math") -> dict:
        """
        Execute the symbolic path WITH FALLBACK to CoT.
        If the symbolic engine cannot solve the problem,
        automatically fall back to neural CoT.
        """
        sym_result = self.symbolic_engine.solve(problem)

        # ══════════════════════════════════════════════
        # FIX: If symbolic fails, fall back to CoT.
        # This prevents empty answers from tanking accuracy.
        # ══════════════════════════════════════════════
        if not sym_result.success or not sym_result.answer:
            # Symbolic failed → fall back to CoT
            cot_result = self.cot_engine.solve(problem, prompt_type=prompt_type)
            cost = self.cot_engine.estimate_cost(
                cot_result.tokens_input, cot_result.tokens_output
            )
            return {
                'answer': cot_result.answer,
                'tokens_input': cot_result.tokens_input,
                'tokens_output': cot_result.tokens_output,
                'tokens_total': cot_result.tokens_total,
                'cost': cost,
                'engine': f'symbolic_fallback_to_cot/{self.cot_config.name}',
                'details': {
                    'symbolic_attempted': True,
                    'symbolic_failed': True,
                    'symbolic_steps': sym_result.steps,
                    'fallback_provider': cot_result.provider,
                    'fallback_model': cot_result.model,
                    'fallback_success': cot_result.success,
                    'fallback_latency': cot_result.latency_seconds,
                },
            }

        # Symbolic succeeded → use it (0 tokens!)
        return {
            'answer': sym_result.answer,
            'tokens_input': 0,
            'tokens_output': 0,
            'tokens_total': 0,
            'cost': 0.0,
            'engine': f'symbolic/{sym_result.method_used}',
            'details': {
                'method': sym_result.method_used,
                'steps': sym_result.steps,
                'success': sym_result.success,
                'latency': sym_result.latency_seconds,
                'symbolic_attempted': True,
                'symbolic_failed': False,
            },
        }

    def _execute_cot(self, problem: str, prompt_type: str) -> dict:
        """Execute the pure Neural CoT path."""
        cot_result = self.cot_engine.solve(problem, prompt_type=prompt_type)

        cost = self.cot_engine.estimate_cost(
            cot_result.tokens_input, cot_result.tokens_output
        )

        return {
            'answer': cot_result.answer,
            'tokens_input': cot_result.tokens_input,
            'tokens_output': cot_result.tokens_output,
            'tokens_total': cot_result.tokens_total,
            'cost': cost,
            'engine': f'cot/{self.cot_config.name}',
            'details': {
                'provider': cot_result.provider,
                'model': cot_result.model,
                'success': cot_result.success,
                'latency': cot_result.latency_seconds,
                'full_response_length': len(cot_result.full_response),
            },
        }

    def _execute_hybrid(self, problem: str, prompt_type: str) -> dict:
        """
        Execute the hybrid path:
            1. CoT generates step-by-step solution
            2. Symbolic engine verifies the answer
            3. If verification fails, trust CoT (but flag it)
        """
        # ── Phase 1: CoT reasoning ──────────────────
        cot_result = self.cot_engine.solve(problem, prompt_type=prompt_type)
        cot_answer = cot_result.answer

        # ── Phase 2: Symbolic verification ───────────
        verification_status = "not_attempted"
        verified = False

        if cot_answer:
            # Try to verify CoT's answer symbolically
            try:
                is_verified, explanation = self.symbolic_engine.verify(
                    problem, cot_answer
                )
                verified = is_verified
                verification_status = explanation
            except Exception as e:
                verification_status = f"verification_error: {e}"

            # If CoT fails but symbolic can solve directly, use symbolic
            if not cot_result.success:
                sym_result = self.symbolic_engine.solve(problem)
                if sym_result.success:
                    cot_answer = sym_result.answer
                    verification_status = "symbolic_fallback"

        cost = self.cot_engine.estimate_cost(
            cot_result.tokens_input, cot_result.tokens_output
        )

        return {
            'answer': cot_answer,
            'cot_answer': cot_answer,
            'symbolic_verification': verification_status,
            'tokens_input': cot_result.tokens_input,
            'tokens_output': cot_result.tokens_output,
            'tokens_total': cot_result.tokens_total,
            'cost': cost,
            'engine': f'hybrid/cot+symbolic',
            'details': {
                'cot_provider': cot_result.provider,
                'cot_model': cot_result.model,
                'cot_success': cot_result.success,
                'cot_latency': cot_result.latency_seconds,
                'verified': verified,
                'verification_status': verification_status,
            },
        }

    # ─── ANSWER CHECKING ────────────────────────────

    def _check_answer(self, predicted: str, expected: str) -> bool:
        """Check if predicted answer matches expected answer."""
        if not predicted or not expected:
            return False

        predicted = predicted.strip()
        expected = expected.strip()

        # ── Exact match ─────────────────────────────
        if predicted == expected:
            return True

        # ── Clean both ──────────────────────────────
        pred_clean = self._clean_answer(predicted)
        exp_clean = self._clean_answer(expected)

        if pred_clean == exp_clean:
            return True

        # ── List/set handling ───────────────────────
        pred_values = self._extract_values_from_answer(pred_clean)
        exp_values = self._extract_values_from_answer(exp_clean)

        # Both are sets: check set equality
        if len(pred_values) > 1 and len(exp_values) > 1:
            try:
                pred_floats = sorted([float(v) for v in pred_values])
                exp_floats = sorted([float(v) for v in exp_values])
                if len(pred_floats) == len(exp_floats):
                    if all(abs(a - b) < 1e-4 for a, b in zip(pred_floats, exp_floats)):
                        return True
            except (ValueError, TypeError):
                if set(pred_values) == set(exp_values):
                    return True

        # Expected is single value, predicted is list: check membership
        if len(exp_values) == 1 and len(pred_values) >= 1:
            for pv in pred_values:
                try:
                    if abs(float(pv) - float(exp_values[0])) < 1e-4:
                        return True
                except (ValueError, TypeError):
                    if str(pv).strip() == str(exp_values[0]).strip():
                        return True

        # Predicted is single value, expected is list: check membership
        if len(pred_values) == 1 and len(exp_values) >= 1:
            for ev in exp_values:
                try:
                    if abs(float(pred_values[0]) - float(ev)) < 1e-4:
                        return True
                except (ValueError, TypeError):
                    pass

        # ── Numerical comparison ─────────────────────
        try:
            pred_num = float(self._to_number(pred_clean))
            exp_num = float(self._to_number(exp_clean))
            if abs(pred_num - exp_num) < 1e-4:
                return True
            if exp_num != 0 and abs((pred_num - exp_num) / exp_num) < 1e-4:
                return True
        except (ValueError, TypeError, ZeroDivisionError):
            pass

        # ── Fraction comparison ──────────────────────
        try:
            from fractions import Fraction
            pred_frac = Fraction(pred_clean).limit_denominator(10000)
            exp_frac = Fraction(exp_clean).limit_denominator(10000)
            if pred_frac == exp_frac:
                return True
        except (ValueError, ZeroDivisionError):
            pass

        return False


    #------ Extract value from answer
    def _extract_values_from_answer(self, answer: str) -> list:
        """
        Extract individual values from an answer that might be a list.
        "[-2, 2]" → ["-2", "2"]
        "5"       → ["5"]
        "2/3"     → ["2/3"]
        """
        s = answer.strip()
        s = s.strip('[]{}()')
        if ',' in s:
            parts = [p.strip() for p in s.split(',')]
            return [p for p in parts if p]
        return [s] if s else []

    def _clean_answer(self, answer: str) -> str:
        """Clean answer string for comparison."""
        s = answer.strip()
        s = s.replace('\\boxed{', '').replace('}', '')
        s = s.replace('$', '').replace('\\', '')
        s = s.rstrip('.')
        s = s.replace(' ', '')

        # Remove commas ONLY in pure numbers (e.g., "1,000" → "1000")
        # but NOT in lists like "[-5,5]"
        if not ('[' in s or '(' in s or '{' in s):
            s = s.replace(',', '')

        return s

    def _to_number(self, s: str) -> float:
        """Convert string to number, handling fractions."""
        s = s.strip()
        if '/' in s:
            parts = s.split('/')
            return float(parts[0]) / float(parts[1])
        return float(s)

    # ─── RESULTS MANAGEMENT ──────────────────────────

    def get_summary(self) -> dict:
        """Get aggregate statistics from all processed problems."""
        if not self.results:
            return {}

        total = len(self.results)
        correct = sum(1 for r in self.results if r.is_correct)
        total_tokens = sum(r.tokens_total for r in self.results)
        total_cost = sum(r.api_cost_usd for r in self.results)
        total_latency = sum(r.latency_seconds for r in self.results)

        route_counts = {}
        route_accuracy = {}
        route_tokens = {}

        for r in self.results:
            rt = r.route_taken
            route_counts[rt] = route_counts.get(rt, 0) + 1
            if rt not in route_accuracy:
                route_accuracy[rt] = {'correct': 0, 'total': 0}
                route_tokens[rt] = []
            route_accuracy[rt]['total'] += 1
            if r.is_correct:
                route_accuracy[rt]['correct'] += 1
            route_tokens[rt].append(r.tokens_total)

        for rt in route_accuracy:
            ra = route_accuracy[rt]
            ra['accuracy'] = round(ra['correct'] / ra['total'], 4) if ra['total'] > 0 else 0

        import numpy as np
        for rt in route_tokens:
            tokens = route_tokens[rt]
            route_tokens[rt] = {
                'mean': round(float(np.mean(tokens)), 1),
                'median': round(float(np.median(tokens)), 1),
                'total': sum(tokens),
            }

        return {
            'total_problems': total,
            'correct': correct,
            'accuracy': round(correct / total, 4),
            'total_tokens': total_tokens,
            'avg_tokens': round(total_tokens / total, 1),
            'total_cost_usd': round(total_cost, 6),
            'avg_latency_seconds': round(total_latency / total, 3),
            'route_distribution': route_counts,
            'route_accuracy': route_accuracy,
            'route_tokens': route_tokens,
        }

    def _save_results(self, results: List[PipelineResult], path: str):
        """Save results to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in results]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Results saved to {path}")

    def reset(self):
        """Clear all stored results."""
        self.results = []
        self.router.reset_history()
