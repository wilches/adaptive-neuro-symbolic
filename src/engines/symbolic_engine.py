"""
Symbolic Reasoning Engine
==========================

Interfaces with SymPy and Z3 for formal mathematical solving.
Runs entirely LOCAL — zero API cost, zero tokens consumed.

Handles:
    - Algebraic equations and systems
    - Simplification and evaluation
    - Arithmetic word problem expressions
    - Basic logical formulas (via Z3)
"""

import re
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SymbolicResult:
    """Result from the symbolic engine."""
    answer: str
    method_used: str            # 'sympy_solve', 'sympy_simplify', 'z3', 'eval'
    steps: List[str]            # Intermediate steps taken
    success: bool
    latency_seconds: float
    tokens_consumed: int = 0    # Always 0 — local computation
    error_message: str = ""


class SymbolicEngine:
    """
    Local symbolic solver using SymPy and Z3.

    Strategy:
        1. Try to parse the math expression from the problem text
        2. Attempt solving with SymPy (equations, simplification)
        3. Fall back to Z3 for logical/constraint problems
        4. Fall back to safe eval for pure arithmetic

    All computation is LOCAL. No API calls, no tokens, no cost.
    """

    def __init__(self):
        self._check_dependencies()

    def _check_dependencies(self):
        """Verify that SymPy and Z3 are available."""
        try:
            import sympy
            self._sympy = sympy
            logger.info(f"SymPy {sympy.__version__} loaded")
        except ImportError:
            raise ImportError("Install sympy: pip install sympy")

        try:
            import z3
            self._z3 = z3
            logger.info(f"Z3 loaded")
        except ImportError:
            logger.warning(
                "Z3 not found. Install with: pip install z3-solver. "
                "Symbolic engine will work without Z3 but with reduced capability."
            )
            self._z3 = None

    def solve(self, problem: str) -> SymbolicResult:
        """
        Attempt to solve a mathematical problem symbolically.

        Tries multiple strategies in order of specificity.

        Args:
            problem: Mathematical problem text (may contain LaTeX).

        Returns:
            SymbolicResult with answer and method details.
        """
        start_time = time.time()
        steps = []

        # ── Strategy 1: Extract and solve equations ──
        result = self._try_equation_solving(problem, steps)
        if result is not None:
            elapsed = round(time.time() - start_time, 4)
            return SymbolicResult(
                answer=str(result),
                method_used="sympy_solve",
                steps=steps,
                success=True,
                latency_seconds=elapsed,
            )

        # ── Strategy 2: Extract and simplify expression ──
        result = self._try_simplification(problem, steps)
        if result is not None:
            elapsed = round(time.time() - start_time, 4)
            return SymbolicResult(
                answer=str(result),
                method_used="sympy_simplify",
                steps=steps,
                success=True,
                latency_seconds=elapsed,
            )

        # ── Strategy 3: Direct arithmetic evaluation ──
        result = self._try_arithmetic_eval(problem, steps)
        if result is not None:
            elapsed = round(time.time() - start_time, 4)
            return SymbolicResult(
                answer=str(result),
                method_used="arithmetic_eval",
                steps=steps,
                success=True,
                latency_seconds=elapsed,
            )

        # ── Strategy 4: Z3 constraint solving ───────
        if self._z3 is not None:
            result = self._try_z3_solve(problem, steps)
            if result is not None:
                elapsed = round(time.time() - start_time, 4)
                return SymbolicResult(
                    answer=str(result),
                    method_used="z3",
                    steps=steps,
                    success=True,
                    latency_seconds=elapsed,
                )

        # ── All strategies failed ────────────────────
        elapsed = round(time.time() - start_time, 4)
        return SymbolicResult(
            answer="",
            method_used="none",
            steps=steps,
            success=False,
            latency_seconds=elapsed,
            error_message="Could not parse or solve symbolically",
        )

    def verify(
        self,
        expression: str,
        claimed_answer: str,
    ) -> Tuple[bool, str]:
        """
        Verify whether a claimed answer is correct for an expression.

        Used in HYBRID path to verify CoT results.

        Args:
            expression: The mathematical expression/equation.
            claimed_answer: The answer to verify.

        Returns:
            (is_correct, explanation)
        """
        try:
            sympy = self._sympy

            # Try to parse both expression and answer
            parsed_expr = self._parse_math_expression(expression)
            if parsed_expr is None:
                return False, "Could not parse expression"

            # If it's an equation, check if the answer satisfies it
            if isinstance(parsed_expr, sympy.Eq):
                # Substitute answer into equation
                variables = list(parsed_expr.free_symbols)
                if len(variables) == 1:
                    var = variables[0]
                    try:
                        answer_val = sympy.sympify(claimed_answer)
                        left = parsed_expr.lhs.subs(var, answer_val)
                        right = parsed_expr.rhs.subs(var, answer_val)
                        result = sympy.simplify(left - right)
                        if result == 0:
                            return True, f"Verified: substituting {var}={claimed_answer} satisfies equation"
                        else:
                            return False, f"Failed: {var}={claimed_answer} gives residual {result}"
                    except Exception as e:
                        return False, f"Verification error: {e}"

            # If it's a numerical expression, compute and compare
            try:
                computed = float(sympy.N(parsed_expr))
                claimed = float(claimed_answer)
                if abs(computed - claimed) < 1e-6:
                    return True, f"Verified: computed {computed} matches claimed {claimed}"
                else:
                    return False, f"Mismatch: computed {computed} vs claimed {claimed}"
            except (ValueError, TypeError):
                pass

            return False, "Could not verify"

        except Exception as e:
            return False, f"Verification error: {e}"

    # ─── SOLVING STRATEGIES ──────────────────────────

    def _try_equation_solving(
        self, problem: str, steps: List[str]
    ) -> Optional[Any]:
        """Try to extract and solve equations from the problem."""
        sympy = self._sympy
        steps.append("Attempting equation extraction and solving")

        # ── Extract LaTeX equations ──────────────────
        equation_patterns = [
            # Pattern: ... = ...
            r'\$([^$]*=[^$]*)\$',
            r'\\\(([^)]*=[^)]*)\\\)',
            r'([a-zA-Z0-9\s\+\-\*/\^()]+=[a-zA-Z0-9\s\+\-\*/\^()]+)',
        ]

        for pattern in equation_patterns:
            matches = re.findall(pattern, problem)
            for match in matches:
                parsed = self._parse_equation(match)
                if parsed is not None:
                    steps.append(f"Found equation: {match}")
                    try:
                        variables = list(parsed.free_symbols)
                        if len(variables) == 1:
                            solutions = sympy.solve(parsed, variables[0])
                            if solutions:
                                steps.append(f"Solved: {variables[0]} = {solutions}")
                                return solutions[0] if len(solutions) == 1 else solutions
                        elif len(variables) > 1:
                            steps.append(f"Multiple variables found: {variables}")
                            # Can't solve without more equations
                    except Exception as e:
                        steps.append(f"Solve failed: {e}")

        return None

    def _try_simplification(
        self, problem: str, steps: List[str]
    ) -> Optional[Any]:
        """Try to extract and simplify a mathematical expression."""
        sympy = self._sympy
        steps.append("Attempting expression simplification")

        # Extract math expressions
        math_patterns = [
            r'\$([^$]+)\$',
            r'\\\((.+?)\\\)',
        ]

        for pattern in math_patterns:
            matches = re.findall(pattern, problem)
            for match in matches:
                if '=' in match:
                    continue  # Skip equations, handled above

                parsed = self._parse_math_expression(match)
                if parsed is not None:
                    steps.append(f"Simplifying: {match}")
                    try:
                        simplified = sympy.simplify(parsed)
                        evaluated = sympy.N(simplified)
                        steps.append(f"Result: {evaluated}")
                        return evaluated
                    except Exception as e:
                        steps.append(f"Simplification failed: {e}")

        return None

    def _try_arithmetic_eval(
        self, problem: str, steps: List[str]
    ) -> Optional[Any]:
        """Try to evaluate pure arithmetic expressions."""
        steps.append("Attempting arithmetic evaluation")

        # Look for computable expressions
        # Extract things like "48 + 24", "48 / 2", etc.
        arithmetic_patterns = [
            r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)',
        ]

        results = []
        for pattern in arithmetic_patterns:
            for match in re.finditer(pattern, problem):
                expr_str = match.group(0)
                try:
                    # Safe evaluation — only arithmetic
                    result = self._safe_eval(expr_str)
                    if result is not None:
                        results.append(result)
                        steps.append(f"Evaluated: {expr_str} = {result}")
                except Exception:
                    pass

        # If we found exactly one result, return it
        if len(results) == 1:
            return results[0]

        # For word problems, try to extract ALL numbers and guess operation
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', problem)
        if numbers:
            steps.append(f"Found numbers: {numbers}")
            # Don't guess — if we can't determine the operation,
            # this should go to CoT anyway
            pass

        return None

    def _try_z3_solve(
        self, problem: str, steps: List[str]
    ) -> Optional[Any]:
        """Try to solve using Z3 SMT solver."""
        if self._z3 is None:
            return None

        z3 = self._z3
        steps.append("Attempting Z3 constraint solving")

        # ── Basic constraint extraction ──────────────
        # This handles simple cases. Complex translation
        # is delegated to the HYBRID path (CoT translates, Z3 verifies)

        # Look for inequality/equation patterns with numbers
        # Example: "x + 3 = 7", "2x < 10"
        simple_patterns = [
            r'([a-z])\s*([+\-])\s*(\d+)\s*([=<>]|<=|>=)\s*(\d+)',
            r'(\d+)\s*([a-z])\s*([=<>]|<=|>=)\s*(\d+)',
        ]

        for pattern in simple_patterns:
            match = re.search(pattern, problem)
            if match:
                try:
                    groups = match.groups()
                    steps.append(f"Z3 pattern found: {match.group(0)}")

                    x = z3.Int('x')
                    solver = z3.Solver()

                    # Build constraint from pattern
                    # (simplified — real implementation would be more robust)
                    expr_str = match.group(0)
                    # This is a placeholder — real Z3 integration
                    # is better done in the hybrid path
                    steps.append("Z3: complex translation needed, deferring")
                    return None
                except Exception as e:
                    steps.append(f"Z3 error: {e}")

        return None

    # ─── PARSING HELPERS ─────────────────────────────

    def _parse_math_expression(self, expr_str: str) -> Optional[Any]:
        """Parse a LaTeX or plain math expression into SymPy."""
        sympy = self._sympy

        # Clean the expression
        cleaned = expr_str.strip()
        cleaned = cleaned.replace('\\left', '').replace('\\right', '')
        cleaned = cleaned.replace('\\,', ' ').replace('\\;', ' ')

        # ── Try SymPy's LaTeX parser ────────────────
        try:
            from sympy.parsing.latex import parse_latex
            result = parse_latex(cleaned)
            return result
        except Exception:
            pass

        # ── Try converting LaTeX to SymPy string ────
        converted = self._latex_to_sympy_string(cleaned)
        if converted:
            try:
                result = sympy.sympify(converted)
                return result
            except Exception:
                pass

        # ── Try direct sympify ──────────────────────
        try:
            result = sympy.sympify(cleaned)
            return result
        except Exception:
            pass

        return None

    def _parse_equation(self, eq_str: str) -> Optional[Any]:
        """Parse an equation string into a SymPy Eq object."""
        sympy = self._sympy

        if '=' not in eq_str:
            return None

        parts = eq_str.split('=', 1)
        if len(parts) != 2:
            return None

        left = self._parse_math_expression(parts[0].strip())
        right = self._parse_math_expression(parts[1].strip())

        if left is not None and right is not None:
            return sympy.Eq(left, right)

        return None

    def _latex_to_sympy_string(self, latex: str) -> Optional[str]:
        """Convert common LaTeX patterns to SymPy-parseable strings."""
        s = latex

        # ── Fractions ────────────────────────────────
        # \frac{a}{b} → (a)/(b)
        while '\\frac' in s:
            match = re.search(
                r'\\frac\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
                r'\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
                s
            )
            if match:
                num, den = match.group(1), match.group(2)
                s = s[:match.start()] + f'(({num})/({den}))' + s[match.end():]
            else:
                break

        # ── Square roots ─────────────────────────────
        s = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', s)
        s = re.sub(r'\\sqrt$$(\d+)$$\{([^{}]+)\}', r'(\2)**(1/\1)', s)

        # ── Powers ───────────────────────────────────
        s = re.sub(r'\^{([^{}]+)}', r'**(\1)', s)
        s = re.sub(r'\^(\d)', r'**\1', s)

        # ── Common functions ─────────────────────────
        for func in ['sin', 'cos', 'tan', 'log', 'ln', 'exp']:
            s = s.replace(f'\\{func}', func)

        # ── Multiplication ───────────────────────────
        s = s.replace('\\cdot', '*').replace('\\times', '*')

        # ── Clean up ─────────────────────────────────
        s = s.replace('\\', '').replace('{', '(').replace('}', ')')
        s = re.sub(r'\s+', ' ', s).strip()

        return s if s else None

    def _safe_eval(self, expr_str: str) -> Optional[float]:
        """Safely evaluate a pure arithmetic expression."""
        # Only allow: digits, +, -, *, /, ., (, ), spaces
        if not re.match(r'^[\d\s\+\-\*/\.\(\)]+$', expr_str):
            return None
        try:
            result = eval(expr_str, {"__builtins__": {}}, {})
            if isinstance(result, (int, float)):
                return result
        except Exception:
            pass
        return None
