"""
Diagnose answer matching issues.
Finds cases where the answer is mathematically correct
but our checker says FALSE.
"""

import json
import sys
import re
from pathlib import Path
from fractions import Fraction

sys.path.insert(0, str(Path(__file__).parent.parent))


def latex_to_comparable(s: str) -> str:
    """
    Convert LaTeX math notation to plain comparable format.
    This is what _clean_answer SHOULD be doing.
    """
    if not s:
        return s

    result = s.strip()

    # Remove \boxed{}
    boxed = re.findall(r'\\boxed\{(.+)\}', result)
    if boxed:
        result = boxed[-1]

    # \dfrac, \tfrac, \frac → division
    while True:
        match = re.search(
            r'\$$dt]?frac\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
            r'\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
            result
        )
        if match:
            num, den = match.group(1), match.group(2)
            # Recursively clean num and den
            num_clean = latex_to_comparable(num)
            den_clean = latex_to_comparable(den)
            result = result[:match.start()] + f'({num_clean})/({den_clean})' + result[match.end():]
        else:
            break

    # \sqrt[n]{x} → (x)**(1/n)
    result = re.sub(r'\\sqrt\[([^$$]+)\]\{([^{}]+)\}', r'(\2)**(1/\1)', result)
    # \sqrt{x} → sqrt(x)
    result = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', result)

    # \pi → pi
    result = result.replace('\\pi', 'pi')
    # \infty → oo
    result = result.replace('\\infty', 'oo')

    # Powers: ^{n} → **(n)
    result = re.sub(r'\^\{([^{}]+)\}', r'**(\1)', result)
    result = re.sub(r'\^(\d)', r'**\1', result)

    # \cdot, \times → *
    result = result.replace('\\cdot', '*').replace('\\times', '*')

    # \left, \right → nothing
    result = result.replace('\\left', '').replace('\\right', '')

    # Remove remaining LaTeX commands
    result = re.sub(r'\$$a-zA-Z]+', '', result)

    # Clean braces
    result = result.replace('{', '').replace('}', '')

    # Clean spaces
    result = result.replace(' ', '')

    # Remove trailing dots
    result = result.rstrip('.')

    # Remove $
    result = result.replace('$', '')

    return result


def try_numeric_equal(a: str, b: str, tol: float = 1e-4) -> bool:
    """Try to compare two strings as numbers."""
    def to_float(s):
        s = s.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            pass
        if '/' in s:
            try:
                parts = s.split('/')
                return float(parts[0].strip('() ')) / float(parts[1].strip('() '))
            except:
                pass
        try:
            import sympy
            val = sympy.sympify(s)
            return float(val.evalf())
        except:
            pass
        return None

    fa, fb = to_float(a), to_float(b)
    if fa is not None and fb is not None:
        if abs(fa - fb) < tol:
            return True
        if fb != 0 and abs((fa - fb) / fb) < tol:
            return True
    return False


def improved_check(predicted: str, expected: str) -> dict:
    """
    Enhanced answer checking with full diagnosis.
    Returns dict with match result and explanation.
    """
    if not predicted or not expected:
        return {'match': False, 'reason': 'empty'}

    # Level 0: Exact match
    if predicted.strip() == expected.strip():
        return {'match': True, 'reason': 'exact'}

    # Level 1: LaTeX-aware cleaning
    pred_clean = latex_to_comparable(predicted)
    exp_clean = latex_to_comparable(expected)

    if pred_clean == exp_clean:
        return {'match': True, 'reason': 'latex_cleaned'}

    # Level 2: Numeric comparison
    if try_numeric_equal(pred_clean, exp_clean):
        return {'match': True, 'reason': 'numeric'}

    # Level 3: SymPy symbolic comparison
    try:
        import sympy
        pred_sym = sympy.sympify(pred_clean)
        exp_sym = sympy.sympify(exp_clean)
        if sympy.simplify(pred_sym - exp_sym) == 0:
            return {'match': True, 'reason': 'sympy_equal'}
    except:
        pass

    # Level 4: Numerical evaluation via SymPy
    try:
        import sympy
        pred_val = complex(sympy.sympify(pred_clean).evalf())
        exp_val = complex(sympy.sympify(exp_clean).evalf())
        if abs(pred_val - exp_val) < 1e-4:
            return {'match': True, 'reason': 'sympy_numeric'}
    except:
        pass

    return {'match': False, 'reason': 'no_match',
            'pred_clean': pred_clean, 'exp_clean': exp_clean}


def main():
    results_dir = Path("results/20260224_210542")

    # Load all results
    all_results = {}
    for name in ['results_symbolic', 'results_cot_math', 'results_cot_gsm8k',
                  'results_hybrid', 'results_adaptive_math', 'results_adaptive_gsm8k']:
        path = results_dir / f"{name}.json"
        if path.exists():
            with open(path) as f:
                all_results[name] = json.load(f)

    print("=" * 70)
    print("ANSWER MATCHING DIAGNOSIS")
    print("=" * 70)

    for result_name, results in all_results.items():
        print(f"\n{'='*60}")
        print(f"Analyzing: {result_name} ({len(results)} problems)")
        print(f"{'='*60}")

        original_correct = 0
        improved_correct = 0
        flipped = []  # Cases where improved says TRUE but original said FALSE

        for r in results:
            predicted = r.get('predicted_answer', '')
            expected = r.get('expected_answer', '')
            original_result = r.get('is_correct', False)

            if original_result:
                original_correct += 1

            # Run improved check
            check = improved_check(predicted, expected)

            if check['match']:
                improved_correct += 1

            if check['match'] and not original_result:
                flipped.append({
                    'id': r.get('problem_id', ''),
                    'predicted': predicted,
                    'expected': expected,
                    'reason': check['reason'],
                    'route': r.get('route_taken', ''),
                    'engine': r.get('engine_used', ''),
                })

        n = len(results)
        print(f"  Original accuracy:  {original_correct}/{n} = "
              f"{original_correct/n*100:.2f}%")
        print(f"  Improved accuracy:  {improved_correct}/{n} = "
              f"{improved_correct/n*100:.2f}%")
        print(f"  Flipped (FALSE→TRUE): {len(flipped)}")

        if flipped:
            print(f"\n  --- Flipped cases (first 10) ---")
            for f in flipped[:10]:
                print(f"    [{f['id']}] route={f['route']}")
                print(f"      predicted: {f['predicted'][:60]}")
                print(f"      expected:  {f['expected'][:60]}")
                print(f"      reason:    {f['reason']}")
                print()

    # ── Summary ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("IMPACT SUMMARY")
    print("=" * 70)

    for result_name in all_results:
        results = all_results[result_name]
        n = len(results)
        orig = sum(1 for r in results if r.get('is_correct', False))

        improved = 0
        for r in results:
            check = improved_check(
                r.get('predicted_answer', ''),
                r.get('expected_answer', '')
            )
            if check['match']:
                improved += 1

        delta = improved - orig
        print(f"  {result_name:<30} "
              f"original={orig/n*100:5.2f}%  "
              f"improved={improved/n*100:5.2f}%  "
              f"delta=+{delta} ({delta/n*100:.2f}pp)")


if __name__ == "__main__":
    main()
