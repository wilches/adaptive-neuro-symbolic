"""
Results analyzer and figure generator for the paper.
Generates all tables and plots needed for the 3-6 page Springer paper.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['figure.dpi'] = 150

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(results_dir: str) -> Dict:
    """Load all result files from a results directory."""
    rdir = Path(results_dir)
    data = {}

    for json_file in rdir.glob("*.json"):
        with open(json_file) as f:
            data[json_file.stem] = json.load(f)

    return data


def generate_main_comparison_table(summary: dict, output_path: str = None):
    """
    Generate Table 1: Main results comparison.
    This is the most important table in the paper.
    """
    results = summary.get('results', summary)

    rows = []
    for method, data in results.items():
        name = method.replace('_', ' ').title()
        rows.append({
            'Method': name,
            'Accuracy (%)': round(data.get('accuracy', 0) * 100, 2),
            'Total Tokens': data.get('total_tokens', 0),
            'Avg Tokens': data.get('avg_tokens', 0),
            'Cost (USD)': round(data.get('total_cost_usd', 0), 6),
            'Avg Latency (s)': data.get('avg_latency_seconds', 0),
        })

    df = pd.DataFrame(rows)
    print("\n=== TABLE 1: Main Results ===")
    print(df.to_string(index=False))

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    # Also generate LaTeX version for the paper
    latex = df.to_latex(index=False, float_format="%.2f")
    print("\nLaTeX version:")
    print(latex)

    return df


def generate_routing_analysis(adaptive_results: list, output_dir: str = None):
    """
    Generate routing distribution analysis and figures.
    """
    if not adaptive_results:
        print("No adaptive results to analyze")
        return

    df = pd.DataFrame([r if isinstance(r, dict) else r for r in adaptive_results])

    # ── Routing distribution ─────────────────────────
    route_counts = df['route_taken'].value_counts()
    print("\n=== Routing Distribution ===")
    print(route_counts)

    # ── Accuracy by route ────────────────────────────
    print("\n=== Accuracy by Route ===")
    route_acc = df.groupby('route_taken')['is_correct'].mean()
    print(route_acc)

    # ── Tokens by route ──────────────────────────────
    print("\n=== Avg Tokens by Route ===")
    route_tokens = df.groupby('route_taken')['tokens_total'].mean()
    print(route_tokens)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # ── Figure 1: Routing Distribution Pie Chart ──
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        colors = {
            'symbolic': '#2ecc71',
            'neural_cot': '#3498db',
            'hybrid': '#e74c3c',
        }

        # Pie chart
        labels = route_counts.index.tolist()
        sizes = route_counts.values.tolist()
        pie_colors = [colors.get(l, '#95a5a6') for l in labels]

        axes[0].pie(
            sizes, labels=labels, colors=pie_colors,
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 11}
        )
        axes[0].set_title('Routing Distribution')

        # Bar chart: accuracy by route
        routes = route_acc.index.tolist()
        accs = route_acc.values.tolist()
        bar_colors = [colors.get(r, '#95a5a6') for r in routes]

        axes[1].bar(routes, [a * 100 for a in accs], color=bar_colors)
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy by Routing Path')
        axes[1].set_ylim(0, 100)

        for i, (r, a) in enumerate(zip(routes, accs)):
            axes[1].text(i, a * 100 + 2, f'{a*100:.1f}%', ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(out / 'fig_routing_distribution.png', bbox_inches='tight')
        plt.savefig(out / 'fig_routing_distribution.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved routing distribution figure to {out}")

    return df


def generate_complexity_scatter(adaptive_results: list, output_dir: str = None):
    """
    Generate Figure 2: Complexity Score vs Accuracy, colored by route.
    """
    if not adaptive_results:
        return

    df = pd.DataFrame(adaptive_results)

    # Extract composite score from complexity profile
    if 'complexity_profile' in df.columns:
        df['composite_score'] = df['complexity_profile'].apply(
            lambda x: x.get('composite_score', 0) if isinstance(x, dict) else 0
        )
    else:
        print("No complexity_profile in results, skipping scatter plot")
        return

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = {
            'symbolic': '#2ecc71',
            'neural_cot': '#3498db',
            'hybrid': '#e74c3c',
        }
        markers = {
            'symbolic': 's',
            'neural_cot': 'o',
            'hybrid': '^',
        }

        for route in df['route_taken'].unique():
            mask = df['route_taken'] == route
            subset = df[mask]

            correct = subset[subset['is_correct'] == True]
            wrong = subset[subset['is_correct'] == False]

            # Correct answers: filled
            if len(correct) > 0:
                ax.scatter(
                    correct['composite_score'],
                    correct['tokens_total'],
                    c=colors.get(route, '#95a5a6'),
                    marker=markers.get(route, 'o'),
                    label=f'{route} (correct)',
                    alpha=0.7,
                    s=40,
                )

            # Wrong answers: hollow
            if len(wrong) > 0:
                ax.scatter(
                    wrong['composite_score'],
                    wrong['tokens_total'],
                    c='none',
                    edgecolors=colors.get(route, '#95a5a6'),
                    marker=markers.get(route, 'o'),
                    label=f'{route} (wrong)',
                    alpha=0.5,
                    s=40,
                    linewidths=1.5,
                )

        ax.set_xlabel('Composite Complexity Score C(p)')
        ax.set_ylabel('Tokens Consumed')
        ax.set_title('Complexity vs Token Usage by Routing Path')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add threshold lines
        ax.axvline(x=0.30, color='gray', linestyle='--', alpha=0.5, label='τ_low')
        ax.axvline(x=0.65, color='gray', linestyle=':', alpha=0.5, label='τ_high')

        plt.tight_layout()
        plt.savefig(out / 'fig_complexity_scatter.png', bbox_inches='tight')
        plt.savefig(out / 'fig_complexity_scatter.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved complexity scatter to {out}")


def generate_token_comparison(summary: dict, output_dir: str = None):
    """
    Generate Figure 3: Token consumption comparison bar chart.
    """
    results = summary.get('results', summary)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        methods = []
        tokens = []
        accuracies = []

        for method, data in results.items():
            methods.append(method.replace('_', ' ').title())
            tokens.append(data.get('avg_tokens', 0))
            accuracies.append(data.get('accuracy', 0) * 100)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        x = np.arange(len(methods))
        width = 0.35

        # Bars for tokens
        bars1 = ax1.bar(
            x - width/2, tokens, width,
            label='Avg Tokens', color='#3498db', alpha=0.8
        )
        ax1.set_ylabel('Average Tokens per Problem', color='#3498db')
        ax1.tick_params(axis='y', labelcolor='#3498db')

        # Second y-axis for accuracy
        ax2 = ax1.twinx()
        bars2 = ax2.bar(
            x + width/2, accuracies, width,
            label='Accuracy (%)', color='#2ecc71', alpha=0.8
        )
        ax2.set_ylabel('Accuracy (%)', color='#2ecc71')
        ax2.tick_params(axis='y', labelcolor='#2ecc71')
        ax2.set_ylim(0, 100)

        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=15, ha='right')
        ax1.set_title('Token Consumption vs Accuracy by Method')

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9
            )
        for bar in bars2:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9
            )

        fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
        plt.tight_layout()
        plt.savefig(out / 'fig_token_comparison.png', bbox_inches='tight')
        plt.savefig(out / 'fig_token_comparison.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved token comparison to {out}")


def generate_accuracy_by_level(adaptive_results: list, output_dir: str = None):
    """
    Generate Figure 4: Accuracy by MATH difficulty level.
    """
    if not adaptive_results:
        return

    df = pd.DataFrame(adaptive_results)

    # Filter to MATH problems only (they have levels > 0)
    if 'complexity_profile' in df.columns:
        # Level info might be in the problem_id: math_0, math_1, etc.
        # We need to recover level from original data or from the ID
        pass

    # If we have level info embedded somewhere, plot it
    # For now, use composite score as proxy for difficulty
    if 'complexity_profile' in df.columns:
        df['composite_score'] = df['complexity_profile'].apply(
            lambda x: x.get('composite_score', 0) if isinstance(x, dict) else 0
        )

        # Bin into difficulty levels based on composite score
        bins = [0, 0.15, 0.30, 0.50, 0.70, 1.0]
        labels_bins = ['Very Easy', 'Easy', 'Medium', 'Hard', 'Very Hard']
        df['difficulty_bin'] = pd.cut(
            df['composite_score'], bins=bins, labels=labels_bins
        )

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)

            # Accuracy by difficulty bin
            acc_by_diff = df.groupby('difficulty_bin')['is_correct'].agg(['mean', 'count'])
            acc_by_diff.columns = ['accuracy', 'count']

            fig, ax = plt.subplots(figsize=(8, 5))

            valid = acc_by_diff[acc_by_diff['count'] > 0]
            bars = ax.bar(
                range(len(valid)),
                valid['accuracy'] * 100,
                color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#8e44ad'][:len(valid)],
                alpha=0.8,
            )

            ax.set_xticks(range(len(valid)))
            ax.set_xticklabels(valid.index, rotation=15)
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Accuracy by Problem Difficulty (Adaptive System)')
            ax.set_ylim(0, 100)

            for i, bar in enumerate(bars):
                height = bar.get_height()
                count = valid.iloc[i]['count']
                ax.text(
                    bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%\n(n={int(count)})',
                    ha='center', va='bottom', fontsize=9
                )

            plt.tight_layout()
            plt.savefig(out / 'fig_accuracy_by_difficulty.png', bbox_inches='tight')
            plt.savefig(out / 'fig_accuracy_by_difficulty.pdf', bbox_inches='tight')
            plt.close()
            print(f"Saved accuracy by difficulty to {out}")


def generate_all_figures(results_dir: str):
    """
    Generate all paper figures from a results directory.
    """
    rdir = Path(results_dir)
    figures_dir = rdir / "figures"
    figures_dir.mkdir(exist_ok=True)

    data = load_results(results_dir)

    # Main summary
    if 'final_summary' in data:
        summary = data['final_summary']
        generate_main_comparison_table(
            summary, str(rdir / "table_main_results.csv")
        )
        generate_token_comparison(summary, str(figures_dir))

    # Adaptive results
    adaptive_results = []
    for key in ['results_adaptive_math', 'results_adaptive_gsm8k']:
        if key in data and isinstance(data[key], list):
            adaptive_results.extend(data[key])

    if adaptive_results:
        generate_routing_analysis(adaptive_results, str(figures_dir))
        generate_complexity_scatter(adaptive_results, str(figures_dir))
        generate_accuracy_by_level(adaptive_results, str(figures_dir))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results_directory>")
        print("Example: python analyze_results.py results/20250219_143022")
        sys.exit(1)

    generate_all_figures(sys.argv[1])
