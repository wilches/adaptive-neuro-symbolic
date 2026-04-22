"""Generate all figures for the paper using optimization_results_v2.json"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['figure.dpi'] = 150

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    results_dir = Path("results/20260224_210542")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    with open(results_dir / "optimization_results_v2.json") as f:
        opt_data = json.load(f)

    test = opt_data['test_results']

    # ═══════════════════════════════════════════════
    # FIGURE 1: Main Comparison Bar Chart
    # ═══════════════════════════════════════════════

    methods = [
        'Symbolic\nOnly',
        'Random\nRouting',
        'CoT\nOnly',
        'Fixed\nHybrid',
        'Adaptive\n(original)',
        'Adaptive\n(optimized)',
    ]

    accuracies = [
        test['symbolic_only']['accuracy'] * 100,
        test['random_routing']['accuracy'] * 100,
        test['cot_only']['accuracy'] * 100,
        test['fixed_hybrid']['accuracy'] * 100,
        test['adaptive_original']['accuracy'] * 100,
        test['adaptive_optimized']['accuracy'] * 100,
    ]

    avg_tokens = [
        test['symbolic_only']['avg_tokens'],
        test['random_routing']['avg_tokens'],
        test['cot_only']['avg_tokens'],
        test['fixed_hybrid']['avg_tokens'],
        test['adaptive_original']['avg_tokens'],
        test['adaptive_optimized']['avg_tokens'],
    ]

    colors = ['#95a5a6', '#bdc3c7', '#3498db', '#e74c3c', '#f39c12', '#2ecc71']

    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax1.bar(x - width/2, accuracies, width,
                     label='Accuracy (%)', color=colors, alpha=0.85,
                     edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, avg_tokens, width,
                     label='Avg Tokens', color=colors, alpha=0.4,
                     edgecolor='black', linewidth=0.5, hatch='//')
    ax2.set_ylabel('Average Tokens per Problem', fontsize=12)

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.set_title('Method Comparison on Held-Out Test Set (n=120)', fontsize=14)

    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(figures_dir / 'fig1_main_comparison.png', bbox_inches='tight')
    plt.savefig(figures_dir / 'fig1_main_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("Saved fig1_main_comparison")

    # ═══════════════════════════════════════════════
    # FIGURE 2: Routing Distribution (Optimized)
    # ═══════════════════════════════════════════════

    route_counts = test['adaptive_optimized']['route_counts']
    route_correct = test['adaptive_optimized']['route_correct']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    labels = list(route_counts.keys())
    sizes = list(route_counts.values())
    pie_colors = {
        'symbolic': '#2ecc71',
        'neural_cot': '#3498db',
        'hybrid': '#e74c3c'
    }
    colors_pie = [pie_colors.get(l, '#95a5a6') for l in labels]

    axes[0].pie(sizes, labels=[f"{l}\n(n={s})" for l, s in zip(labels, sizes)],
                colors=colors_pie, autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 10})
    axes[0].set_title('Routing Distribution\n(Adaptive Optimized)', fontsize=12)

    # Accuracy by route
    routes = list(route_counts.keys())
    accs = [route_correct[r] / max(1, route_counts[r]) * 100 for r in routes]
    colors_bar = [pie_colors.get(r, '#95a5a6') for r in routes]

    bars = axes[1].bar(routes, accs, color=colors_bar, alpha=0.85,
                        edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy by Routing Path', fontsize=12)
    axes[1].set_ylim(0, 100)
    axes[1].axhline(y=67.50, color='gray', linestyle='--', alpha=0.7,
                     label='CoT baseline (67.5%)')
    axes[1].legend()

    for bar, acc in zip(bars, accs):
        axes[1].text(bar.get_x() + bar.get_width()/2., acc + 2,
                     f'{acc:.1f}%', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(figures_dir / 'fig2_routing_analysis.png', bbox_inches='tight')
    plt.savefig(figures_dir / 'fig2_routing_analysis.pdf', bbox_inches='tight')
    plt.close()
    print("Saved fig2_routing_analysis")

    # ═══════════════════════════════════════════════
    # FIGURE 3: Token Savings Waterfall
    # ═══════════════════════════════════════════════

    fig, ax = plt.subplots(figsize=(8, 5))

    savings_methods = ['Fixed\nHybrid', 'Adaptive\n(original)', 'Adaptive\n(optimized)']
    cot_tokens_val = test['cot_only']['avg_tokens']
    savings = [
        ((cot_tokens_val - test['fixed_hybrid']['avg_tokens']) / cot_tokens_val) * 100,
        ((cot_tokens_val - test['adaptive_original']['avg_tokens']) / cot_tokens_val) * 100,
        ((cot_tokens_val - test['adaptive_optimized']['avg_tokens']) / cot_tokens_val) * 100,
    ]

    bar_colors = ['#e74c3c' if s < 0 else '#2ecc71' for s in savings]
    bars = ax.bar(savings_methods, savings, color=bar_colors, alpha=0.85,
                   edgecolor='black', linewidth=0.5)

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Token Savings vs CoT-only (%)')
    ax.set_title('Token Efficiency Relative to CoT Baseline')

    for bar, s in zip(bars, savings):
        ax.text(bar.get_x() + bar.get_width()/2.,
                bar.get_height() + (0.5 if s >= 0 else -1.5),
                f'{s:+.1f}%', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(figures_dir / 'fig3_token_savings.png', bbox_inches='tight')
    plt.savefig(figures_dir / 'fig3_token_savings.pdf', bbox_inches='tight')
    plt.close()
    print("Saved fig3_token_savings")

    # ═══════════════════════════════════════════════
    # FIGURE 4: Ablation — Adaptive vs Random
    # ═══════════════════════════════════════════════

    fig, ax = plt.subplots(figsize=(7, 5))

    ablation_methods = ['Random\nRouting', 'Adaptive\n(original τ)', 'Adaptive\n(optimized τ)']
    ablation_acc = [
        test['random_routing']['accuracy'] * 100,
        test['adaptive_original']['accuracy'] * 100,
        test['adaptive_optimized']['accuracy'] * 100,
    ]
    ablation_colors = ['#bdc3c7', '#f39c12', '#2ecc71']

    bars = ax.bar(ablation_methods, ablation_acc, color=ablation_colors,
                   alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Ablation: Router Effectiveness\n(Random vs Complexity-Guided Routing)')
    ax.set_ylim(0, 100)

    # Add CoT baseline line
    ax.axhline(y=67.50, color='#3498db', linestyle='--', alpha=0.7,
               label='CoT-only baseline (67.5%)')
    ax.legend()

    for bar, acc in zip(bars, ablation_acc):
        ax.text(bar.get_x() + bar.get_width()/2., acc + 1.5,
                f'{acc:.1f}%', ha='center', fontsize=11, fontweight='bold')

    # Add arrow showing improvement
    ax.annotate('', xy=(1, ablation_acc[1]), xytext=(0, ablation_acc[0]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    mid_y = (ablation_acc[0] + ablation_acc[1]) / 2
    ax.text(0.5, mid_y + 2, f'+{ablation_acc[1]-ablation_acc[0]:.1f}pp',
            ha='center', color='red', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(figures_dir / 'fig4_ablation.png', bbox_inches='tight')
    plt.savefig(figures_dir / 'fig4_ablation.pdf', bbox_inches='tight')
    plt.close()
    print("Saved fig4_ablation")

    print(f"\nAll figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
