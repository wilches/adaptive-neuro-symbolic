# Adaptive Neuro-Symbolic Reasoning via Dynamic Routing

> **Paper:** *Adaptive Neuro-Symbolic Reasoning via Dynamic Routing and Structural Complexity Metrics for Mathematical Problem Solving*
> Submitted to **MAPI 4 — Conferencia Colombiana de Matemáticas Aplicadas e Industriales** (2026).

This repository contains the code, cached experimental outputs, and reproducibility scripts for an adaptive framework that routes mathematical problems between three reasoning paths — **symbolic** (SymPy), **neural chain-of-thought** (LLM), and **hybrid** — based on five structural complexity metrics computed directly from the problem statement.

---

## TL;DR

- We compute five structural metrics on a math problem ($d_L, n_V, \sigma_S, \omega_N, \kappa_Q$) and use a threshold-based router to choose a reasoning path *before* spending tokens.
- Evaluated on **MATH + GSM8K** ($n = 399$) with **stratified 5-fold cross-validation** using **DeepSeek-R1-Distill-Qwen-14B** ($T = 0$). Total API cost: **≈ $0.35**, **no GPU required**.
- Headline numbers (held-out folds):
  - Adaptive routing **outperforms random routing by +6.5pp** → the complexity metrics carry real signal.
  - Adaptive routing **saves 3.9–7.5% tokens vs. CoT-only**, at a controlled **−3.0 to −4.5pp accuracy cost**. The contribution is an **efficiency trade-off**, *not* accuracy parity with CoT.
  - Fixed CoT + symbolic-answer-checking is **counterproductive** (lower accuracy, more tokens) under our implementation.
  - The standalone symbolic engine is limited by **parser fidelity** — SymPy parses many MATH problems but silently drops constructs.

> ⚠️ **Honest framing.** Adaptive routing does **not** beat CoT-only in accuracy on this benchmark. It saves tokens at a small accuracy cost and beats a random-routing baseline by a clear margin. We report this explicitly per reviewer feedback; the original abstract is corrected in `MAPI4_revised.tex`.

---

## Table of contents

1. [What's in this repo](#whats-in-this-repo)
2. [Quick start (5 minutes, no API key)](#quick-start-5-minutes-no-api-key)
3. [Full reproduction (with API key)](#full-reproduction-with-api-key)
4. [Reproducing the paper's tables from cached results](#reproducing-the-papers-tables-from-cached-results)
5. [System architecture](#system-architecture)
6. [Results summary](#results-summary)
7. [Known limitations](#known-limitations)
8. [Citation](#citation)
9. [License](#license)
10. [Contact](#contact)

---

## What's in this repo

```
.
├── src/                              # Core library
│   ├── complexity_analyzer.py        # Extracts (d_L, n_V, σ_S, ω_N, κ_Q)
│   ├── router.py                     # Threshold-based path selection
│   ├── pipeline.py                   # End-to-end orchestration
│   ├── config.py                     # API provider presets
│   └── engines/
│       ├── symbolic_engine.py        # SymPy + Z3 cascade (local, $0)
│       └── cot_engine.py             # LLM CoT (OpenAI-compatible / Gemini)
│
├── experiments/                      # Scripts to reproduce paper numbers
│   ├── load_data.py                  # Loads MATH + GSM8K from Hugging Face
│   ├── run_experiment.py             # Runs all baselines + adaptive system
│   ├── kfold_evaluation.py           # Stratified 5-fold CV (main Table 1)
│   ├── optimize_thresholds.py        # Single-split threshold grid search
│   ├── optimize_thresholds_v2.py     # Stricter held-out evaluation
│   ├── analyze_results.py            # Generates figures and LaTeX tables
│   ├── revision_analyses.py          # Reviewer-response analyses (no new API calls)
│   ├── quick_test.py                 # Local smoke tests (no API)
│   ├── test_raw_api.py               # Minimal API connectivity check
│   └── test_api_connection.py        # Full end-to-end API check
│
├── results/
│   └── 20260224_210542/              # Cached experiment outputs
│       ├── final_summary.json        # Aggregated metrics (main numbers)
│       ├── results_symbolic.json     # Per-problem symbolic outputs
│       ├── results_cot_math.json     # Per-problem CoT outputs (MATH)
│       ├── results_cot_gsm8k.json    # Per-problem CoT outputs (GSM8K)
│       ├── results_hybrid.json       # Per-problem hybrid outputs
│       ├── results_adaptive_math.json
│       ├── results_adaptive_gsm8k.json
│       ├── optimal_thresholds.json   # Grid-search optimum
│       └── revision_numbers.json     # Reviewer-response analyses
│
├── paper/                            # (optional) LaTeX sources of submission
│   ├── MAPI4_revised.tex             # Reviewer-revised version (long)
│   └── MAPI4_revised_6pg.tex         # Reviewer-revised version (6-page)
│
├── requirements.txt
├── .env.example                      # Template for API keys
├── REPRODUCIBILITY.md                # Step-by-step replication guide
├── CITATION.cff
└── README.md
```

---

## Quick start (5 minutes, no API key)

This path verifies the symbolic engine, complexity analyzer, and router on a handful of problems **without any API calls**.

```bash
# 1. Clone
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Create a clean environment (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run local-only smoke tests (no API key needed)
python experiments/quick_test.py
```

You should see four sections passing: complexity analyzer, symbolic engine, router, and a small symbolic-only end-to-end run.

---

## Full reproduction (with API key)

The full pipeline calls an LLM via API. The paper uses **DeepSeek-R1-Distill-Qwen-14B** through **SiliconFlow**, but any OpenAI-compatible provider works (OpenAI, Gemini are also supported). Total API spend for the full 399-problem run was **≈ $0.35**.

### 1. Get an API key

Pick one provider and follow the instructions:

| Provider     | Default model used in paper                            | Where to get a key                             |
|--------------|--------------------------------------------------------|------------------------------------------------|
| SiliconFlow  | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`             | https://cloud.siliconflow.com/account/ak       |
| OpenAI       | `gpt-4o-mini` (fallback default)                       | https://platform.openai.com/api-keys           |
| Gemini       | `gemini-1.5-flash` (free tier)                         | https://aistudio.google.com/app/apikey         |

### 2. Configure the environment

```bash
cp .env.example .env
# Edit .env and set ONE of:
#   SILICONFLOW_API_KEY=sk-...
#   OPENAI_API_KEY=sk-...
#   GEMINI_API_KEY=...
```

### 3. Verify the API works

```bash
python experiments/test_raw_api.py           # Minimal check
python experiments/test_api_connection.py    # End-to-end on 3 problems
```

### 4. Run the full experiment

> ⚠️ This calls the LLM 3×399 = 1197 times across baselines (CoT-only + Fixed Hybrid + Adaptive) plus 399 zero-cost symbolic runs. Expect ~20–40 minutes wall-clock and **≈ $0.35** on SiliconFlow with DeepSeek-R1-Distill-Qwen-14B.

```bash
python experiments/run_experiment.py \
    --provider siliconflow \
    --math-per-level 60 \       # 5 levels × 60 ≈ 300 MATH problems (paper uses 299 after filtering)
    --gsm8k-size 100 \
    --output-dir results
```

Results land in `results/<timestamp>/`. Then run cross-validation and analysis:

```bash
# Stratified 5-fold CV → reproduces Table 1
python experiments/kfold_evaluation.py

# Grid-search thresholds on a held-out split
python experiments/optimize_thresholds_v2.py

# Tables + figures
python experiments/analyze_results.py
```

---

## Reproducing the paper's tables from cached results

To avoid spending any API budget, the repo ships the **exact cached outputs** of the run reported in the paper (`results/20260224_210542/`). All analyses can be re-derived from these JSONs in seconds:

```bash
# Table 1 + per-fold breakdown (5-fold stratified CV)
python experiments/kfold_evaluation.py

# Threshold optimization on cached data
python experiments/optimize_thresholds_v2.py

# Reviewer-response analyses (routing distribution, parse-vs-solve failures,
# token accounting, 15-config weight sensitivity)
python experiments/revision_analyses.py
```

`experiments/revision_analyses.py` writes a machine-readable dump to `results/20260224_210542/revision_numbers.json` — this is the source of all numbers added in the revision (Section 3 + reviewer responses).

---

## System architecture

```
                Input problem p
                       │
                       ▼
        ┌─────────────────────────────────┐
        │ Complexity analyzer  Φ(p)        │
        │  d_L  n_V  σ_S  ω_N  κ_Q         │
        └─────────────────────────────────┘
                       │
                       ▼
        ┌─────────────────────────────────┐
        │ Adaptive router  R               │
        │ thresholds: τ_low, τ_high,       │
        │             σ_sym, σ_neu         │
        └─────────────────────────────────┘
              │           │           │
              ▼           ▼           ▼
        ┌────────┐  ┌──────────┐  ┌──────────────┐
        │Symbolic│  │ Neural   │  │  Hybrid      │
        │ SymPy  │  │  CoT     │  │ CoT then     │
        │ (+ Z3) │  │  (LLM)   │  │ answer-check │
        └────────┘  └──────────┘  └──────────────┘
              │           │           │
              ▼           ▼           ▼
           Result + per-problem efficiency metrics
```

### The five complexity metrics

| Symbol     | Name                        | What it captures                                                        |
|------------|-----------------------------|-------------------------------------------------------------------------|
| $d_L$      | Logical depth               | LaTeX expression-tree depth + logical connectives + proof indicators    |
| $n_V$      | Variable count              | Distinct variables (single letters, Greek, subscripted)                 |
| $\sigma_S$ | Symbolic structure ratio    | `math_tokens / (math_tokens + nl_tokens)` ∈ [0, 1]                      |
| $\omega_N$ | Nesting complexity          | Max depth of nested brackets / braces                                   |
| $\kappa_Q$ | Quantifier alternation      | Alternations between ∀ and ∃ (LaTeX + natural language)                 |

Metrics are min-max normalized using 5th/95th percentiles and combined as
$$C(p) = \sum_i w_i \cdot \hat{\phi}_i(p), \quad w = (0.25,\, 0.15,\, 0.25,\, 0.15,\, 0.20).$$

Weights are fixed a priori. Robustness to the choice of weights is characterized in the paper (3pp accuracy band over 15 configurations).

### The routing rule

$$
R(p) =
\begin{cases}
\text{Symbolic}   & C(p) < \tau_{\text{low}} \;\land\; \sigma_S(p) > \sigma_{\text{sym}} \\
\text{Neural CoT} & C(p) > \tau_{\text{high}} \;\lor\; \sigma_S(p) < \sigma_{\text{neu}} \\
\text{Hybrid}     & \text{otherwise}
\end{cases}
$$

Thresholds optimized on training folds by grid search over a scalarized objective $S = 0.7\cdot\text{accuracy} + 0.3\cdot\text{token-efficiency}$.

### The hybrid path is *answer-checking*, not verification

The hybrid path generates a CoT solution, then substitutes the final answer into the parsed problem expression: the answer is accepted iff the residual simplifies to zero (or, for purely numerical problems, the SymPy-computed value matches within $10^{-6}$). **Intermediate CoT steps are not validated.** Problems that don't parse as equations or numerical expressions skip the check and are accepted with an `unverified` flag. We call this *answer-checking* rather than *verification* explicitly to avoid overclaiming (see reviewer comment C4 in the revised paper).

---

## Results summary

**Main results (5-fold stratified CV on MATH + GSM8K, $n = 399$):**

| Method                  | Accuracy (%)      | Avg tokens | Token save vs. CoT |
|-------------------------|-------------------|------------|--------------------|
| Symbolic-first (+CoT fallback) | 36.86 ± 3.32 |    535.9   |     —              |
| Random routing          | 54.39 ± 2.63      |   1123.2   |     —              |
| **CoT-only**            | **63.91 ± 3.52**  |   1363.6   |   baseline         |
| Fixed Hybrid            | 61.91 ± 3.31      |   1453.0   |   −6.6%            |
| Adaptive (orig. $\tau$) | 59.40 ± 3.01      |   1260.9   |   **+7.5%**        |
| Adaptive (opt. $\tau$)  | 60.90 ± 4.50      |   1310.9   |   **+3.9%**        |

> The symbolic-first accuracy (36.86%) is dominated by **CoT fallback on word problems**. Disaggregated MATH-vs-GSM8K numbers and a parse-vs-solve breakdown are reported in the paper's Section 3.3 (Failure-mode anatomy).

**Adaptive router's per-route outcomes (original thresholds, full $n = 399$):**

| Route       | Count |   %   | Per-route accuracy | Avg tokens |
|-------------|------:|------:|-------------------:|-----------:|
| Symbolic    |    36 |  9.0% |              22.22 |      415.7 |
| Neural CoT  |   184 | 46.1% |          **80.98** |      988.8 |
| Hybrid      |   179 | 44.9% |              50.84 |     1713.9 |

The Neural-CoT route concentrates the LLM's reliable region: routed problems achieve **~81%** accuracy, well above the 63.91% CoT-only baseline. The hybrid route is the main aggregate bottleneck.

**Weight robustness (paper's Section 3 sensitivity analysis):**
Sweeping 15 weight configurations (uniform, single-metric-emphasis 0.45, single-metric-ablation, seeded random permutations) with thresholds fixed at the optimized values, accuracy stays in **[61.40%, 64.41%], mean 63.27 ± 0.80** — the routing decision depends on *which* metrics are used, not on their precise weighting.

---

## Known limitations

These are the limitations explicitly acknowledged in the paper (Sec. 4) — please don't read more into the results than they support:

1. **The symbolic engine's parser fidelity is the dominant bottleneck.** SymPy returns syntactically valid expressions that silently *drop* parts of the original problem (e.g., dropping a `\sin` inside an equation) and then "solve" the gutted expression confidently. Improving the LaTeX→SymPy parser would directly amplify routing benefits.
2. **Single LLM, two benchmarks.** All numbers use DeepSeek-R1-Distill-Qwen-14B on MATH + GSM8K. Behavior with stronger models or other benchmarks (MMLU-MATH, OlympiadBench, etc.) is not characterized here.
3. **Manually-set metric weights.** Robustness is characterized empirically, but weights are not learned end-to-end.
4. **Adaptive routing is an efficiency trade-off**, not an accuracy improvement over CoT-only — see honest framing above.
5. **Hybrid is answer-checking, not verification.** Statements about hybrid being "counterproductive" are specific to this answer-checking variant; richer step-level verification (e.g., proof checking) is left to future work.

---

## Citation

If you use this code, please cite the paper (final BibTeX will be added once proceedings are published):

```bibtex
@inproceedings{herrera2026adaptive,
  title     = {Adaptive Neuro-Symbolic Reasoning via Dynamic Routing
               and Structural Complexity Metrics for Mathematical Problem Solving},
  author    = {Herrera Wilches, Juan Camilo},
  booktitle = {Proceedings of the 4th Colombian Conference on Applied
               and Industrial Mathematics (MAPI 4)},
  year      = {2026},
  address   = {Colombia}
}
```

A `CITATION.cff` is also provided for GitHub's "Cite this repository" widget.

---

## License

Code is released under the **MIT License** (see `LICENSE`). Benchmarks (MATH, GSM8K) follow their original licenses.

---

## Contact

Juan Camilo Herrera Wilches — Independent Researcher, Colombia
Email: wilchesch2010@gmail.com

Issues and pull requests welcome. For questions about the paper, please reference the section / reviewer comment you have in mind.
