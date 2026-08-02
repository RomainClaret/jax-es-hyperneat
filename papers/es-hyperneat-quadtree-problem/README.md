# On Scaling Coordinate-Based Neuroevolution: The Quadtree Bottleneck in ES-HyperNEAT

Romain Claret, Michael O'Neill, Paul Cotofrei, Kilian Stoffel. Preprint / technical report.

**Read the paper:** https://claret.tech/pdf/claret2026quadtree — also on arXiv (identifier added on
announcement). The manuscript itself is not kept in this repository: it is maintained alongside the
arXiv submission, and those two are the authoritative versions. This directory holds the data,
scripts and figures behind it.

A diagnostic study of **JAX-ESHN**, a JAX/TensorNEAT implementation of ES-HyperNEAT. It
shows that ES-HyperNEAT's adaptive quadtree subdivision is structurally incompatible with
population-level `vmap`/XLA batching: each CPPN discovers a different, variable-cardinality
set of substrate positions, so the population cannot be vectorized. Batching within a CPPN
plateaus near ~1.7×; replacing the quadtree with a precomputed spatial hash grid
(HSHG) fails by order-of-magnitude over-discovery. Validated across five benchmarks: XOR
(GPU depth × population scaling) plus Parity-3, circle, sine, and CartPole (a CPU-vs-CPU
control that isolates the implementation from the hardware). It is the diagnostic companion
to EMR-HyperNEAT, which resolves the bottleneck.

## Contents

```
figures/                  rendered figures reproducing those in the paper (pdf + png only)
results/                  committed per-seed raw results (JSON), the source of truth for every number
  jax_eshn_*_results.json     JAX-ESHN multi-benchmark runs (sine/circle/parity3)
  cartpole_d{2,3,4}/          JAX-ESHN CartPole runs (per depth)
  pureples_*_results.json     PUREPLES CPU baseline runs
  hshg_*_results.json         HSHG ablation runs
  compile_vs_eval_isolation.json   the §"isolation" compile-vs-eval measurement
  benchmark_results_summary.md     consolidated multi-benchmark summary tables
benchmark_checkpoints/    XOR depth × population scaling checkpoints + the consolidated
                          es_hyperneat_population_scaling_tables.md (feed Tables 2-3 and Fig. 4)
scripts/
  runners/                verify_results.py, verify_scaling_tables.py: PASS/FAIL recompute of the paper tables
  analysis/               analyze_results.py, analyze_hshg_results.py, analyze_compile_vs_eval.py, compute_table_values.py
  figures/                the 5 standalone figure .tex + build_figures.sh + the extract_* TikZ-coord scripts
REPRODUCIBILITY.md        table/figure -> script -> data map, config, and known caveats
```

## Reproduce every number (no GPU, no evolution)

From this directory:

```bash
python scripts/runners/verify_results.py            # multi-benchmark Tables (solve rates, s/gen, scaling) -> all PASS
python scripts/runners/verify_scaling_tables.py     # XOR scaling Tables 2-3 + D8-D10 -> all PASS
python scripts/analysis/analyze_hshg_results.py     # HSHG over-discovery / 0% solve (Table 7)
python scripts/analysis/analyze_results.py          # multi-benchmark statistics + LaTeX
python scripts/analysis/analyze_compile_vs_eval.py  # compile-vs-eval isolation ratios
python scripts/analysis/compute_table_values.py     # Table 2 / Fig. 4 values from the scaling md
```

The two `scripts/runners/verify_*.py` print `RESULT: N/N passed, 0 failed` when the committed
data reproduces the paper. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the per-table data map
and the exact hyperparameters.

## Rebuild the figures

The paper `\includegraphics` the pdfs in `figures/`; each is generated from a standalone
`.tex` in `scripts/figures/`:

```bash
bash scripts/figures/build_figures.sh   # scripts/figures/*.tex -> figures/*.pdf + *.png
```

Requires `pdflatex` (with TikZ/pgfplots) and `pdftocairo`. Rebuilt PDFs are visually identical
but byte-differ from the committed ones (embedded PDF timestamps), so a rebuild shows the five
pdfs as modified in `git status`; restore with `git checkout -- figures/` unless you intend to
update them.

## Re-run experiments (optional)

The committed JSON is canonical. To re-run with the standalone code (from the repo root):

```bash
python -m jax_es_hyperneat.driver --task xor  --impl jax-eshn --depths 2 --pop 150 --gens 60 --seeds 42-44
python -m jax_es_hyperneat.driver --task sine --impl pureples --depths 2 3 4 --seeds 0-29
python -m jax_es_hyperneat.driver --task xor  --impl hshg     --depths 2 --pop 50 --gens 50 --seeds 0-29
```

Solve rates reproduce; per-seed counts may differ by ±1–2 at the noise floor and absolute
timings are machine-bound (see REPRODUCIBILITY.md).
