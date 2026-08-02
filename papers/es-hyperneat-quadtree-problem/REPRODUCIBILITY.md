# Reproducibility

The verify scripts recompute the paper's numbers from the committed raw data in `results/` and
`benchmark_checkpoints/`, printing `PASS/FAIL` per cell: `97/97` (multi-benchmark) and `46/46`
(XOR scaling incl. solve rates).

## Table / figure → script → data

| Paper artifact | Recompute with | Reads |
|----------------|----------------|-------|
| Tables: multi-benchmark solve rates, s/gen, d2→d4 scaling (sine/circle/parity3) | `scripts/runners/verify_results.py` (JAX-ESHN columns only), `scripts/analysis/analyze_results.py` (both, printed not gated) | `results/{jax_eshn,pureples}_{sine,circle,parity3}_*` |
| Table: CartPole per-generation cost | `scripts/analysis/analyze_results.py` (printed, not gated); Baseline solve rates are gated by `tests/test_paper_findings.py` | `results/pureples_cartpole_d{2,3,4}_*`, `results/cartpole_d{2,3,4}/jax_eshn_cartpole_results.json` |
| Table: construction overhead by depth × population (D3–D7) | `scripts/runners/verify_scaling_tables.py`, `scripts/analysis/compute_table_values.py` | `benchmark_checkpoints/tensorneat_eshyperneat_pop{50,150,300,500,1000}_10gens/checkpoint.json` |
| Table: XOR projected 30-gen Total at Pop 500 (construction + 29×per-gen) | `scripts/runners/verify_scaling_tables.py` | `benchmark_checkpoints/..._pop500_10gens/checkpoint.json` |
| Table: JAX-ESHN solve rates D3–D7 + 30-gen-cap rates (24 trials/depth) | `scripts/runners/verify_scaling_tables.py` | `benchmark_checkpoints/tensorneat_eshyperneat_pop{50,100,150,200,250,500,750,1000}/checkpoint.json` |
| D8–D10 Pop-1000 construction overhead + gen-1 solves | `scripts/runners/verify_scaling_tables.py` | `benchmark_checkpoints/..._pop1000{,_depth10}/checkpoint.json` |
| Fig.: XOR runtime figure — **JAX-ESHN band, construction band, construction Pop-1000 line only** | `scripts/figures/extract_jaxeshn_iqr.py`, `scripts/figures/extract_jit_iqr.py`, `scripts/figures/extract_jit_times.py` (coords for `scripts/figures/baseline_iqr.tex`) | `benchmark_checkpoints/es_hyperneat_population_scaling_tables.md` |
| Fig.: same figure — **Baseline band, Baseline solve line, JAX-ESHN solve line** — no generator script; coordinates were computed once and are maintained by hand (recipes in the header of `scripts/figures/baseline_iqr.tex`) | *(none)* | `benchmark_checkpoints/pureples_xor_scaling/`, `benchmark_checkpoints/tensorneat_eshyperneat_pop1000{,_10gens}/checkpoint.json` |
| Table: XOR Baseline runtime + solve rates (D3–D7), and all Section 6 statistics (ANOVA *F*, χ², *r*, *R²*) | *(no committed script — recompute per the note below)* | `benchmark_checkpoints/pureples_xor_scaling/` |
| Table: Quadtree vs HSHG (full pipeline, 0% solve) | `scripts/analysis/analyze_hshg_results.py` | `results/hshg_*`, `results/pureples_{xor,parity3,sine}_*` |
| Compile-vs-eval isolation ratios | `scripts/analysis/analyze_compile_vs_eval.py` | `results/compile_vs_eval_isolation.json` |

## Configuration

Shared ES-HyperNEAT substrate discovery: `initial_depth=0`, `variance_threshold=0.03`,
`division_threshold=0.5`, `iteration_level=1`, substrate output activation `sigmoid`,
hidden `tanh`, CPPN pool {tanh, sin, gauss}.

| Task | depths | pop | gens | seeds | threshold |
|------|--------|-----|------|-------|-----------|
| XOR (scaling) | 1–7 (8–10 exploratory) | 50–1000 | 10 (timing); solve 30 (Baseline) / 300 (JAX-ESHN) | 42,43,44 | 0.98 |
| Parity-3 | 2–4 | 150 | 100 | 0–29 | 0.975 |
| Circle | 2–4 | 150 | 100 | 0–29 | 0.975 |
| Sine | 2–4 | 150 | 100 | 0–29 | 0.95 |
| CartPole | 2–4 | 150 | 100 | 0–29 | 0.95 |

Sine targets differ by side on purpose: the JAX-ESHN side regresses `(sin(πx)+1)/2`
(scaled to [0,1] for the sigmoid output); the PUREPLES side regresses raw `sin(πx)`. Both
reduce `solved` via `1 − MSE ≥ threshold`. Do not harmonize them.

## Baseline XOR depth×population campaign (456 trials)

`benchmark_checkpoints/pureples_xor_scaling/` holds the PUREPLES campaign behind Table 2's
Baseline column, the Baseline series of the runtime figure, and every Section 6 statistic.
Design: D1–D6 × 8 populations × 3 iteration levels × 3 seeds, plus D7 at iteration level 1
(8 × 3) = **456 runs**, 30-generation cap.

| File | Runs | Covers |
|------|------|--------|
| `pureples_eshyperneat_comprehensive.json` | 360 | D1–D5, all populations and iteration levels |
| `pureples_depth6_part2.json` / `_part3.json` | 18 / 36 | D6 (pops 750–1000 / 50–500) |
| `pureples_depth7_isolated_20260108_200334.json` | 18 | D7, pops 50–500 |
| `pureples_depth7_isolated_20260115_211601.json` | 3 | D7, pop 750 |
| `pureples_depth7_isolated_20260113_132240.json` | 3 | D7, pop 1000 |
| `pureples_depth6_partial.md` | 18 | D6 iteration level 1 — **markdown table only, not JSON** |
| `pureples_eshyperneat_avg_solve_generations.md` | — | derived: per-seed solve generations + solve-rate table |

**Reproduction notes.**
- The paper reports **population** standard deviation (`ddof=0`). With sample std the D7 row comes
  out 118.7 / 646.7 / 275.6 / 609.0 and matches nothing; with `ddof=0` it reproduces exactly
  (291.36 ± 96.95, 824.44 ± 528.01, 644.87 ± 225.06, 1010.28 ± 497.27 minutes).
- Solve rates are the **iteration-level-1** subset, 24 trials per depth: D3 15/24 = 62.5%,
  D4 9/24 = 37.5%, D5 8/24 = 33.3%, D6 2/24 = 8.3%, D7 1/24 = 4.2%. The D6 count needs both
  sources (1 solve in the JSON files + 1 in `pureples_depth6_partial.md`, pop 300 / seed 42).
- Section 6's statistics use all 456 runs (the reported *F* has df-within 449 = 456 − 7). From the
  438 JSON runs alone you get *F* = 85.34, η² = 0.543, *r*(positions, time) = 0.735,
  *r*(depth, time) = 0.452, against the published 87.04 / 0.54 / 0.73 / 0.44 — the difference is
  exactly the 18 markdown-only runs.
- D6 pops 200/500 have no stored total runtime; the table's 110 ± 80 cell is reconstructed as
  mean per-generation time × 30.

## Known caveats

- **The 30-generation Total is a projection.** The runtime table's JAX-ESHN Total is
  construction overhead + 29 × the post-construction per-generation rate from the 10-generation
  timing reruns (as the table caption defines), not a measured 30-generation wall-clock. The
  factor is 29, not 30, because generation 1 runs inside construction overhead: the benchmark
  times its warm-up generation as `jit_compilation_time_s` and divides the remainder by
  `generations - 1` to get `time_per_gen_post_jit_s`. In every committed record
  `total_evolution_time_s / time_per_gen_post_jit_s` equals `generations - 1` exactly, and
  `total_time_s = jit + total_evolution`, so a measured 30-generation run costs
  construction + 29 × per-gen. The D8–D10 points come from the Pop-1,000 campaigns.
- **`max_weight` / `band_threshold`.** The JAX-ESHN runs use `max_weight=8.0` and
  `band_threshold=0.3` (`jax_es_hyperneat/tasks/base.py:ES_PARAMS`; also the implementation
  defaults), which reproduce the committed JSON. PUREPLES applies `max_weight=5.0` and, inside
  its `query_cppn`, a hardcoded `0.2` connection-expression cutoff, which the HSHG ablation
  mirrors (`n=0.2`).
- **Frozen data is canonical.** Re-running the standalone code reproduces *solve rates*, but
  per-seed solve *counts* near the noise floor (sine D2 sits at 29/30; Parity-3 is in the
  13–23% band) can shift by ±1–2 seeds depending on RNG threading, and *absolute timings*
  are machine- and (for CartPole) gymnasium-version-bound. The committed JSONs are the source
  of truth; the verify scripts are the acceptance gate.
- **Compile-vs-eval isolation reproduces from committed data.** The §"isolation" ratios
  recompute from the committed `compile_vs_eval_isolation.json` via
  `scripts/analysis/analyze_compile_vs_eval.py`; a from-scratch re-measurement harness is not
  part of this release.
- **PUREPLES population size is fixed at 150.** The PUREPLES baseline takes its population
  size solely from the neat-python config (`pop_size = 150`), so every PUREPLES run used
  pop 150 regardless of the `pop50`/`pop150` suffix in its filename, and `driver --pop`
  affects only the `jax-eshn` and `hshg` implementations, not `pureples`.
- **HSHG pop-150 partials.** A few HSHG pop-150 conditions are incomplete in the frozen data
  (`scripts/analysis/analyze_hshg_results.py` prints `running...` for them). The paper's Table 7 uses
  the complete pop-50 conditions; the 0%-solve conclusion holds at every condition.
- **neat-python is pinned to 0.92** (the version that produced the data). The PUREPLES/HSHG
  baselines read neat-python config files that predate the `no_fitness_termination` parameter
  that neat-python 1.0+/2.0 made mandatory; with a newer neat-python a fresh install of the
  baselines errors at config load. The `[baseline]` extra pins `neat-python==0.92` accordingly
  (`gymnasium` is bounded `<2` for the same forward-compat reason). This affects only re-running
  the baselines from scratch; reproducing the paper from the committed JSON needs neither.
- **JAX / NumPy version range.** The published data was produced with `jax==0.6.1` /
  `numpy==1.26.4` (the versions CI pins and the env spec records). The submodules pull unpinned
  `jax`/`numpy`, so a plain fresh install resolves to newer (e.g. jax 0.10.x / numpy 2.x); the
  core dep bounds are therefore set wide (`jax<0.12`, `numpy<3`) to match what actually installs.
  The algorithm runs identically across that range: jax-eshn sine d2 seed 0 yields the same
  `0.976051` on jax 0.6.1/numpy 1.26 and jax 0.10.2/numpy 2.5, and both verify scripts pass.
