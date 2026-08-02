# Reproducing the paper

The paper is *On Scaling Coordinate-Based Neuroevolution: The Quadtree Bottleneck in
ES-HyperNEAT* (`papers/es-hyperneat-quadtree-problem/`). There are two ways to reproduce it:

- **A. Verify the frozen data**: fast, stdlib-only, no GPU, no evolution. This is the canonical
  acceptance gate and the recommended path.
- **B. Re-run experiments live**: evolve small versions yourself. Slow and machine-dependent; the
  committed JSON remains the source of truth.

> The per-table → script → data map, the code-true configuration table, and the full list of known
> caveats live in
> [`papers/es-hyperneat-quadtree-problem/REPRODUCIBILITY.md`](../papers/es-hyperneat-quadtree-problem/REPRODUCIBILITY.md).
> This page is the how-to; that file is the reference. Read both.

## A. Verify the frozen data (recommended)

Every table and figure recomputes from the committed result data. Run from the paper directory:

```bash
cd papers/es-hyperneat-quadtree-problem

python scripts/runners/verify_results.py          # multi-benchmark tables  -> "RESULT: 97/97 passed, 0 failed"
python scripts/runners/verify_scaling_tables.py   # XOR depth x population   -> "RESULT: 46/46 passed, 0 failed"
python scripts/analysis/analyze_hshg_results.py   # HSHG over-discovery -> "0% solve rate" conclusion
```

The two `scripts/runners/verify_*.py` and `scripts/analysis/analyze_hshg_results.py` are **pure Python
stdlib** (no JAX, no GPU, no extra dependencies), which is what makes the reproduction claim cheap and
portable. They independently recompute each statistic from `results/` and `benchmark_checkpoints/` and
print `PASS/FAIL` per cell.

One more script needs the `[analysis]` extra (`scipy`):

```bash
pip install -e ".[analysis]"
python scripts/analysis/analyze_results.py   # multi-benchmark statistics + LaTeX tables (no PASS/FAIL line)
```

Other helpers feed specific figures/tables: `scripts/analysis/analyze_compile_vs_eval.py`,
`scripts/analysis/compute_table_values.py`, and the `scripts/figures/extract_*` trio (which emit the
TikZ coordinates for the XOR runtime figure, `figures/baseline_iqr.pdf`); the `REPRODUCIBILITY.md`
table says which artifact each one produces.

### Where the data lives

- `papers/es-hyperneat-quadtree-problem/results/` (~2.4 MB): per-seed result JSON. Two schemas
  (both normalized by the tests): JAX-ESHN files are `{metadata, summary, results}`; PUREPLES and
  HSHG files are `{metadata, runs}`. Naming: the JAX-ESHN multi-benchmark files are
  `jax_eshn_{sine,circle,parity3}_results.json` (XOR's depth × population scaling lives under
  `benchmark_checkpoints/`, CartPole under `cartpole_d<depth>/jax_eshn_cartpole_results.json`);
  baseline/ablation files are `pureples_<task>_d<depth>[_pop<N>]_results.json` and
  `hshg_<task>_d<depth>[_pop<N>]_results.json`. Each pairs with a `_checkpoint.json`.
- `papers/es-hyperneat-quadtree-problem/benchmark_checkpoints/`: the XOR depth × population scaling
  checkpoints (one `checkpoint.json` per `tensorneat_eshyperneat_pop*` subdir).

The raw data is committed in-repo (also archived on Zenodo, 10.5281/zenodo.21761118);
the repo-root `scripts/fetch_results.py` is an optional checksum-verifier, not a required download.

To rebuild the paper's figures from their standalone sources, run
`bash scripts/figures/build_figures.sh` from the paper directory; it regenerates
`figures/*.pdf` + `*.png` from `scripts/figures/*.tex` (see the paper README). Rebuilt PDFs are
visually identical but byte-differ from the committed ones (embedded PDF timestamps), so expect
`git status` churn; restore with `git checkout -- figures/` unless you mean to update them.

## B. Re-run experiments live (slow)

The same driver can re-run any experiment. A live re-run writes the exact committed
JSON schema. For example, the faithful sine d2 cell (paper config: pop 150, 100 gens, 30 seeds):

```bash
python -m jax_es_hyperneat.driver --task sine --impl jax-eshn --depths 2 --pop 150 --gens 100 --seeds 0-29
```

A cheap single-seed smoke variant (matches the live test):

```bash
python -m jax_es_hyperneat.driver --task sine --impl jax-eshn --depths 2 --seeds 0 --gens 8
```

`--out` defaults to a `results/` dir under your **current** directory, so a re-run does **not**
clobber the paper's committed `papers/.../results/`.

### Cost: read before you start

Live runs JIT-compile, then evolve. JAX-ESHN and HSHG have **no early stopping** (they run the full
`--gens`); the PUREPLES baseline (neat-python) stops early once its neat config `fitness_threshold`
(0.975; 0.95 for cartpole) is reached; that early-stop knob is distinct from the per-task solve
thresholds in REPRODUCIBILITY.md (e.g. sine counts as solved at 0.95). From the committed timings:

- **sine d2** ≈ 56 s JIT + ~13 s/gen → ~23 min for one 100-gen seed, ≈ **11 h for the full 30-seed
  cell**. Depth inflates both sharply (sine d4 JIT ~330 s; per-gen ~5× from d2→d4).
- **CartPole** is far heavier (d2 ≈ **8.0 h/seed median**, 8.75 h mean; right-skewed) and
  gymnasium-version-bound.
- **PUREPLES** on deeper/harder tasks is the very CPU bottleneck the paper studies (e.g. parity3 d3
  ≈ **26 s/gen** median (29 mean); parity3 d4 ≈ 180 s/gen median).

What reproduces vs. what drifts: **solve rates** reproduce, but per-seed solve *counts* near the
noise floor (sine d2 sits at 29/30; parity3 at 13–23%) can shift ±1–2 seeds with RNG threading, and
**absolute timings** are machine-bound. The committed JSON is canonical; the `verify_*.py` scripts
are the gate. (See the "Known caveats" in `REPRODUCIBILITY.md` for the full list, including the
`max_weight=8.0`/`band_threshold=0.3` config note and the fixed PUREPLES pop=150.)

## Test-driven reproduction

The test suite wraps both paths: `test_paper_findings.py` asserts the frozen solve counts,
`test_reproduction.py` shells out to the verify scripts, and `test_paper_findings_live.py` evolves
small live versions and checks the published findings. See [testing.md](testing.md).

## Next

- Run the tests / CI / `act`: [testing.md](testing.md)
- Re-run with your own configs: [writing-your-own-experiment.md](writing-your-own-experiment.md)
