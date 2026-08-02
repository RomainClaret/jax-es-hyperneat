# Installation

`jax_es_hyperneat` is a small JAX package that builds on pinned forks of **TensorNEAT** (the
CPPN/NEAT engine, required) and **PUREPLES** (the CPU baseline, optional), included as git
submodules.

## Requirements

- **Python ≥ 3.10**
- **JAX** in the range `>=0.5,<0.12` (CI pins `jax==0.6.1` / `jaxlib==0.6.1`; see the version note
  below). CPU is the default and is all you need: the paper is a CPU-bound diagnosis.
- A GPU is **optional**. NVIDIA GPU metrics are recorded when present but are not load-bearing.

## 1. Clone with submodules

```bash
git clone --recursive https://github.com/RomainClaret/jax-es-hyperneat.git
cd jax-es-hyperneat

# if you forgot --recursive:
git submodule update --init --recursive
```

The submodule URLs are HTTPS, so an anonymous clone (and CI) works with no SSH key. If you prefer
SSH for pushing to the forks, add a one-time rewrite:

```bash
git config --global url."git@github.com:".insteadOf "https://github.com/"
```

## 2. Install the package + engine

```bash
pip install -e .                       # the jax_es_hyperneat package (core deps: jax, jaxlib, numpy, pydantic, pyyaml)
pip install -e third_party/tensorneat  # the pinned TensorNEAT fork (REQUIRED)
pip install -e third_party/pureples    # the PUREPLES CPU baseline, optional (only for --impl pureples)
```

## 3. Optional extras

```bash
pip install -e ".[dev]"        # pytest>=7.0, to run the test suite
pip install -e ".[analysis]"   # scipy>=1.10 (for the paper's analysis scripts)
pip install -e ".[baseline]"   # neat-python==0.92, gymnasium>=0.29,<2 (to re-run PUREPLES / CartPole from scratch)
```

You only need `[baseline]` if you want to **re-run** the PUREPLES baseline, the HSHG ablation, or
CartPole live. Reproducing the paper from the committed data needs none of these (it is pure
stdlib; see [reproducing-the-paper.md](reproducing-the-paper.md)).

## 4. Quiet CPU runs

JAX prints backend chatter by default. For quiet CPU-only runs:

```bash
export JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=3
```

## 5. Verify the install

```bash
# quick smoke (3-generation XOR evolution; prints "SMOKE OK")
JAX_PLATFORMS=cpu python jax_es_hyperneat/tests/test_smoke.py

# fast test suite (no evolution beyond a couple of tiny runs; ~2-4 min)
JAX_PLATFORMS=cpu pytest jax_es_hyperneat/tests -m "not slow"
```

If both pass, the install is good. See [testing.md](testing.md) for the full test story.

## Troubleshooting

- **`ModuleNotFoundError: tensorneat`**. You skipped `pip install -e third_party/tensorneat`, or
  cloned without `--recursive` (the `third_party/` dirs are empty). Run
  `git submodule update --init --recursive`, then install the submodule.
- **`neat-python` errors at config load** (e.g. missing `no_fitness_termination`): you have
  neat-python ≥ 1.0. The baselines use config files that predate that mandatory parameter. The
  `[baseline]` extra pins `neat-python==0.92` for exactly this reason; install with
  `pip install -e ".[baseline]"` (do not upgrade neat-python).
- **pip "incompatible" warning about `flax` wanting newer JAX**: cosmetic. TensorNEAT pulls
  `flax`, which prefers `jax>=0.10`, but CI pins `jax==0.6.1`; the warning is non-fatal and the
  tests pass.
- **A fresh install resolves newer JAX/NumPy than the paper used**: expected. The submodules pull
  unpinned `jax`/`numpy`, so a plain install may land on jax 0.10.x / numpy 2.x. The core bounds
  are deliberately wide (`jax<0.12`, `numpy<3`) and the algorithm runs identically across the range:
  a fixed-seed re-run gives the same result on both jax 0.6.1/numpy 1.26 and jax 0.10.2/numpy 2.5,
  and both verify scripts pass (see the JAX/NumPy caveat in
  [REPRODUCIBILITY.md](../papers/es-hyperneat-quadtree-problem/REPRODUCIBILITY.md)). To match the
  published stack exactly, `pip install "jax==0.6.1" "jaxlib==0.6.1" "numpy<2"` before installing the
  package (this is what CI does).
- **macOS / Apple Silicon**: runs on CPU out of the box; keep `JAX_PLATFORMS=cpu` set to avoid the
  experimental Metal backend.

## Next

- Understand the codebase: [architecture.md](architecture.md)
- Run experiments / build your own: [writing-your-own-experiment.md](writing-your-own-experiment.md)
