# jax-es-hyperneat documentation

`jax_es_hyperneat` is a JAX/TensorNEAT implementation of **ES-HyperNEAT**, packaged to diagnose
why the algorithm's adaptive quadtree resists GPU batching. These pages are the researcher's guide;
the root [README](../README.md) has the short version.

## I want to…

| Goal | Start here |
|------|-----------|
| **Install** the package and its submodules | [installation.md](installation.md) |
| **Understand the code**: the layer this repository adds, the modules, a component diagram | [architecture.md](architecture.md) |
| **Run experiments** (CLI / Python API) and **build my own task** | [writing-your-own-experiment.md](writing-your-own-experiment.md) |
| **Reproduce the paper**: frozen-data verification or live re-runs | [reproducing-the-paper.md](reproducing-the-paper.md) |
| **Run the tests** and validate the CI workflows locally | [testing.md](testing.md) |

## The 60-second tour

```bash
# install (see installation.md for extras + troubleshooting)
git clone --recursive https://github.com/RomainClaret/jax-es-hyperneat.git && cd jax-es-hyperneat
pip install -e . && pip install -e third_party/tensorneat

# it works?
JAX_PLATFORMS=cpu python jax_es_hyperneat/tests/test_smoke.py        # -> SMOKE OK

# reproduce the paper from committed data (no GPU, stdlib only)
cd papers/es-hyperneat-quadtree-problem && python scripts/runners/verify_results.py  # -> 97/97 passed

# run one experiment
python -m jax_es_hyperneat.driver --task xor --impl jax-eshn --depths 2 --gens 60 --seeds 42
```

## Beyond these docs

- Per-table → script → data map and the full caveat list:
  [`papers/es-hyperneat-quadtree-problem/REPRODUCIBILITY.md`](../papers/es-hyperneat-quadtree-problem/REPRODUCIBILITY.md).
- The paper folder: [`papers/es-hyperneat-quadtree-problem/`](../papers/es-hyperneat-quadtree-problem/).
- The resolution of the bottleneck diagnosed here: the companion
  [EMR-HyperNEAT](https://github.com/RomainClaret/emr-hyperneat).
