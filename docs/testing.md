# Testing

The suite has **175 tests** in `jax_es_hyperneat/tests/`, split into a fast gate and a slow live
tier by a single `slow` marker.

```bash
pip install -e ".[dev]"   # pytest>=7.0  (or: pip install pytest)
```

## The two tiers

| Tier | Command | Count | What it covers |
|------|---------|-------|----------------|
| **Fast gate** | `JAX_PLATFORMS=cpu pytest jax_es_hyperneat/tests -m "not slow"` | **164** | unit tests, frozen-data paper findings, the `verify_*.py` reproduction (via subprocess), the standalone isolation guard (asserts no parent-framework imports), and a couple of tiny live runs. ~2–4 min. |
| **Live** | `JAX_PLATFORMS=cpu pytest jax_es_hyperneat/tests -m slow` | **11** | actually evolves small versions of the paper experiments and asserts the published findings. Minutes per test. |
| All | `JAX_PLATFORMS=cpu pytest jax_es_hyperneat/tests` | 175 | both tiers. |
| Smoke only | `JAX_PLATFORMS=cpu python jax_es_hyperneat/tests/test_smoke.py` | 1 | 3-generation XOR evolution; prints `SMOKE OK`. |

The `slow` marker is declared in `pyproject.toml`:
`"slow: end-to-end runs that JIT-compile and evolve (minutes); excluded from the fast CI gate"`.

## What needs the baseline extra

Tests that touch PUREPLES / neat-python / gymnasium guard with `pytest.importorskip(...)`, so they
**skip** (not fail) without the `[baseline]` extra; the fast gate stays green on a minimal install.
Install `pip install -e ".[baseline]"` (and `pip install -e third_party/pureples`) to actually run
them:

- Fast gate: a couple of pureples-vs-jax data tests (`test_tasks.py`) and one driver test skip.
- Live tier: `test_pureples_fails_sine`, `test_pureples_fails_circle` (need pureples+neat),
  `test_hshg_fails_xor` (needs neat).

The 11 slow tests are the 7 in `test_paper_findings_live.py` (jax solves xor/sine, fails circle, is
deterministic; pureples fails sine and circle; hshg fails xor) plus the 4
`test_substrate_building.py::test_all_static_tasks_run_one_generation[xor|parity3|circle|sine]`.

## Continuous integration

Three GitHub Actions workflows in `.github/workflows/`:

| Workflow | Trigger | Runs |
|----------|---------|------|
| `ci.yml` | `push` to `main` + `pull_request` | install (jax/tensorneat, no baseline) → smoke → fast gate (`-m "not slow"`) → the paper `verify_*.py` reproduction. |
| `quick-experiments.yml` | `workflow_dispatch` | the `[baseline]` install + a ~15-min live subset (`-m slow -k "one_generation or hshg_fails_xor or pureples_fails_sine"`). |
| `paper-findings.yml` | `workflow_dispatch` + weekly cron | the `[baseline]` install + the full live tier (`-m slow`), `timeout-minutes: 90`. |

## Validating workflows locally with `act`

`ci.yml` triggers on `push`, so:

```bash
act push -W .github/workflows/ci.yml \
  --container-architecture linux/amd64 \
  -P ubuntu-latest=catthehacker/ubuntu:act-latest --rm
```

The two experiment workflows have **no `push` trigger** (they are `workflow_dispatch`/`schedule`),
so `act push` would silently no-op (exit 0 in ~1 s). Use the dispatch event:

```bash
act workflow_dispatch -W .github/workflows/quick-experiments.yml \
  --container-architecture linux/amd64 \
  -P ubuntu-latest=catthehacker/ubuntu:act-latest --rm

act workflow_dispatch -W .github/workflows/paper-findings.yml \
  --container-architecture linux/amd64 \
  -P ubuntu-latest=catthehacker/ubuntu:act-latest --rm
```

Under `act`'s x86 emulation these run modestly slower than native: `quick-experiments` ≈ 15–20 min, the
full `paper-findings` ≈ 45–80 min (hence the 90-min job timeout). All three are validated green on
`linux/amd64`.

## Next

- Reproduce the paper without evolving anything: [reproducing-the-paper.md](reproducing-the-paper.md)
- Understand what the tests exercise: [architecture.md](architecture.md)
