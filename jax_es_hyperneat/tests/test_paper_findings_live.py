"""Live paper-finding tests: run a small version of each experiment and confirm the claim.

Marked ``slow`` because they JIT-compile and evolve (tens of seconds per generation on CPU);
they run in the dedicated ``paper-findings`` workflow, not the per-push gate. These complement
the frozen-data tests in ``test_paper_findings.py`` by exercising the actual algorithm.
"""
import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest

from jax_es_hyperneat.tasks import TASKS
from jax_es_hyperneat.run_jax_eshn import run_single as jax_run

pytestmark = pytest.mark.slow

_CFG = lambda gens, thr, pop=150: {
    "population_size": pop, "max_generations": gens,
    "fitness_threshold": thr, "timeout_per_run_s": 86400,
}


def test_jax_eshn_solves_xor():
    r = jax_run(TASKS["xor"], 2, 42, _CFG(15, 0.98), verbose=False)
    assert r.solved and r.fitness >= 0.98


def test_jax_eshn_solves_sine():
    r = jax_run(TASKS["sine"], 2, 0, _CFG(8, 0.95), verbose=False)
    assert r.solved and r.fitness >= 0.95


def test_jax_eshn_fails_circle():
    # two-layer ceiling: never solves, plateaus below threshold around ~0.88
    r = jax_run(TASKS["circle"], 2, 0, _CFG(12, 0.975), verbose=False)
    assert not r.solved and 0.80 < r.fitness < 0.975


def test_jax_eshn_is_deterministic():
    a = jax_run(TASKS["sine"], 2, 0, _CFG(5, 0.95), verbose=False)
    b = jax_run(TASKS["sine"], 2, 0, _CFG(5, 0.95), verbose=False)
    assert (a.solved, a.solved_at_gen, a.fitness) == (b.solved, b.solved_at_gen, b.fitness)


def test_pureples_fails_sine():
    pytest.importorskip("pureples")
    pytest.importorskip("neat")
    from jax_es_hyperneat.baseline.pureples_harness import run_single as pp_run
    r = pp_run(TASKS["sine"], 2, 0, _CFG(8, 0.95), verbose=False)
    assert not r["solved"] and r["best_fitness"] <= 0.76  # sigmoid output can't fit raw sin


def test_pureples_fails_circle():
    # Both implementations fail circle -> a shared two-layer ceiling, not a library artifact.
    # JAX side is live in test_jax_eshn_fails_circle; this completes the agreement on the
    # PUREPLES side. Reliable: frozen data is 0/30 with the plateau (~0.86) reached by gen 0,
    # so a few generations are ample to confirm it never crosses the 0.975 threshold.
    pytest.importorskip("pureples")
    pytest.importorskip("neat")
    from jax_es_hyperneat.baseline.pureples_harness import run_single as pp_run
    r = pp_run(TASKS["circle"], 2, 0, _CFG(5, 0.975), verbose=False)
    assert not r["solved"] and r["best_fitness"] < 0.975


def test_hshg_fails_xor():
    pytest.importorskip("neat")
    from jax_es_hyperneat.hshg.run_hshg import run_single as hshg_run
    r = hshg_run(TASKS["xor"], 2, 0, _CFG(5, 0.98, pop=30), verbose=False)
    assert not r["solved"]  # over-discovery -> bloated substrate -> 0% solve
