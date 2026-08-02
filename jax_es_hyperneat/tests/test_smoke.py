"""Smoke test: the standalone JAX-ESHN package runs evolution end-to-end on XOR.

Builds the XOR substrate, runs a few generations, and asserts the algorithm produces
finite, in-range fitness. Validates the standalone port (import + config + initialize +
run_generation) without any geenns dependency.

Run directly:
    JAX_PLATFORMS=cpu python jax_es_hyperneat/tests/test_smoke.py
(requires `pip install -e .` and `pip install -e third_party/tensorneat`) or via pytest.
"""
import math

from jax_es_hyperneat import JAXESHyperNEAT
from jax_es_hyperneat.tasks import TASKS


def _run(n_generations=3, seed=42, depth=2, pop=50):
    task = TASKS["xor"]
    algo = JAXESHyperNEAT()
    problem = task.make_problem()
    cfg = task.build_config(algo, depth, pop)
    state = algo.initialize(cfg, problem, seed=seed)
    best = None
    for _ in range(n_generations):
        state, metrics = algo.run_generation(state, problem)
        best = float(metrics.best_fitness)
    return best


def test_smoke_xor_runs():
    best = _run()
    assert best is not None and math.isfinite(best) and 0.0 <= best <= 1.0


if __name__ == "__main__":
    b = _run()
    print(f"XOR smoke: best_fitness={b:.4f}")
    assert b is not None and math.isfinite(b), "smoke run did not produce finite fitness"
    print("SMOKE OK")
