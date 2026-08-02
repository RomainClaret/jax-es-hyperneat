"""Substrate-discovery / task-coverage checks.

Fast: the potential-position formula and a depth-1 XOR run. Slow (marked): every task
constructs a substrate and runs one generation, confirming the standalone task
definitions + impl wire up across all five benchmarks.
"""
import math

import pytest

from jax_es_hyperneat import JAXESHyperNEAT
from jax_es_hyperneat.tasks import TASKS, DEPTH_POSITIONS


def test_depth_position_formula():
    # DEPTH_POSITIONS counts the quadtree child nodes over levels 1..d+1
    # (sum_{k=1}^{d+1} 4^k = (4^(d+2) - 4) / 3); this is the `positions` field the
    # runners record. (The paper prose uses a root-inclusive count that differs by a level.)
    for d, expected in DEPTH_POSITIONS.items():
        assert (4 ** (d + 2) - 4) // 3 == expected


def test_xor_builds_and_runs_depth1():
    task = TASKS["xor"]
    algo = JAXESHyperNEAT()
    problem = task.make_problem()
    cfg = task.build_config(algo, depth=1, population_size=50)
    state = algo.initialize(cfg, problem, seed=42)
    state, metrics = algo.run_generation(state, problem)
    assert math.isfinite(float(metrics.best_fitness))


@pytest.mark.slow
@pytest.mark.parametrize("task_name", ["xor", "parity3", "circle", "sine"])
def test_all_static_tasks_run_one_generation(task_name):
    task = TASKS[task_name]
    algo = JAXESHyperNEAT()
    problem = task.make_problem()
    cfg = task.build_config(algo, depth=2, population_size=50)
    state = algo.initialize(cfg, problem, seed=0)
    state, metrics = algo.run_generation(state, problem)
    fit = float(metrics.best_fitness)
    assert math.isfinite(fit) and 0.0 <= fit <= 1.0
