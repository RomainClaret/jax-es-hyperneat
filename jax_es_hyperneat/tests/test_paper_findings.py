"""Paper-finding validation from the committed per-seed result JSON (fast, no evolution).

Each test encodes a published claim of the paper and asserts it against the frozen data
(N=30 per condition). Exact solve counts are taken verbatim from the paper's own
``verify_results.py`` and double as regression guards against accidental data corruption.
All tests are stdlib-only (no jax/scipy), so they run in the fast CI gate on every push.
"""
import statistics

import pytest


# ---------------------------------------------------------------------------- #
# Solve rates (out of 30 seeds), straight from verify_results.py's gold table
# ---------------------------------------------------------------------------- #

@pytest.mark.parametrize("task,depth,expected", [
    ("sine", 2, 29), ("sine", 3, 30), ("sine", 4, 30),
    ("circle", 2, 0), ("circle", 3, 0), ("circle", 4, 0),
    ("parity3", 2, 4), ("parity3", 3, 7), ("parity3", 4, 5),
    ("cartpole", 2, 30), ("cartpole", 3, 30), ("cartpole", 4, 30),
])
def test_jax_eshn_solve_rate(load_runs, solved_count, task, depth, expected):
    runs = load_runs("jax-eshn", task, depth)
    assert len(runs) == 30
    assert solved_count(runs) == expected


@pytest.mark.parametrize("task,depth,expected", [
    ("sine", 2, 0), ("sine", 3, 0), ("sine", 4, 0),
    ("circle", 2, 0), ("circle", 3, 0), ("circle", 4, 0),
    ("parity3", 2, 23), ("parity3", 3, 30), ("parity3", 4, 29),
    ("cartpole", 2, 30), ("cartpole", 3, 30), ("cartpole", 4, 30),
])
def test_pureples_solve_rate(load_runs, solved_count, task, depth, expected):
    runs = load_runs("pureples", task, depth)
    assert len(runs) == 30
    assert solved_count(runs) == expected


@pytest.mark.parametrize("task,depth", [("xor", 2), ("xor", 3), ("sine", 2), ("parity3", 2)])
def test_hshg_never_solves(load_runs, solved_count, task, depth):
    runs = load_runs("hshg", task, depth)
    assert len(runs) == 30
    assert solved_count(runs) == 0  # over-discovery -> 0% solve everywhere (Table 7)


# ---------------------------------------------------------------------------- #
# Headline qualitative findings
# ---------------------------------------------------------------------------- #

def test_sine_inversion(load_runs, solved_count):
    """JAX-ESHN solves sine where PUREPLES cannot (a solve-rate inversion)."""
    jax = load_runs("jax-eshn", "sine", 2)
    pp = load_runs("pureples", "sine", 2)
    assert solved_count(jax) >= 29          # jax-eshn solves
    assert solved_count(pp) == 0            # pureples fails
    assert max(r["fitness"] for r in pp) < 0.76  # plateaus at the ~0.75 ceiling


def test_parity3_inversion(load_runs, solved_count):
    """PUREPLES solves Parity-3 where JAX-ESHN rarely does (the opposite inversion)."""
    pp = load_runs("pureples", "parity3", 3)
    jax = load_runs("jax-eshn", "parity3", 3)
    assert solved_count(pp) >= 23           # pureples solves (76%+)
    assert solved_count(jax) <= 8           # jax-eshn rarely (13-23%)
    assert solved_count(pp) > solved_count(jax)


def test_circle_agreement_on_failure(load_runs, solved_count):
    """Both implementations fail circle (a shared two-layer ceiling, not a library artifact)."""
    jax = load_runs("jax-eshn", "circle", 2)
    pp = load_runs("pureples", "circle", 2)
    assert solved_count(jax) == 0 and solved_count(pp) == 0
    # both plateau just below the 0.975 threshold, around 0.88-0.90
    assert 0.85 < max(r["fitness"] for r in jax) < 0.975
    assert 0.85 < max(r["fitness"] for r in pp) < 0.975


def test_xor_positive_control(load_runs, solved_count):
    """XOR is solved by the baseline at both populations (positive control)."""
    for pop in (50, 150):
        runs = load_runs("pureples", "xor", 2, pop=pop)
        assert len(runs) == 30 and solved_count(runs) == 30


# ---------------------------------------------------------------------------- #
# Configuration + scaling findings
# ---------------------------------------------------------------------------- #

@pytest.mark.parametrize("task,depth,threshold", [
    ("sine", 2, 0.95), ("circle", 2, 0.975), ("parity3", 2, 0.975), ("cartpole", 2, 0.95),
])
def test_fitness_threshold_recorded(results_dir, task, depth, threshold):
    import json
    fname = ("jax_eshn_%s_results.json" % task) if task != "cartpole" else None
    if task == "cartpole":
        meta = json.load(open(results_dir / "cartpole_d2" / "jax_eshn_cartpole_results.json"))["metadata"]
    else:
        meta = json.load(open(results_dir / fname))["metadata"]
    assert meta["config"]["fitness_threshold"] == pytest.approx(threshold, abs=1e-6)


@pytest.mark.parametrize("task,expected_ratio", [("sine", 5.0), ("circle", 4.8), ("parity3", 4.8)])
def test_jax_eshn_depth_scaling_is_shallow(load_runs, task, expected_ratio):
    """JAX-ESHN per-generation cost scales only ~5x from d2->d4 (vs the baseline's ~65x)."""
    def mean_tpg(depth):
        return statistics.mean(r["time_per_gen"] for r in load_runs("jax-eshn", task, depth))
    ratio = mean_tpg(4) / mean_tpg(2)
    assert ratio == pytest.approx(expected_ratio, abs=0.2)


def test_parity3_reaches_perfect_fitness_when_solved(load_runs):
    """When JAX-ESHN does crack Parity-3, it solves it perfectly (fitness 1.0)."""
    solved = [r for r in load_runs("jax-eshn", "parity3", 3) if r["solved"]]
    assert solved  # 7/30 at d3
    assert all(r["fitness"] >= 0.975 for r in solved)
    assert max(r["fitness"] for r in solved) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_cartpole_solves_within_a_few_generations(load_runs, depth):
    runs = [r for r in load_runs("jax-eshn", "cartpole", depth) if r["solved"]]
    assert len(runs) == 30
    assert all(r["solved_at_gen"] is not None and r["solved_at_gen"] <= 3 for r in runs)


@pytest.mark.parametrize("task,depth", [("xor", 2), ("xor", 3), ("sine", 2), ("parity3", 2)])
def test_hshg_plateaus_far_below_threshold(load_runs, task, depth):
    """Over-discovery leaves HSHG stuck on a low plateau (max fitness <= ~0.77)."""
    runs = load_runs("hshg", task, depth)
    assert max(r["fitness"] for r in runs) <= 0.78


def test_circle_unsolved_fitness_band(load_runs):
    """JAX-ESHN circle plateaus tightly around ~0.87 (the two-layer ceiling), never at threshold."""
    runs = load_runs("jax-eshn", "circle", 2)
    assert all(0.84 <= r["fitness"] <= 0.89 for r in runs)
