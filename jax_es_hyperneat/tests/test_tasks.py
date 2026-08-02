"""Unit tests for the standalone task definitions (no evolution; fast).

These guard the per-task data, coordinates, thresholds, and the shared ES-HyperNEAT
hyperparameters against regression, the inputs that the paper's experiments depend on.
"""
import math

import numpy as np
import pytest

from jax_es_hyperneat.tasks import TASKS, DEPTH_POSITIONS, ES_PARAMS


def test_registry_has_all_five_tasks():
    assert set(TASKS) == {"xor", "parity3", "circle", "sine", "cartpole"}


def test_task_thresholds_match_paper():
    expected = {"xor": 0.98, "parity3": 0.975, "circle": 0.975, "sine": 0.95, "cartpole": 0.95}
    assert {name: t.fitness_threshold for name, t in TASKS.items()} == expected


def test_task_geometry():
    # input_coords count == problem.input_size; single output node.
    for name, task in TASKS.items():
        problem = task.make_problem()
        assert len(task.input_coords) == problem.input_size, name
        assert len(task.output_coords) == 1 == problem.output_size, name
        assert task.output_coords == [(0.0, 1.0)], name
    assert TASKS["cartpole"].is_gym and not TASKS["xor"].is_gym


def test_xor_truth_table():
    data = TASKS["xor"].make_problem().get_data()
    assert len(data) == 4
    for inp, target in data:
        a, b, bias = inp
        assert bias == 1.0
        assert float(target[0]) == float(int(a) ^ int(b))


def test_parity3_is_odd_parity():
    data = TASKS["parity3"].make_problem().get_data()
    assert len(data) == 8
    for inp, target in data:
        a, b, c, bias = inp
        assert bias == 1.0
        assert float(target[0]) == float((int(a) + int(b) + int(c)) % 2)


def test_sine_target_is_rescaled_to_unit_interval():
    # JAX-ESHN side regresses (sin(pi*x)+1)/2 in [0, 1] (for the sigmoid output).
    data = TASKS["sine"].make_problem().get_data()
    assert len(data) == 20
    xs = [float(inp[0]) for inp, _ in data]
    ys = [float(t[0]) for _, t in data]
    assert min(xs) == pytest.approx(-1.0) and max(xs) == pytest.approx(1.0)
    assert all(0.0 <= y <= 1.0 for y in ys)
    # spot-check the transform at x=0.5 -> (sin(pi/2)+1)/2 = 1.0
    i = xs.index(min(xs, key=lambda x: abs(x - 0.5)))
    assert ys[i] == pytest.approx((math.sin(math.pi * xs[i]) + 1.0) / 2.0, abs=1e-6)


def test_circle_labels_inside_radius_half():
    data = TASKS["circle"].make_problem().get_data()
    assert len(data) == 100
    for inp, target in data:
        x, y, bias = inp
        assert bias == 1.0
        assert float(target[0]) == float((x * x + y * y) < 0.25)


def test_circle_dataset_is_seeded_and_deterministic():
    a = TASKS["circle"].make_problem().get_data()
    b = TASKS["circle"].make_problem().get_data()
    assert np.allclose([i for i, _ in a], [i for i, _ in b])  # RandomState(42) -> identical


def test_es_params_are_code_true_values():
    assert ES_PARAMS == dict(
        initial_depth=0, variance_threshold=0.03, division_threshold=0.5,
        band_threshold=0.3, max_weight=8.0, iteration_level=1,
    )


def test_depth_positions_closed_form():
    # cumulative quadtree child nodes over levels 1..d+1 = (4^(d+2) - 4) / 3
    for d, expected in DEPTH_POSITIONS.items():
        assert (4 ** (d + 2) - 4) // 3 == expected


def test_build_config_structure():
    task = TASKS["xor"]

    class _Algo:
        def create_config(self, cfg):  # capture without instantiating the real algorithm
            return cfg

    cfg = task.build_config(_Algo(), depth=3, population_size=150)
    es = cfg["algorithm_params"]["eshyperneat"]
    assert es["population_size"] == 150
    assert es["es_hyperneat"]["max_depth"] == 3
    assert es["es_hyperneat"]["max_weight"] == 8.0 and es["es_hyperneat"]["band_threshold"] == 0.3
    assert es["substrate"]["output_activation"] == "sigmoid"
    assert es["substrate"]["hidden_activation"] == "tanh"
    assert es["substrate"]["input_coords"] == task.input_coords


def test_pureples_sine_uses_raw_target_not_rescaled():
    # The paper's sine asymmetry: PUREPLES regresses RAW sin(pi*x) in [-1, 1], not the
    # rescaled JAX target. Needs pureples+neat (skipped on the fast CI job).
    pytest.importorskip("pureples")
    pytest.importorskip("neat")
    from jax_es_hyperneat.baseline.pureples_harness import _pureples_data
    _, targets = _pureples_data(TASKS["sine"])
    assert min(targets) < -0.9 and max(targets) > 0.9  # spans [-1, 1], unlike the JAX [0, 1] target


def test_all_tasks_have_default_depths():
    for name, task in TASKS.items():
        assert task.default_depths and all(d >= 1 for d in task.default_depths), name


def test_cartpole_constants():
    from jax_es_hyperneat.tasks import cartpole as cp
    assert (cp.CART_POS_SCALE, cp.CART_VEL_SCALE) == (2.4, 3.0)
    assert (cp.POLE_ANGLE_SCALE, cp.POLE_VEL_SCALE) == (0.2095, 3.0)
    assert cp.NUM_EPISODES == 5 and cp.MAX_STEPS == 500


def test_cartpole_get_data_is_a_dummy_pair():
    # CartPole fitness comes from a gym episode loop, so get_data() returns a single zero
    # placeholder pair (5 inputs incl. bias, 1 output) just to initialize the pipeline.
    from jax_es_hyperneat.tasks.cartpole import CartPoleProblem
    data = CartPoleProblem().get_data()
    assert len(data) == 1
    inp, target = data[0]
    assert inp.shape == (5,) and target.shape == (1,)
    assert inp.dtype == np.float32 and target.dtype == np.float32
    assert not np.any(inp) and not np.any(target)  # all zeros


@pytest.mark.parametrize("task_name", ["xor", "parity3", "circle"])
def test_pureples_data_matches_jax_for_non_sine_tasks(task_name):
    # Only sine differs between the JAX and PUREPLES sides; the other static tasks share data.
    pytest.importorskip("pureples")
    pytest.importorskip("neat")
    from jax_es_hyperneat.baseline.pureples_harness import _pureples_data
    pp_inputs, pp_targets = _pureples_data(TASKS[task_name])
    jax_data = TASKS[task_name].make_problem().get_data()
    assert len(pp_inputs) == len(jax_data)
    for (pp_in, pp_t), (jx_in, jx_t) in zip(zip(pp_inputs, pp_targets), jax_data):
        assert np.allclose(pp_in, np.asarray(jx_in))
        assert float(pp_t) == pytest.approx(float(jx_t[0]))
