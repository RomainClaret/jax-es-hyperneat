"""Unit tests for the JAX-ESHN runner's pure helpers (summarize + RunResult); no evolution."""
from jax_es_hyperneat.run_jax_eshn import RunResult, summarize


def _result(depth, solved, fitness, jit=10.0, tpg=1.0, positions=84, error=None):
    return {
        "depth": depth, "solved": solved, "fitness": fitness,
        "jit_compilation_time_s": jit, "time_per_gen_post_jit_s": tpg,
        "positions": positions, "error": error,
    }


def test_summarize_solve_rate_and_aggregates():
    results = [
        _result(2, True, 0.99), _result(2, True, 0.98), _result(2, False, 0.80),
        _result(3, False, 0.50),
    ]
    s = summarize(results)
    assert set(s) == {"d2", "d3"}
    assert s["d2"]["n_runs"] == 3
    assert s["d2"]["solve_rate"] == 2 / 3
    assert s["d2"]["max_fitness"] == 0.99
    assert abs(s["d2"]["avg_fitness"] - (0.99 + 0.98 + 0.80) / 3) < 1e-9
    assert s["d2"]["positions"] == 84
    assert s["d3"]["solve_rate"] == 0.0


def test_summarize_counts_errors_and_excludes_them_from_timing():
    results = [_result(2, False, 0.0, error="Boom"), _result(2, True, 0.99, jit=20.0)]
    s = summarize(results)
    assert s["d2"]["n_errors"] == 1
    # errored run is excluded from the JIT mean -> only the 20.0 contributes
    assert s["d2"]["avg_jit_time_s"] == 20.0


def test_summarize_empty():
    assert summarize([]) == {}


def test_runresult_to_dict_roundtrips():
    r = RunResult(
        implementation="tensorneat-eshyperneat", depth=2, seed=0, problem="xor",
        solved=True, fitness=0.999, generations=5, jit_compilation_time_s=50.0,
        time_per_gen_post_jit_s=15.0, total_evolution_time_s=75.0, total_time_s=125.0,
        solved_at_gen=5, positions=84,
    )
    d = r.to_dict()
    assert d["implementation"] == "tensorneat-eshyperneat"
    assert d["solved"] is True and d["solved_at_gen"] == 5 and d["positions"] == 84
    assert "timestamp" in d and d["error"] is None


def test_runresult_error_shape():
    r = RunResult(
        implementation="tensorneat-eshyperneat", depth=2, seed=0, problem="xor",
        solved=False, fitness=0.0, generations=0, jit_compilation_time_s=0.0,
        time_per_gen_post_jit_s=0.0, total_evolution_time_s=0.0, total_time_s=0.0,
        error="ValueError: boom",
    )
    assert r.solved is False and r.error.startswith("ValueError")
