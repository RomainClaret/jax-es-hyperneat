"""Unit tests for the benchmark driver CLI (no evolution; a stub runner exercises the loop)."""
import json

import pytest

from jax_es_hyperneat import driver
from jax_es_hyperneat.tasks import TASKS

CONFIG = {"population_size": 10, "max_generations": 1, "fitness_threshold": 0.9,
          "timeout_per_run_s": 60}


@pytest.mark.parametrize("spec,expected", [
    ("0-4", [0, 1, 2, 3, 4]),
    ("42,43,44", [42, 43, 44]),
    ("7", [7]),
    ("0-2,5", [0, 1, 2, 5]),
    (" 1 , 3 ", [1, 3]),
    ("", []),
])
def test_parse_seeds(spec, expected):
    assert driver.parse_seeds(spec) == expected


def test_impl_prefix_map():
    assert driver.IMPL_PREFIX == {"jax-eshn": "jax_eshn", "pureples": "pureples", "hshg": "hshg"}


def test_select_runner_jax_eshn():
    from jax_es_hyperneat.run_jax_eshn import run_single
    assert driver.select_runner("jax-eshn") is run_single


def test_select_runner_unknown_raises():
    with pytest.raises(ValueError):
        driver.select_runner("nope")


def test_select_runner_pureples_and_hshg():
    pytest.importorskip("pureples")
    pytest.importorskip("neat")
    from jax_es_hyperneat.baseline.pureples_harness import run_single as pp
    from jax_es_hyperneat.hshg.run_hshg import run_single as hh
    assert driver.select_runner("pureples") is pp
    assert driver.select_runner("hshg") is hh


def _fake_jax_runner(task, depth, seed, config, verbose=False):
    return {"depth": depth, "seed": seed, "problem": task.name,
            "solved": seed % 2 == 0, "fitness": 0.9, "positions": 84,
            "jit_compilation_time_s": 1.0, "time_per_gen_post_jit_s": 0.5,
            "solved_at_gen": 3 if seed % 2 == 0 else None, "error": None}


def _fake_pureples_runner(task, depth, seed, config, verbose=False):
    return {"implementation": "pureples", "depth": depth, "seed": seed,
            "solved": False, "best_fitness": 0.7, "solved_gen": None}


def test_run_benchmark_jax_eshn_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "select_runner", lambda impl: _fake_jax_runner)
    driver.run_benchmark(TASKS["xor"], "jax-eshn", CONFIG, depths=[1], seeds=[0, 1],
                         out_dir=tmp_path, resume=False, verbose=False)
    out = json.loads((tmp_path / "jax_eshn_xor_results.json").read_text())
    assert set(out) == {"metadata", "summary", "results"}
    assert len(out["results"]) == 2
    assert out["summary"]["d1"]["solve_rate"] == 0.5  # seed0 solved, seed1 not
    assert out["summary"]["d1"]["n_runs"] == 2 and out["summary"]["d1"]["positions"] == 84
    assert (tmp_path / "jax_eshn_xor_checkpoint.json").exists()


def test_run_benchmark_pureples_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "select_runner", lambda impl: _fake_pureples_runner)
    driver.run_benchmark(TASKS["sine"], "pureples", CONFIG, depths=[2], seeds=[0],
                         out_dir=tmp_path, resume=False, verbose=False)
    out = json.loads((tmp_path / "pureples_sine_results.json").read_text())
    assert set(out) == {"metadata", "runs"}  # non-jax impls use the {metadata, runs} schema
    assert len(out["runs"]) == 1 and out["runs"][0]["best_fitness"] == 0.7


def test_run_benchmark_resume_skips_completed(tmp_path, monkeypatch):
    calls = {"n": 0}

    def counting_runner(task, depth, seed, config, verbose=False):
        calls["n"] += 1
        return _fake_jax_runner(task, depth, seed, config)

    monkeypatch.setattr(driver, "select_runner", lambda impl: counting_runner)
    driver.run_benchmark(TASKS["xor"], "jax-eshn", CONFIG, depths=[1], seeds=[0, 1],
                         out_dir=tmp_path, resume=False, verbose=False)
    assert calls["n"] == 2
    # second pass with resume=True: both run_ids already in the checkpoint -> no new calls
    driver.run_benchmark(TASKS["xor"], "jax-eshn", CONFIG, depths=[1], seeds=[0, 1],
                         out_dir=tmp_path, resume=True, verbose=False)
    assert calls["n"] == 2


def test_main_cli_end_to_end(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(driver, "select_runner", lambda impl: _fake_jax_runner)
    argv = ["driver", "--task", "xor", "--impl", "jax-eshn", "--depths", "1",
            "--seeds", "0", "--gens", "1", "--out", str(tmp_path), "--quiet"]
    monkeypatch.setattr(sys, "argv", argv)
    driver.main()  # argparse -> config -> run_benchmark, no evolution (stub runner)
    out = json.loads((tmp_path / "jax_eshn_xor_results.json").read_text())
    assert out["metadata"]["task"] == "xor" and len(out["results"]) == 1
