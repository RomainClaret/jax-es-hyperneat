"""Shared pytest fixtures + the committed-result data loader.

The paper-finding tests read the per-seed result JSON committed under
``papers/es-hyperneat-quadtree-problem/results/`` and assert the paper's published numbers.
The two result schemas are normalized here: jax-eshn files are ``{metadata, summary, results}``
with ``solved``/``fitness``/``solved_at_gen``; pureples and hshg files are ``{metadata, runs}``
with ``solved``/``best_fitness``/``solved_gen``.
"""
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "papers" / "es-hyperneat-quadtree-problem"
RESULTS_DIR = PAPER_DIR / "results"


def _read(path: Path):
    with open(path) as f:
        return json.load(f)


def _runs_array(d):
    return d.get("results") or d.get("runs") or []


def _normalize(run):
    return {
        "solved": bool(run.get("solved")),
        "fitness": float(run.get("fitness", run.get("best_fitness", 0.0))),
        "solved_at_gen": run.get("solved_at_gen", run.get("solved_gen", run.get("solve_generation"))),
        "depth": run.get("depth"),
        "time_per_gen": run.get("time_per_gen_post_jit_s"),  # jax-eshn only
    }


def _load_runs(impl: str, task: str, depth: int, pop=None):
    """Return normalized run dicts for (impl, task, depth) from the committed JSON."""
    if impl == "jax-eshn":
        if task == "cartpole":
            path = RESULTS_DIR / f"cartpole_d{depth}" / "jax_eshn_cartpole_results.json"
        else:
            path = RESULTS_DIR / f"jax_eshn_{task}_results.json"
    elif impl == "pureples":
        if task == "xor":
            path = RESULTS_DIR / f"pureples_xor_d{depth}_pop{pop or 150}_results.json"
        else:
            path = RESULTS_DIR / f"pureples_{task}_d{depth}_results.json"
    elif impl == "hshg":
        suffix = f"_pop{pop}" if pop else ""
        path = RESULTS_DIR / f"hshg_{task}_d{depth}{suffix}_results.json"
    else:
        raise ValueError(f"unknown impl {impl!r}")
    runs = [r for r in _runs_array(_read(path)) if r.get("depth") == depth]
    return [_normalize(r) for r in runs]


def _solved_count(runs):
    return sum(1 for r in runs if r["solved"])


@pytest.fixture(scope="session")
def results_dir():
    if not RESULTS_DIR.exists():
        pytest.skip("committed results/ directory not present")
    return RESULTS_DIR


@pytest.fixture(scope="session")
def paper_dir():
    if not PAPER_DIR.exists():
        pytest.skip("paper directory not present")
    return PAPER_DIR


@pytest.fixture
def load_runs(results_dir):
    """Callable: load_runs(impl, task, depth, pop=None) -> [normalized run dicts]."""
    return _load_runs


@pytest.fixture
def solved_count():
    """Callable: solved_count(runs) -> int."""
    return _solved_count
