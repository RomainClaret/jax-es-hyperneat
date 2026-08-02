"""Isolation guard: the package must run entirely from this repository.

Permanent regression test for standalone-ness. Imports the public API, exercises a
real generation, and asserts that (1) no ``geenns`` module was ever loaded (nothing
falls back to the research framework even where it is installed), and (2) every loaded
``jax_es_hyperneat`` module resolves to a file inside this checkout.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _assert_no_geenns():
    loaded = [m for m in sys.modules if m == "geenns" or m.startswith("geenns.")]
    assert not loaded, f"geenns modules were loaded: {loaded}"


def test_public_api_imports_without_geenns():
    from jax_es_hyperneat import JAXESHyperNEAT  # noqa: F401
    _assert_no_geenns()


def test_one_generation_runs_without_geenns():
    from jax_es_hyperneat import JAXESHyperNEAT
    from jax_es_hyperneat.tasks import TASKS

    task = TASKS["xor"]
    algo = JAXESHyperNEAT()
    problem = task.make_problem()
    cfg = task.build_config(algo, depth=2, population_size=50)
    state = algo.initialize(cfg, problem, seed=42)
    state, metrics = algo.run_generation(state, problem)
    assert float(metrics.best_fitness) >= 0.0
    _assert_no_geenns()


def test_loaded_modules_resolve_inside_repo():
    import jax_es_hyperneat  # noqa: F401
    import tensorneat  # noqa: F401

    for name, mod in list(sys.modules.items()):
        if not (name == "jax_es_hyperneat" or name.startswith("jax_es_hyperneat.")):
            continue
        f = getattr(mod, "__file__", None)
        if f is None:
            continue
        assert str(REPO_ROOT) in str(Path(f).resolve()), f"{name} resolved OUTSIDE the repo: {f}"

    # TensorNEAT must come from the pinned submodule when installed per the README.
    tn = Path(sys.modules["tensorneat"].__file__).resolve()
    assert "third_party" in str(tn) or "site-packages" not in str(tn), (
        f"tensorneat resolved to an unexpected location: {tn}"
    )
