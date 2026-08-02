"""HSHG ES-HyperNEAT benchmark runner (the over-discovery ablation).

Dispatches to the per-task experiment class by ``task.name``, runs the full HSHG
evolutionary loop, and builds a run dict matching the ``hshg_*`` result schema.

The HSHG substrate discovery over-discovers nodes, so the paper reports a 0% solve rate
on every task. This runner reproduces that faithful failure (including ``-Infinity``
fitness when every CPPN transform collapses).
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict

# Per-task experiment classes (each exposes HSHGESHyperNEATExperiment + a
# <Task>Config dataclass). Imported lazily inside run_single to keep import cost off the
# driver until an HSHG run is actually requested.
_EXPERIMENT_MODULES = {
    "xor": ("xor", "XORConfig"),
    "parity3": ("parity3", "Parity3Config"),
    "circle": ("circle", "CircleConfig"),
    "sine": ("sine", "SineConfig"),
    "cartpole": ("cartpole", "CartPoleConfig"),
}

# depth -> PUREPLES-style "version" letter (only used for the experiment's internal
# bookkeeping; max_depth is set explicitly from the config, so the letter is cosmetic).
_DEPTH_TO_VERSION = {2: "S", 3: "M", 4: "L"}


def run_single(task, depth: int, seed: int, config: Dict[str, Any],
               verbose: bool = True) -> Dict[str, Any]:
    """Run one (depth, seed) HSHG experiment for ``task``; return a frozen-schema dict.

    Static tasks (xor, parity3, circle, sine) use the ``solved_gen`` / ``total_time_s`` /
    ``final_*_nodes`` schema; CartPole uses the ``solve_generation`` / ``wall_time_s`` /
    ``final_metrics`` / ``final_mean_fitness`` / ``timestamp`` schema. The driver's
    ``fitness_threshold`` defines ``solved`` exactly as the benchmark drivers did.
    """
    if task.name not in _EXPERIMENT_MODULES:
        raise ValueError(f"no HSHG experiment for task {task.name!r}")
    mod_name, config_cls_name = _EXPERIMENT_MODULES[task.name]

    import importlib
    exp_mod = importlib.import_module(f".experiments.{mod_name}", __package__)
    Experiment = exp_mod.HSHGESHyperNEATExperiment
    Config = getattr(exp_mod, config_cls_name)

    threshold = config["fitness_threshold"]
    version = _DEPTH_TO_VERSION.get(depth, "S")

    exp_config = Config(
        pop_size=config["population_size"],
        max_generations=config["max_generations"],
        seed=seed,
        initial_depth=0,
        max_depth=depth,
    )
    experiment = Experiment(config=exp_config, version=version)

    start = time.time()
    _state, stats = experiment.run()
    elapsed = time.time() - start

    best_list = stats["fitness"]["best"]
    mean_list = stats["fitness"]["mean"]
    final = stats.get("final_metrics", {})
    best_fitness = float(final.get("fitness", max(best_list) if best_list else 0.0))
    solved = best_fitness >= threshold
    solve_gen = next((i for i, f in enumerate(best_list) if f >= threshold), None)
    generations_run = stats["generations"]

    if task.is_gym:
        # CartPole schema (benchmark_cartpole.run_hshg + main()'s timestamp).
        return {
            "implementation": "hshg",
            "depth": depth,
            "seed": seed,
            "generations_run": generations_run,
            "wall_time_s": elapsed,
            "best_fitness": best_fitness,
            "final_mean_fitness": float(mean_list[-1]) if mean_list else 0.0,
            "solved": solved,
            "solve_generation": solve_gen,
            "final_metrics": final,
            "fitness_history_best": [float(f) for f in best_list],
            "fitness_history_mean": [float(f) for f in mean_list],
            "timestamp": datetime.now().isoformat(),
        }

    # Static-task schema (benchmark_xor.run_hshg_xor).
    return {
        "implementation": "hshg",
        "depth": depth,
        "seed": seed,
        "generations_run": generations_run,
        "best_fitness": best_fitness,
        "solved": solved,
        "solved_gen": solve_gen,
        "total_time_s": elapsed,
        "final_discovered_nodes": final.get("discovered_nodes", 0),
        "final_substrate_nodes": final.get("substrate_nodes", 0),
        "final_cppn_nodes": final.get("cppn_nodes", 0),
        "final_cppn_connections": final.get("cppn_connections", 0),
        "fitness_history_best": [float(f) for f in best_list],
        "fitness_history_mean": [float(f) for f in mean_list],
    }
