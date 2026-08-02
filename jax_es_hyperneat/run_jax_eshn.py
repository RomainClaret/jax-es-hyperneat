"""JAX-ESHN benchmark runner.

Drives ``TensorNEATESHyperNEATOptimized`` (the lazy-quadtree ES-HyperNEAT) over a
task for ``max_generations`` with no early stopping, separating construction overhead
(JIT compilation + first generation) from post-JIT per-generation time. This is the
exact procedure the runners that produced the paper's JAX-ESHN numbers used.
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import jax

from .eshyperneat import TensorNEATESHyperNEATOptimized
from .tasks import DEPTH_POSITIONS
from .tasks.cartpole import run_gym_generation
from .gpu_metrics import GPUMonitor

IMPLEMENTATION = "tensorneat-eshyperneat"


@dataclass
class RunResult:
    implementation: str
    depth: int
    seed: int
    problem: str
    solved: bool
    fitness: float
    generations: int
    jit_compilation_time_s: float
    time_per_gen_post_jit_s: float
    total_evolution_time_s: float
    total_time_s: float
    solved_at_gen: Optional[int] = None
    positions: int = 0
    gpu_util_avg: float = 0.0
    gpu_util_max: int = 0
    vram_peak_mb: int = 0
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def clear_jax_caches():
    jax.clear_caches()
    gc.collect()


def run_single(task, depth: int, seed: int, config: Dict[str, Any], verbose: bool = True) -> RunResult:
    """Run one (depth, seed) JAX-ESHN experiment and return its RunResult."""
    positions = DEPTH_POSITIONS.get(depth, 0)
    threshold = config["fitness_threshold"]
    pop = config["population_size"]
    is_gym = task.is_gym

    def one_generation(algo, state, problem):
        if is_gym:
            state, best, _ = run_gym_generation(algo, state, pop)
            return state, best
        state, metrics = algo.run_generation(state, problem)
        return state, float(metrics.best_fitness)

    try:
        problem = task.make_problem()
        algo = TensorNEATESHyperNEATOptimized()
        algo_config = task.build_config(algo, depth, pop)
        state = algo.initialize(algo_config, problem, seed=seed)

        # Phase 1: construction overhead (XLA compilation + first generation).
        jit_start = time.time()
        with GPUMonitor() as monitor:
            state, first_best = one_generation(algo, state, problem)
        jit_time = time.time() - jit_start
        jit_metrics = monitor.get_metrics()
        if verbose:
            print(f"    JIT+gen1: {jit_time:.2f}s | gen 1 fitness: {first_best:.4f}")

        best_fitness = first_best
        solved_at_gen = 1 if first_best >= threshold else None

        # Phase 2: evolution, no early stopping.
        evolution_start = time.time()
        gen = 1
        with GPUMonitor() as monitor:
            while gen < config["max_generations"]:
                if time.time() - jit_start > config["timeout_per_run_s"]:
                    if verbose:
                        print(f"    TIMEOUT after {gen} generations")
                    break
                state, gen_best = one_generation(algo, state, problem)
                gen += 1
                best_fitness = max(best_fitness, gen_best)
                if best_fitness >= threshold and solved_at_gen is None:
                    solved_at_gen = gen
                if verbose and gen % 10 == 0:
                    tpg = (time.time() - evolution_start) / (gen - 1) if gen > 1 else 0
                    print(f"    gen {gen}/{config['max_generations']} | best {best_fitness:.4f} | {tpg:.2f}s/gen")
        evolution_time = time.time() - evolution_start
        total_time = time.time() - jit_start
        evo_metrics = monitor.get_metrics()
        post_jit_gens = gen - 1
        time_per_gen = evolution_time / post_jit_gens if post_jit_gens > 0 else 0.0

        return RunResult(
            implementation=IMPLEMENTATION, depth=depth, seed=seed, problem=task.name,
            solved=solved_at_gen is not None, fitness=best_fitness, generations=gen,
            solved_at_gen=solved_at_gen, jit_compilation_time_s=jit_time,
            time_per_gen_post_jit_s=time_per_gen, total_evolution_time_s=evolution_time,
            total_time_s=total_time, positions=positions,
            gpu_util_avg=evo_metrics.avg_utilization, gpu_util_max=evo_metrics.max_utilization,
            vram_peak_mb=max(jit_metrics.peak_vram_mb, evo_metrics.peak_vram_mb),
        )
    except Exception as e:  # noqa: BLE001 - record the error in the result, keep the sweep going
        import traceback
        traceback.print_exc()
        return RunResult(
            implementation=IMPLEMENTATION, depth=depth, seed=seed, problem=task.name,
            solved=False, fitness=0.0, generations=0, solved_at_gen=None,
            jit_compilation_time_s=0.0, time_per_gen_post_jit_s=0.0,
            total_evolution_time_s=0.0, total_time_s=0.0, positions=positions,
            error=f"{type(e).__name__}: {e}",
        )


def summarize(results) -> Dict[str, Any]:
    """Per-depth summary matching the published result schema."""
    by_depth: Dict[int, list] = {}
    for r in results:
        by_depth.setdefault(r["depth"], []).append(r)

    summary = {}
    for depth, runs in by_depth.items():
        solved = sum(1 for r in runs if r["solved"])
        jit = [r["jit_compilation_time_s"] for r in runs if not r.get("error")]
        post = [r["time_per_gen_post_jit_s"] for r in runs
                if not r.get("error") and r["time_per_gen_post_jit_s"] > 0]
        fit = [r["fitness"] for r in runs]
        summary[f"d{depth}"] = {
            "depth": depth,
            "positions": runs[0]["positions"] if runs else 0,
            "solve_rate": solved / len(runs) if runs else 0,
            "n_runs": len(runs),
            "n_errors": sum(1 for r in runs if r.get("error")),
            "avg_jit_time_s": float(np.mean(jit)) if jit else 0,
            "std_jit_time_s": float(np.std(jit)) if jit else 0,
            "avg_time_per_gen_post_jit_s": float(np.mean(post)) if post else 0,
            "std_time_per_gen_post_jit_s": float(np.std(post)) if post else 0,
            "avg_fitness": float(np.mean(fit)) if fit else 0,
            "max_fitness": float(max(fit)) if fit else 0,
        }
    return summary
