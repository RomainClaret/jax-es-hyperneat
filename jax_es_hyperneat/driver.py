"""Unified benchmark driver for JAX-ESHN, the PUREPLES baseline, and HSHG.

One entry point drives all three implementations over all five tasks, with checkpoint/resume.

Examples
--------
    python -m jax_es_hyperneat.driver --task xor   --impl jax-eshn --depths 2 --pop 150 --gens 30 --seeds 42-44
    python -m jax_es_hyperneat.driver --task sine  --impl jax-eshn --depths 2 3 4 --seeds 0-29
    python -m jax_es_hyperneat.driver --task sine  --impl pureples --depths 2 --seeds 0-4
    python -m jax_es_hyperneat.driver --task xor   --impl hshg     --depths 2 --pop 50 --gens 50 --seeds 0-4
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

IMPL_PREFIX = {"jax-eshn": "jax_eshn", "pureples": "pureples", "hshg": "hshg"}


def parse_seeds(spec: str) -> List[int]:
    """Parse ``"0-29"`` or ``"42,43,44"`` or ``"7"`` into a list of ints."""
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def select_runner(impl: str):
    """Return the ``run_single`` callable for an implementation (imported lazily)."""
    if impl == "jax-eshn":
        from .run_jax_eshn import run_single
        return run_single
    if impl == "pureples":
        from .baseline.pureples_harness import run_single
        return run_single
    if impl == "hshg":
        from .hshg.run_hshg import run_single
        return run_single
    raise ValueError(f"unknown impl: {impl}")


def _as_dict(result) -> Dict[str, Any]:
    return result.to_dict() if hasattr(result, "to_dict") else dict(result)


def run_benchmark(task, impl: str, config: Dict[str, Any], depths: List[int],
                  seeds: List[int], out_dir: Path, resume: bool, verbose: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = IMPL_PREFIX[impl]
    checkpoint_file = out_dir / f"{prefix}_{task.name}_checkpoint.json"
    results_file = out_dir / f"{prefix}_{task.name}_results.json"

    completed: set = set()
    results: List[Dict[str, Any]] = []
    if resume and checkpoint_file.exists():
        data = json.loads(checkpoint_file.read_text())
        completed = set(data.get("completed_runs", []))
        results = data.get("results", [])
        print(f"[RESUME] {len(completed)} runs already completed")

    runner = select_runner(impl)
    total = len(depths) * len(seeds)
    done = len(completed)

    print("=" * 78)
    print(f"{impl.upper()}  task={task.name}  depths={depths}  pop={config['population_size']}  "
          f"gens={config['max_generations']}  thr={config['fitness_threshold']}")
    print(f"seeds={seeds[0]}..{seeds[-1]} ({len(seeds)})  total={total}  completed={done}")
    print("=" * 78)

    import jax
    start = time.time()
    for depth in depths:
        for seed in seeds:
            run_id = f"d{depth}_s{seed}"
            if run_id in completed:
                continue
            done += 1
            print(f"[{done}/{total}] depth={depth} seed={seed}")
            result = _as_dict(runner(task, depth, seed, config, verbose=verbose))
            results.append(result)
            completed.add(run_id)
            checkpoint_file.write_text(json.dumps(
                {"completed_runs": sorted(completed), "results": results, "config": config}, indent=2))
            solve_gen = result.get("solved_at_gen", result.get("solved_gen", result.get("solve_generation")))
            fitness = result.get("fitness", result.get("best_fitness", 0)) or 0
            status = ("ERROR " + (result.get("error") or "")[:40]) if result.get("error") else (
                f"SOLVED@{solve_gen}" if result.get("solved") else "unsolved")
            print(f"    {status} | fitness {fitness:.4f}")
            try:
                jax.clear_caches()
            except Exception:
                pass

    elapsed = time.time() - start
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "task": task.name,
        "implementation": impl,
        "total_runs": len(results),
        "elapsed_time_s": elapsed,
        "jax_backend": jax.default_backend(),
    }

    if impl == "jax-eshn":
        from .run_jax_eshn import summarize
        output = {"metadata": metadata, "summary": summarize(results), "results": results}
    else:
        output = {"metadata": metadata, "runs": results}
    results_file.write_text(json.dumps(output, indent=2))

    solved = sum(1 for r in results if r.get("solved"))
    print("=" * 78)
    print(f"DONE  solved {solved}/{len(results)}  ->  {results_file}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Unified JAX-ESHN / PUREPLES / HSHG benchmark driver")
    parser.add_argument("--task", required=True, choices=["xor", "parity3", "circle", "sine", "cartpole"])
    parser.add_argument("--impl", default="jax-eshn", choices=["jax-eshn", "pureples", "hshg"])
    parser.add_argument("--depths", type=int, nargs="+", default=None, help="default: task's default depths")
    parser.add_argument("--pop", type=int, default=150, dest="population_size",
                        help="population size (jax-eshn and hshg only; the PUREPLES baseline "
                             "is fixed at 150 by its neat config)")
    parser.add_argument("--gens", type=int, default=100, dest="max_generations")
    parser.add_argument("--seeds", default="0-29", help='range "0-29" or list "42,43,44"')
    parser.add_argument("--threshold", type=float, default=None, help="override task fitness threshold")
    parser.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--timeout", type=int, default=86400, dest="timeout_per_run_s")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Backend must be selected before JAX is imported anywhere.
    os.environ.setdefault("JAX_PLATFORM_NAME", args.backend)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    from .tasks import TASKS
    task = TASKS[args.task]
    depths = args.depths if args.depths is not None else task.default_depths
    seeds = parse_seeds(args.seeds)
    config = {
        "population_size": args.population_size,
        "max_generations": args.max_generations,
        "fitness_threshold": args.threshold if args.threshold is not None else task.fitness_threshold,
        "timeout_per_run_s": args.timeout_per_run_s,
    }
    run_benchmark(task, args.impl, config, depths, seeds, args.out,
                  resume=not args.no_resume, verbose=not args.quiet)


if __name__ == "__main__":
    main()
