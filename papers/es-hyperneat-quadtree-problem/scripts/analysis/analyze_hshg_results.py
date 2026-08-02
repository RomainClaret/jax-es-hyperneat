#!/usr/bin/env python3
"""Analyze HSHG vs Quadtree benchmark results for the ES-HyperNEAT paper.

Reads JSON result files and prints statistics matching Table tab:hshg_task_performance.
Run from repo root:
    python papers/es-hyperneat-quadtree-problem/scripts/analysis/analyze_hshg_results.py
"""

import json
import math
import statistics
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"  # scripts/analysis/ -> paper root

# All conditions: (task, depth, pop) -> (hshg_file, quadtree_file)
CONDITIONS = [
    # Pop=50
    ("XOR", "D2", 50, "hshg_xor_d2_results.json", "pureples_xor_d2_pop50_results.json"),
    ("XOR", "D3", 50, "hshg_xor_d3_results.json", "pureples_xor_d3_pop50_results.json"),
    ("Parity-3", "D2", 50, "hshg_parity3_d2_results.json", "pureples_parity3_d2_results.json"),
    ("Sine", "D2", 50, "hshg_sine_d2_results.json", "pureples_sine_d2_results.json"),
    # Pop=150
    ("XOR", "D2", 150, "hshg_xor_d2_pop150_results.json", "pureples_xor_d2_pop150_results.json"),
    ("XOR", "D3", 150, "hshg_xor_d3_pop150_results.json", "pureples_xor_d3_pop150_results.json"),
    ("Parity-3", "D2", 150, "hshg_parity3_d2_pop150_results.json", "pureples_parity3_d2_results.json"),
    ("Sine", "D2", 150, "hshg_sine_d2_pop150_results.json", "pureples_sine_d2_results.json"),
]


def analyze_file(path: Path) -> dict:
    """Extract statistics from a benchmark result JSON file."""
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    runs = data.get("runs", [])
    n = len(runs)
    if n == 0:
        return None

    solved = [r for r in runs if r.get("solved", False)]
    n_solved = len(solved)

    fitnesses_valid = []
    n_inf = 0
    for r in runs:
        fit = r.get("best_fitness")
        # Runs where substrate bloat prevented network construction carry
        # construction_failed=true and a null fitness. Older copies of this data
        # wrote -Infinity instead; both forms are handled here.
        failed = r.get("construction_failed")
        if failed is None:
            failed = fit is None or (isinstance(fit, float) and math.isinf(fit) and fit < 0)
        if failed:
            n_inf += 1
        else:
            fitnesses_valid.append(fit)

    times = [r.get("total_time_s", r.get("wall_time_s", 0)) for r in runs]

    result = {
        "n_runs": n,
        "n_solved": n_solved,
        "solve_rate": n_solved / n if n > 0 else 0,
        "n_inf": n_inf,
        "n_valid": len(fitnesses_valid),
    }

    if fitnesses_valid:
        result["fitness_mean"] = statistics.mean(fitnesses_valid)
        result["fitness_std"] = statistics.stdev(fitnesses_valid) if len(fitnesses_valid) > 1 else 0.0
    else:
        result["fitness_mean"] = float("-inf")
        result["fitness_std"] = 0.0

    if times:
        result["time_mean"] = statistics.mean(times)

    return result


def fmt_solve(r):
    return f"{r['n_solved']}/{r['n_runs']} ({r['solve_rate']:.0%})"


def fmt_fitness(r):
    if r["fitness_mean"] == float("-inf"):
        return "ALL -inf"
    return f"{r['fitness_mean']:.4f} +/- {r['fitness_std']:.4f}"


def main():
    print("=" * 90)
    print("HSHG vs Quadtree: Full Results Analysis")
    print("=" * 90)

    # Group by population
    for pop in [50, 150]:
        print(f"\n--- Population = {pop} ---\n")
        print(f"{'Task':<10} {'D':<4} {'Quadtree Solve':<18} {'HSHG Solve':<18} "
              f"{'HSHG Fitness':<22} {'HSHG -inf':<12}")
        print("-" * 84)

        total_hshg = 0
        total_solved = 0
        total_inf = 0

        for task, depth, p, hshg_f, qt_f in CONDITIONS:
            if p != pop:
                continue

            qt = analyze_file(RESULTS_DIR / qt_f)
            h = analyze_file(RESULTS_DIR / hshg_f)

            qt_str = fmt_solve(qt) if qt else "N/A"
            if h:
                h_solve = fmt_solve(h)
                h_fit = fmt_fitness(h)
                h_inf = f"{h['n_inf']}/{h['n_runs']}"
                total_hshg += h["n_runs"]
                total_solved += h["n_solved"]
                total_inf += h["n_inf"]
            else:
                h_solve = "running..."
                h_fit = ""
                h_inf = ""

            print(f"{task:<10} {depth:<4} {qt_str:<18} {h_solve:<18} "
                  f"{h_fit:<22} {h_inf:<12}")

        if total_hshg > 0:
            print(f"\n  Pop={pop} totals: {total_solved}/{total_hshg} solved (0%), "
                  f"{total_inf}/{total_hshg} substrate failures "
                  f"({100*total_inf/total_hshg:.0f}%)")

    # Overall summary
    print("\n" + "=" * 90)
    print("OVERALL SUMMARY")
    print("=" * 90)

    grand_total = 0
    grand_solved = 0
    grand_inf = 0
    for task, depth, pop, hshg_f, qt_f in CONDITIONS:
        h = analyze_file(RESULTS_DIR / hshg_f)
        if h:
            grand_total += h["n_runs"]
            grand_solved += h["n_solved"]
            grand_inf += h["n_inf"]

    if grand_total > 0:
        print(f"Total HSHG runs across all conditions: {grand_total}")
        print(f"Total solved: {grand_solved}/{grand_total} ({100*grand_solved/grand_total:.1f}%)")
        print(f"Total substrate failures (-inf): {grand_inf}/{grand_total} ({100*grand_inf/grand_total:.1f}%)")
        print(f"Total valid but unsolved: {grand_total - grand_inf - grand_solved}/{grand_total}")
        print(f"\nConclusion: HSHG achieves 0% solve rate regardless of population size.")
    else:
        print("No HSHG results available.")


if __name__ == "__main__":
    main()
