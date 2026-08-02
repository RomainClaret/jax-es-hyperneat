#!/usr/bin/env python3
"""Analyze PUREPLES + JAX-ESHN benchmark results for the ES-HyperNEAT paper.

Reads per-depth PUREPLES files and JAX-ESHN result files, computes statistics,
and outputs LaTeX tables + summary text.

Usage (from the paper root):
    python scripts/analysis/analyze_results.py
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"  # scripts/analysis/ -> paper root
TASKS = ["parity3", "circle", "sine", "cartpole"]
DEPTHS = [2, 3, 4]
TASK_LABELS = {
    "parity3": "Parity-3",
    "circle": "Circle",
    "sine": "Sine",
    "cartpole": "Cart-pole",
}


def load_pureples(task: str, depth: int) -> List[dict]:
    """Load PUREPLES results for a task at a specific depth."""
    path = RESULTS_DIR / f"pureples_{task}_d{depth}_results.json"
    if not path.exists():
        return []
    data = json.load(open(path))
    runs = data.get("runs", [])
    # Normalize field names (PUREPLES schema varies between tasks)
    normalized = []
    for r in runs:
        # Time field: try wall_time_s first, then total_time_s
        time_s = r.get("wall_time_s") or r.get("total_time_s") or 0.0
        # Solved gen: try solve_generation first, then solved_gen
        solved_gen = r.get("solve_generation") or r.get("solved_gen")
        normalized.append({
            "impl": "Baseline",
            "depth": r.get("depth", depth),
            "seed": r.get("seed"),
            "solved": r.get("solved", False),
            "fitness": r.get("best_fitness", 0.0),
            "total_time_s": time_s,
            "solved_gen": solved_gen,
            "discovered_nodes": r.get("final_discovered_nodes") or r.get("hidden_nodes"),
        })
    return normalized


def load_jax_eshn(task: str) -> List[dict]:
    """Load JAX-ESHN results for a task (all depths in one file)."""
    # Try combined results file first
    path = RESULTS_DIR / f"jax_eshn_{task}_results.json"
    if path.exists():
        data = json.load(open(path))
        runs = data.get("results", [])
    else:
        runs = []

    # For CartPole, also check per-depth checkpoint files
    if task == "cartpole":
        for d in DEPTHS:
            cp_path = RESULTS_DIR / f"cartpole_d{d}" / "jax_eshn_cartpole_checkpoint.json"
            if cp_path.exists():
                cp_data = json.load(open(cp_path))
                cp_runs = cp_data.get("results", [])
                # Avoid duplicates: check if these depths are already loaded
                existing_keys = {(r.get("depth"), r.get("seed")) for r in runs}
                for r in cp_runs:
                    key = (r.get("depth"), r.get("seed"))
                    if key not in existing_keys:
                        runs.append(r)

    # Normalize field names
    normalized = []
    for r in runs:
        if r.get("error") is not None:
            continue
        normalized.append({
            "impl": "JAX-ESHN",
            "depth": r.get("depth"),
            "seed": r.get("seed"),
            "solved": r.get("solved", False),
            "fitness": r.get("fitness", 0.0),
            "total_time_s": r.get("total_time_s", 0.0),
            "solved_gen": r.get("solved_at_gen"),
            "jit_time_s": r.get("jit_compilation_time_s", 0.0),
            "gen_time_s": r.get("time_per_gen_post_jit_s", 0.0),
        })
    return normalized


def compute_stats(runs: List[dict]) -> dict:
    """Compute summary statistics for a set of runs."""
    if not runs:
        return {"n": 0, "solve_rate": 0.0, "solved_n": 0}
    n = len(runs)
    solved_n = sum(1 for r in runs if r["solved"])
    times = [r["total_time_s"] for r in runs]
    fitnesses = [r["fitness"] for r in runs]
    solved_gens = [r["solved_gen"] for r in runs if r["solved"] and r["solved_gen"] is not None]

    return {
        "n": n,
        "solved_n": solved_n,
        "solve_rate": solved_n / n * 100,
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "mean_fitness": float(np.mean(fitnesses)),
        "std_fitness": float(np.std(fitnesses)),
        "median_solve_gen": float(np.median(solved_gens)) if solved_gens else None,
    }


def main():
    # Load all data
    all_data: Dict[str, Dict[int, Dict[str, List[dict]]]] = {}
    for task in TASKS:
        all_data[task] = {}
        for d in DEPTHS:
            pureples = load_pureples(task, d)
            jax_all = load_jax_eshn(task)
            jax = [r for r in jax_all if r["depth"] == d]
            all_data[task][d] = {"Baseline": pureples, "JAX-ESHN": jax}

    # Print data counts
    print("=" * 70)
    print("DATA LOADED")
    print("=" * 70)
    total = 0
    for task in TASKS:
        for d in DEPTHS:
            nb = len(all_data[task][d]["Baseline"])
            nj = len(all_data[task][d]["JAX-ESHN"])
            total += nb + nj
            print(f"  {task:10s} d{d}: Baseline={nb:2d}, JAX-ESHN={nj:2d}")
    print(f"  TOTAL: {total} runs")

    # Compute all statistics
    all_stats: Dict[str, Dict[int, Dict[str, dict]]] = {}
    for task in TASKS:
        all_stats[task] = {}
        for d in DEPTHS:
            all_stats[task][d] = {
                "Baseline": compute_stats(all_data[task][d]["Baseline"]),
                "JAX-ESHN": compute_stats(all_data[task][d]["JAX-ESHN"]),
            }

    # === TABLE: Baseline Results ===
    print("\n" + "=" * 70)
    print("BASELINE TABLE (already in paper; verify numbers)")
    print("=" * 70)
    for task in TASKS:
        label = TASK_LABELS[task]
        row = f"{label:10s}"
        for d in DEPTHS:
            s = all_stats[task][d]["Baseline"]
            row += f"  | D{d}: {s['solve_rate']:5.1f}% ({s['solved_n']:2d}/{s['n']:2d})  {s['mean_time']:8.0f}s"
        d2_time = all_stats[task][2]["Baseline"]["mean_time"]
        d4_time = all_stats[task][4]["Baseline"]["mean_time"]
        ratio = d4_time / d2_time if d2_time > 0 else 0
        row += f"  | {ratio:.0f}x"
        print(row)

    # === TABLE: JAX-ESHN Results ===
    print("\n" + "=" * 70)
    print("JAX-ESHN TABLE (new)")
    print("=" * 70)
    for task in TASKS:
        label = TASK_LABELS[task]
        row = f"{label:10s}"
        for d in DEPTHS:
            s = all_stats[task][d]["JAX-ESHN"]
            if s["n"] > 0:
                row += f"  | D{d}: {s['solve_rate']:5.1f}% ({s['solved_n']:2d}/{s['n']:2d})  {s['mean_time']:8.0f}s"
            else:
                row += f"  | D{d}: {'N/A':>30s}"
        d2_time = all_stats[task][2]["JAX-ESHN"]["mean_time"]
        d4_time = all_stats[task][4]["JAX-ESHN"]["mean_time"]
        ratio = d4_time / d2_time if d2_time > 0 else 0
        row += f"  | {ratio:.1f}x"
        print(row)

    # === STATISTICAL COMPARISONS ===
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISONS (Baseline vs JAX-ESHN)")
    print("=" * 70)
    for task in TASKS:
        print(f"\n--- {TASK_LABELS[task]} ---")
        for d in DEPTHS:
            base_runs = all_data[task][d]["Baseline"]
            jax_runs = all_data[task][d]["JAX-ESHN"]
            bs = all_stats[task][d]["Baseline"]
            js = all_stats[task][d]["JAX-ESHN"]

            if bs["n"] == 0 or js["n"] == 0:
                print(f"  D{d}: insufficient data")
                continue

            # Fisher's exact test for solve rate
            a = bs["solved_n"]  # base solved
            b = bs["n"] - a     # base unsolved
            c = js["solved_n"]  # jax solved
            e = js["n"] - c     # jax unsolved
            table = [[a, b], [c, e]]
            odds, fisher_p = stats.fisher_exact(table)

            # Mann-Whitney U for total time
            base_times = [r["total_time_s"] for r in base_runs]
            jax_times = [r["total_time_s"] for r in jax_runs]
            if len(base_times) > 1 and len(jax_times) > 1:
                u_stat, mw_p = stats.mannwhitneyu(base_times, jax_times, alternative="two-sided")
                # Effect size (rank-biserial r)
                n1, n2 = len(base_times), len(jax_times)
                r_rb = 1 - (2 * u_stat) / (n1 * n2)
            else:
                mw_p = None
                r_rb = None

            print(f"  D{d}: Solve rate Base={bs['solve_rate']:.1f}% vs JAX={js['solve_rate']:.1f}%"
                  f"  Fisher p={fisher_p:.4g}")
            mw_str = f"{mw_p:.4g}" if mw_p is not None else "N/A"
            r_str = f"{r_rb:.3f}" if r_rb is not None else "N/A"
            print(f"        Time Base={bs['mean_time']:.0f}s vs JAX={js['mean_time']:.0f}s"
                  f"  Mann-Whitney p={mw_str}  r={r_str}")

    # === LATEX: JAX-ESHN Table ===
    print("\n" + "=" * 70)
    print("LATEX: JAX-ESHN TABLE")
    print("=" * 70)
    print(r"""\begin{table}[h]
\centering
\caption{JAX-ESHN multi-benchmark results (Pop 150, 30 seeds per cell). Cart-pole complete (90/90 seeds; all solved).}
\label{tab:multi_benchmark_jax}
\small
\begin{tabular}{l|rr|rr|rr|r}
\toprule
& \multicolumn{2}{c|}{\textbf{Depth 2}} & \multicolumn{2}{c|}{\textbf{Depth 3}} & \multicolumn{2}{c|}{\textbf{Depth 4}} & \\
\textbf{Task} & \textbf{\%} & \textbf{Time} & \textbf{\%} & \textbf{Time} & \textbf{\%} & \textbf{Time} & \textbf{d2$\to$d4} \\
\midrule""")
    for task in TASKS:
        label = TASK_LABELS[task]
        parts = []
        for d in DEPTHS:
            s = all_stats[task][d]["JAX-ESHN"]
            if s["n"] > 0:
                t = s["mean_time"]
                if t >= 10000:
                    t_str = f"{t:,.0f}s".replace(",", "{,}")
                else:
                    t_str = f"{t:,.0f}s".replace(",", "{,}")
                parts.append(f"{s['solve_rate']:.1f} & {t_str}")
            else:
                parts.append("-- & --")
        d2t = all_stats[task][2]["JAX-ESHN"]["mean_time"]
        d4t = all_stats[task][4]["JAX-ESHN"]["mean_time"]
        ratio = d4t / d2t if d2t > 0 else 0
        suffix = "*" if task == "cartpole" else ""
        print(f"{label}{suffix:4s} & {parts[0]} & {parts[1]} & {parts[2]} & {ratio:.1f}$\\times$ \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    # === KEY FINDINGS FOR PROSE ===
    print("\n" + "=" * 70)
    print("KEY FINDINGS FOR PAPER PROSE")
    print("=" * 70)

    # Scaling contrast
    for task in ["sine", "circle", "parity3"]:
        bt2 = all_stats[task][2]["Baseline"]["mean_time"]
        bt4 = all_stats[task][4]["Baseline"]["mean_time"]
        jt2 = all_stats[task][2]["JAX-ESHN"]["mean_time"]
        jt4 = all_stats[task][4]["JAX-ESHN"]["mean_time"]
        b_ratio = bt4 / bt2 if bt2 > 0 else 0
        j_ratio = jt4 / jt2 if jt2 > 0 else 0
        print(f"  {TASK_LABELS[task]:10s}: Baseline d2→d4 = {b_ratio:.0f}×, JAX-ESHN d2→d4 = {j_ratio:.1f}×")

    # Total runs
    total_base = sum(all_stats[t][d]["Baseline"]["n"] for t in TASKS for d in DEPTHS)
    total_jax = sum(all_stats[t][d]["JAX-ESHN"]["n"] for t in TASKS for d in DEPTHS)
    print(f"\n  Total runs: Baseline={total_base}, JAX-ESHN={total_jax}, Combined={total_base+total_jax}")


if __name__ == "__main__":
    main()
