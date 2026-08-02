#!/usr/bin/env python3
"""Verify all benchmark statistics against raw JSON data files.

Reads the 6 result JSON files and independently computes every statistic
reported in benchmark_results_summary.md. Prints PASS/FAIL for each check.

Uses only Python stdlib (json, statistics, os, sys).
"""

import json
import os
import statistics
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "..", "..", "results")  # scripts/runners/ -> paper root


def load_json(path):
    with open(path) as f:
        return json.load(f)


def by_depth(results, depth):
    return [r for r in results if r["depth"] == depth]


def check(name, computed, expected, tol=0.05):
    ok = abs(computed - expected) <= tol
    status = "PASS" if ok else "FAIL"
    msg = f"  {status} {name}: {computed}"
    if not ok:
        msg += f" (expected {expected}, diff={computed - expected:+.4f})"
    return ok, msg


def main():
    passes = 0
    fails = 0

    # --- Load data ---
    file_map = {
        "sine": os.path.join(RESULTS, "jax_eshn_sine_results.json"),
        "circle": os.path.join(RESULTS, "jax_eshn_circle_results.json"),
        "parity3": os.path.join(RESULTS, "jax_eshn_parity3_results.json"),
        "cartpole_d2": os.path.join(RESULTS, "cartpole_d2", "jax_eshn_cartpole_results.json"),
        "cartpole_d3": os.path.join(RESULTS, "cartpole_d3", "jax_eshn_cartpole_results.json"),
        "cartpole_d4": os.path.join(RESULTS, "cartpole_d4", "jax_eshn_cartpole_results.json"),
    }

    data = {}
    for key, path in file_map.items():
        if not os.path.exists(path):
            print(f"ERROR: Missing file: {path}")
            sys.exit(1)
        data[key] = load_json(path)

    cartpole_all = (
        data["cartpole_d2"]["results"]
        + data["cartpole_d3"]["results"]
        + data["cartpole_d4"]["results"]
    )

    def get_runs(prob, depth):
        if prob == "cartpole":
            return by_depth(cartpole_all, depth)
        return by_depth(data[prob]["results"], depth)

    def std_for(prob, vals):
        """Use sample std (ddof=1) for sine/circle/parity3, population std for cartpole.

        Issue 8 (NOTE): The benchmark scripts used different conventions.
        Difference is ~1.75% at N=30. Both are valid; documenting for awareness.
        """
        if prob == "cartpole":
            return statistics.pstdev(vals)
        return statistics.stdev(vals)

    # ========================================
    # 1. Fitness thresholds
    # ========================================
    print("=" * 60)
    print("VERIFICATION: Fitness Thresholds")
    print("=" * 60)

    expected_thresholds = {
        "sine": 0.95,
        "circle": 0.975,
        "parity3": 0.975,
        "cartpole_d2": 0.95,
    }
    for key, expected in expected_thresholds.items():
        actual = data[key]["metadata"]["config"]["fitness_threshold"]
        ok, msg = check(f"{key} threshold", actual, expected, tol=0.001)
        print(msg)
        passes += ok
        fails += not ok

    # ========================================
    # 2. Solve rates
    # ========================================
    print("\n" + "=" * 60)
    print("VERIFICATION: Solve Rates")
    print("=" * 60)

    expected_solve = {
        ("sine", 2): 29, ("sine", 3): 30, ("sine", 4): 30,
        ("circle", 2): 0, ("circle", 3): 0, ("circle", 4): 0,
        ("parity3", 2): 4, ("parity3", 3): 7, ("parity3", 4): 5,
        ("cartpole", 2): 30, ("cartpole", 3): 30, ("cartpole", 4): 30,
    }

    for (prob, depth), exp_solved in expected_solve.items():
        runs = get_runs(prob, depth)
        n = len(runs)
        solved = sum(1 for r in runs if r["solved"])
        ok = solved == exp_solved and n == 30
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {prob} d{depth}: {solved}/{n}")
        if not ok:
            print(f"        Expected: {exp_solved}/30")
        passes += ok
        fails += not ok

    # ========================================
    # 3. Speed per generation (mean ± std)
    # ========================================
    print("\n" + "=" * 60)
    print("VERIFICATION: Speed per Generation")
    print("=" * 60)

    expected_speed = {
        ("sine", 2): (13.1, 0.9), ("sine", 3): (27.4, 4.1), ("sine", 4): (65.8, 16.9),
        ("circle", 2): (16.4, 1.0), ("circle", 3): (33.5, 7.2), ("circle", 4): (78.1, 24.9),
        ("parity3", 2): (13.6, 1.1), ("parity3", 3): (27.3, 3.7), ("parity3", 4): (65.9, 14.6),
        ("cartpole", 2): (312.6, 107.6), ("cartpole", 3): (476.7, 118.5), ("cartpole", 4): (964.9, 348.7),
    }

    for (prob, depth), (exp_mean, exp_std) in expected_speed.items():
        vals = [r["time_per_gen_post_jit_s"] for r in get_runs(prob, depth)]
        m, s = statistics.mean(vals), std_for(prob, vals)
        ok_m, msg_m = check(f"{prob} d{depth} mean s/gen", round(m, 1), exp_mean, tol=0.15)
        ok_s, msg_s = check(f"{prob} d{depth} std s/gen", round(s, 1), exp_std, tol=0.15)
        print(msg_m)
        print(msg_s)
        passes += ok_m + ok_s
        fails += (not ok_m) + (not ok_s)

    # ========================================
    # 4. JIT compilation time
    # ========================================
    print("\n" + "=" * 60)
    print("VERIFICATION: JIT Compilation Time")
    print("=" * 60)

    expected_jit = {
        ("sine", 2): (56.4, 2.2), ("sine", 3): (117.2, 4.7), ("sine", 4): (328.8, 36.2),
        ("circle", 2): (60.3, 2.8), ("circle", 3): (127.7, 5.0), ("circle", 4): (384.6, 53.1),
        ("parity3", 2): (60.6, 2.8), ("parity3", 3): (128.7, 4.4), ("parity3", 4): (419.3, 49.8),
        ("cartpole", 2): (558.9, 92.9), ("cartpole", 3): (849.0, 130.3), ("cartpole", 4): (1579.3, 243.0),
    }

    for (prob, depth), (exp_mean, exp_std) in expected_jit.items():
        vals = [r["jit_compilation_time_s"] for r in get_runs(prob, depth)]
        m, s = statistics.mean(vals), std_for(prob, vals)
        ok_m, msg_m = check(f"{prob} d{depth} mean JIT", round(m, 1), exp_mean, tol=0.15)
        ok_s, msg_s = check(f"{prob} d{depth} std JIT", round(s, 1), exp_std, tol=0.15)
        print(msg_m)
        print(msg_s)
        passes += ok_m + ok_s
        fails += (not ok_m) + (not ok_s)

    # ========================================
    # 5. Total wall time (mean)
    # ========================================
    print("\n" + "=" * 60)
    print("VERIFICATION: Total Wall Time")
    print("=" * 60)

    expected_wall = {
        ("sine", 2): 1350, ("sine", 3): 2825, ("sine", 4): 6841,
        ("circle", 2): 1685, ("circle", 3): 3447, ("circle", 4): 8113,
        ("parity3", 2): 1406, ("parity3", 3): 2827, ("parity3", 4): 6947,
        ("cartpole", 2): 31506, ("cartpole", 3): 48042, ("cartpole", 4): 79424,
    }

    for (prob, depth), exp_val in expected_wall.items():
        vals = [r["total_time_s"] for r in get_runs(prob, depth)]
        m = round(statistics.mean(vals))
        ok, msg = check(f"{prob} d{depth} wall time (s)", m, exp_val, tol=1)
        print(msg)
        passes += ok
        fails += not ok

    # ========================================
    # 6. Fitness distribution (UNSOLVED ONLY)
    # ========================================
    print("\n" + "=" * 60)
    print("VERIFICATION: Fitness Distribution (unsolved only)")
    print("=" * 60)

    expected_fitness = {
        ("sine", 2): (1, 0.948, 0.948, 0.948),
        ("circle", 2): (30, 0.867, 0.852, 0.887),
        ("circle", 3): (30, 0.865, 0.850, 0.880),
        ("circle", 4): (30, 0.862, 0.842, 0.871),
        ("parity3", 2): (26, 0.861, 0.796, 0.955),
        ("parity3", 3): (23, 0.870, 0.812, 0.936),
        ("parity3", 4): (25, 0.877, 0.840, 0.965),
    }

    for (prob, depth), (exp_n, exp_mean, exp_min, exp_max) in expected_fitness.items():
        unsolved = [r for r in get_runs(prob, depth) if not r["solved"]]
        n = len(unsolved)
        if n == 0:
            print(f"  SKIP {prob} d{depth}: no unsolved runs")
            continue

        fitnesses = [r["fitness"] for r in unsolved]
        m = round(statistics.mean(fitnesses), 3)
        mn = round(min(fitnesses), 3)
        mx = round(max(fitnesses), 3)

        ok = (n == exp_n and m == exp_mean and mn == exp_min and mx == exp_max)
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {prob} d{depth}: N={n} mean={m:.3f} min={mn:.3f} max={mx:.3f}")
        if not ok:
            print(f"        Expected: N={exp_n} mean={exp_mean} min={exp_min} max={exp_max}")
        passes += ok
        fails += not ok

    # ========================================
    # 7. Solved-at generation (min / median / max)
    # ========================================
    print("\n" + "=" * 60)
    print("VERIFICATION: Solved-At Generation")
    print("=" * 60)

    expected_solved_at = {
        ("sine", 2): (1, 2, 12),
        ("sine", 3): (1, 2, 80),
        ("sine", 4): (1, 2, 57),
        ("parity3", 2): (8, 15, 28),  # CORRECTED: was 20
        ("parity3", 3): (1, 7, 65),
        ("parity3", 4): (11, 13, 41),
        ("cartpole", 2): (1, 1, 2),
        ("cartpole", 3): (1, 1, 2),
        ("cartpole", 4): (1, 1, 3),
    }

    for (prob, depth), (exp_min, exp_med, exp_max) in expected_solved_at.items():
        solved = [r for r in get_runs(prob, depth) if r["solved"]]
        if not solved:
            print(f"  SKIP {prob} d{depth}: no solved runs")
            continue

        gens = sorted(r["solved_at_gen"] for r in solved)
        mn, mx = min(gens), max(gens)
        med = statistics.median(gens)
        # Convert to int if whole number
        med_display = int(med) if med == int(med) else med

        ok = (mn == exp_min and med_display == exp_med and mx == exp_max)
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {prob} d{depth}: {mn} / {med_display} / {mx}")
        if not ok:
            print(f"        Expected: {exp_min} / {exp_med} / {exp_max}")
            print(f"        Raw solved_at_gen values: {gens}")
        passes += ok
        fails += not ok

    # ========================================
    # 8. Depth scaling ratios
    # ========================================
    print("\n" + "=" * 60)
    print("VERIFICATION: Depth Scaling")
    print("=" * 60)

    expected_scaling = {
        "sine": (2.1, 5.0),
        "circle": (2.0, 4.8),
        "parity3": (2.0, 4.8),
        "cartpole": (1.5, 3.1),
    }

    for prob, (exp_d3, exp_d4) in expected_scaling.items():
        m2 = statistics.mean(r["time_per_gen_post_jit_s"] for r in get_runs(prob, 2))
        m3 = statistics.mean(r["time_per_gen_post_jit_s"] for r in get_runs(prob, 3))
        m4 = statistics.mean(r["time_per_gen_post_jit_s"] for r in get_runs(prob, 4))
        r3, r4 = round(m3 / m2, 1), round(m4 / m2, 1)

        # Tolerance of 0.15 for rounding boundary cases (e.g. 4.85 -> 4.8 or 4.9)
        ok_d3 = abs(r3 - exp_d3) <= 0.15
        ok_d4 = abs(r4 - exp_d4) <= 0.15
        ok = ok_d3 and ok_d4
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {prob}: d2->d3={r3}x d2->d4={r4}x")
        if not ok:
            print(f"        Expected: d2->d3={exp_d3}x d2->d4={exp_d4}x")
        passes += ok
        fails += not ok

    # ========================================
    # 9. Stale file check
    # ========================================
    print("\n" + "=" * 60)
    print("VERIFICATION: Stale Files")
    print("=" * 60)

    stale = os.path.join(RESULTS, "jax_eshn_cartpole_checkpoint.json")
    if os.path.exists(stale):
        print(f"  FAIL Stale file exists: jax_eshn_cartpole_checkpoint.json")
        fails += 1
    else:
        print(f"  PASS Stale file removed")
        passes += 1

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    total = passes + fails
    print(f"RESULT: {passes}/{total} passed, {fails} failed")
    print("=" * 60)

    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
