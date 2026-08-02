#!/usr/bin/env python3
"""Verify the XOR depth/population scaling numbers (tab:runtime_comparison,
tab:jit_by_pop, and the D8-D10 points) against the raw TensorNEAT checkpoints.

Companion to verify_results.py (which covers the multi-benchmark tables). This
script recomputes, from the per-seed checkpoint files:

  * tab:jit_by_pop (construction overhead by depth x population, D3-D7),
  * tab:runtime_comparison's JAX-ESHN projected 30-generation Total at Pop 500
    (Total = construction overhead + 29 x post-construction per-generation time (generation 1 is inside construction),
    as defined in that table's caption),
  * tab:runtime_comparison's JAX-ESHN solve rates (D3-D7), aggregated over the 8 solve
    campaigns (max_generations=300; 24 trials per depth),
  * the 30-generation-cap solve rates the body quotes for the same campaigns
    (solved runs whose generation count is <= 30), and
  * the D8-D10 Pop-1000 construction-overhead points and gen-1 solves.

It prints PASS/FAIL for each cell against the values printed in the paper.

The scaling checkpoints are committed under this paper folder's
``benchmark_checkpoints/`` (override with JAX_ESHN_BENCH_DIR): the
``tensorneat_eshyperneat_pop{POP}_10gens`` timing reruns, the bare
``tensorneat_eshyperneat_pop{POP}`` solve campaigns, and ``..._pop1000``,
``..._pop1000_depth10``.
"""

import json
import os
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
# This file lives in papers/es-hyperneat-quadtree-problem/scripts/runners/; checkpoints are two levels up.
DEFAULT_BENCH = os.path.normpath(os.path.join(HERE, "..", "..", "benchmark_checkpoints"))
BENCH = os.environ.get("JAX_ESHN_BENCH_DIR", DEFAULT_BENCH)

# --- Paper-reported values -------------------------------------------------

# tab:jit_by_pop: construction overhead (minutes) by depth x population.
PAPER_JIT_MIN = {
    (3, 50): 0.9, (3, 150): 2.5, (3, 300): 4.4, (3, 500): 7.7, (3, 1000): 16.3,
    (4, 50): 2.5, (4, 150): 7.3, (4, 300): 12.4, (4, 500): 23.7, (4, 1000): 51.8,
    (5, 50): 7.3, (5, 150): 22.1, (5, 300): 36.7, (5, 500): 74.4, (5, 1000): 166.4,
    (6, 50): 11.8, (6, 150): 33.4, (6, 300): 52.1, (6, 500): 100.4, (6, 1000): 228.3,
    (7, 50): 12.3, (7, 150): 35.0, (7, 300): 55.5, (7, 500): 105.9, (7, 1000): 269.2,
}

# tab:runtime_comparison: JAX-ESHN projected 30-generation Total at Pop 500 (minutes);
# projection = construction overhead + 29 x per-generation (generation 1 is inside construction), as that table's caption defines.
PAPER_TOTAL30_POP500_MIN = {3: 180, 4: 434, 5: 1618, 6: 2500, 7: 3540}

# tab:runtime_comparison: JAX-ESHN Solve% (D3-D7), 8 solve campaigns (24 trials/depth).
PAPER_SOLVE_PCT = {3: 100.0, 4: 91.7, 5: 100.0, 6: 100.0, 7: 100.0}
SOLVE_POPS = [50, 100, 150, 200, 250, 500, 750, 1000]

# 30-generation-cap solve rates (D3-D7) quoted in the body: solved runs with
# generations <= 30, over the same 8 solve campaigns.
PAPER_SOLVE30_PCT = {3: 95.8, 4: 79.2, 5: 95.8, 6: 100.0, 7: 95.8}

# D8-D10 Pop-1000 construction overhead (seconds), Section "Exploratory deep substrates".
PAPER_DEEP_SEC = {8: 14034, 9: 16575, 10: 21955}


def load_results(path):
    with open(path) as f:
        j = json.load(f)
    return j.get("results", [])


def by_depth(results):
    d = defaultdict(list)
    for r in results:
        d[r["depth"]].append(r)
    return d


def approx(a, b, rel=0.03, absol=0.1):
    return abs(a - b) <= max(absol, rel * abs(b))


def main():
    passes = fails = 0

    if not os.path.isdir(BENCH):
        print(f"ERROR: benchmark dir not found: {BENCH}\n"
              f"Set JAX_ESHN_BENCH_DIR to the directory holding the scaling checkpoints.")
        return 2

    # ---- tab:jit_by_pop: construction overhead by depth x population ----
    print("=" * 60)
    print("tab:jit_by_pop: construction overhead (min) by depth x population")
    print("=" * 60)
    for pop in [50, 150, 300, 500, 1000]:
        path = os.path.join(BENCH, f"tensorneat_eshyperneat_pop{pop}_10gens", "checkpoint.json")
        if not os.path.exists(path):
            print(f"  SKIP pop{pop}: missing {path}")
            continue
        bd = by_depth(load_results(path))
        for d in [3, 4, 5, 6, 7]:
            if d not in bd:
                continue
            m = st.mean(r["jit_compilation_time_s"] for r in bd[d]) / 60.0
            exp = PAPER_JIT_MIN.get((d, pop))
            if exp is None:
                continue
            ok = approx(m, exp)
            passes += ok
            fails += not ok
            print(f"  {'PASS' if ok else 'FAIL'} D{d} pop{pop}: {m:.1f} min (paper {exp})")

    # ---- tab:runtime_comparison: projected 30-gen Total at Pop 500 ----
    print("\n" + "=" * 60)
    print("tab:runtime_comparison: JAX-ESHN projected 30-gen Total at Pop 500 (min)")
    print("  Total = construction overhead + 29 x per-generation time")
    print("=" * 60)
    path = os.path.join(BENCH, "tensorneat_eshyperneat_pop500_10gens", "checkpoint.json")
    if os.path.exists(path):
        bd = by_depth(load_results(path))
        for d in [3, 4, 5, 6, 7]:
            tot = [(r["jit_compilation_time_s"] + 29 * r["time_per_gen_post_jit_s"]) / 60.0
                   for r in bd[d]]
            m = st.mean(tot)
            exp = PAPER_TOTAL30_POP500_MIN[d]
            ok = approx(m, exp, rel=0.02, absol=1)
            passes += ok
            fails += not ok
            print(f"  {'PASS' if ok else 'FAIL'} D{d}: {m:.0f} min (paper {exp})")
    else:
        print(f"  SKIP: missing {path}")

    # ---- tab:runtime_comparison: JAX-ESHN solve rates (D3-D7), 8 solve campaigns ----
    print("\n" + "=" * 60)
    print("tab:runtime_comparison: JAX-ESHN Solve% (D3-D7), 300-gen solve campaigns")
    print("  aggregated over pops " + ",".join(str(p) for p in SOLVE_POPS) + " (24 trials/depth)")
    print("=" * 60)
    solve = defaultdict(lambda: [0, 0])  # depth -> [solved, n]
    missing_solve = []
    for pop in SOLVE_POPS:
        path = os.path.join(BENCH, f"tensorneat_eshyperneat_pop{pop}", "checkpoint.json")
        if not os.path.exists(path):
            missing_solve.append(pop)
            continue
        for d, rs in by_depth(load_results(path)).items():
            if d in PAPER_SOLVE_PCT:
                solve[d][0] += sum(1 for r in rs if r.get("solved"))
                solve[d][1] += len(rs)
    if missing_solve:
        print(f"  SKIP pops {missing_solve}: solve campaigns missing")
    for d in sorted(PAPER_SOLVE_PCT):
        s, n = solve[d]
        if n == 0:
            print(f"  SKIP D{d}: no solve-campaign data")
            continue
        pct = 100.0 * s / n
        ok = approx(pct, PAPER_SOLVE_PCT[d], rel=0.001, absol=0.05) and n == 24
        passes += ok
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'} D{d}: {s}/{n} = {pct:.1f}% (paper {PAPER_SOLVE_PCT[d]})")

    # ---- 30-generation-cap solve rates (D3-D7), same 8 campaigns ----
    print("\n" + "=" * 60)
    print("30-gen-cap Solve% (D3-D7): solved runs with generations <= 30")
    print("=" * 60)
    solve30 = defaultdict(lambda: [0, 0])  # depth -> [solved<=30, n]
    for pop in SOLVE_POPS:
        path = os.path.join(BENCH, f"tensorneat_eshyperneat_pop{pop}", "checkpoint.json")
        if not os.path.exists(path):
            continue
        for d, rs in by_depth(load_results(path)).items():
            if d in PAPER_SOLVE30_PCT:
                solve30[d][0] += sum(1 for r in rs
                                     if r.get("solved") and r.get("generations", 10**9) <= 30)
                solve30[d][1] += len(rs)
    for d in sorted(PAPER_SOLVE30_PCT):
        s, n = solve30[d]
        if n == 0:
            print(f"  SKIP D{d}: no solve-campaign data")
            continue
        pct = 100.0 * s / n
        ok = approx(pct, PAPER_SOLVE30_PCT[d], rel=0.001, absol=0.05) and n == 24
        passes += ok
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'} D{d}: {s}/{n} = {pct:.1f}% (paper {PAPER_SOLVE30_PCT[d]})")

    # ---- D8-D10 Pop 1000 construction overhead ----
    print("\n" + "=" * 60)
    print("D8-D10 Pop-1000 construction overhead (s)")
    print("=" * 60)
    deep = {}
    p1000 = os.path.join(BENCH, "tensorneat_eshyperneat_pop1000", "checkpoint.json")
    p_d10 = os.path.join(BENCH, "tensorneat_eshyperneat_pop1000_depth10", "checkpoint.json")
    for path in (p1000, p_d10):
        if os.path.exists(path):
            for d, rs in by_depth(load_results(path)).items():
                if d in (8, 9, 10):
                    deep[d] = st.mean(r["jit_compilation_time_s"] for r in rs)
    for d in [8, 9, 10]:
        if d in deep:
            ok = approx(deep[d], PAPER_DEEP_SEC[d], rel=0.02, absol=50)
            passes += ok
            fails += not ok
            print(f"  {'PASS' if ok else 'FAIL'} D{d}: {deep[d]:.0f} s (paper {PAPER_DEEP_SEC[d]})")
        else:
            print(f"  SKIP D{d}: not found in checkpoints")

    # ---- D8-D10 Pop-1000 gen-1 solves (Figure/Section "Exploratory deep substrates") ----
    print("\n" + "=" * 60)
    print("D8-D10 Pop-1000 solves (paper: all seeds solve within the first generations)")
    print("=" * 60)
    deep_solve = {}
    for path in (p1000, p_d10):
        if os.path.exists(path):
            for d, rs in by_depth(load_results(path)).items():
                if d in (8, 9, 10):
                    deep_solve[d] = (sum(1 for r in rs if r.get("solved")), len(rs))
    for d in [8, 9, 10]:
        if d in deep_solve:
            s, n = deep_solve[d]
            ok = (s == n == 3)
            passes += ok
            fails += not ok
            print(f"  {'PASS' if ok else 'FAIL'} D{d}: {s}/{n} solved (paper 3/3)")
        else:
            print(f"  SKIP D{d}: not found in checkpoints")

    print("\n" + "=" * 60)
    total = passes + fails
    print(f"RESULT: {passes}/{total} passed, {fails} failed")
    print("=" * 60)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
