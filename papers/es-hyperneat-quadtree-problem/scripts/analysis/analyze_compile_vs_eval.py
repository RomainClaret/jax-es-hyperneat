#!/usr/bin/env python3
"""Summarize compile_vs_eval_isolation.json into the deliverable table.

Reports, per (depth, population), the mean over seeds of:
  - pure XLA compile time of each jittable unit (cppn / substrate / ask / tell)
  - cold-first vs warm per-individual eval cost
  - projected first-gen eval and steady-state per-gen cost (= pop * warm)

Then quantifies how pure compile vs eval scale with population.
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

JSON = Path(__file__).resolve().parents[2] / "results" / "compile_vs_eval_isolation.json"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def main():
    data = json.loads(JSON.read_text())
    cells = [c for c in data["cells"] if c.get("status") == "ok"]
    if not cells:
        print("No completed cells yet.")
        return

    by = defaultdict(list)
    for c in cells:
        by[(c["depth"], c["pop_size"])].append(c)

    def comp(c, unit, field="compile_s"):
        return c.get("pure_compiles", {}).get(unit, {}).get(field)

    print(f"Source: {JSON}")
    print(f"Backend: {data['metadata'].get('backend')} | "
          f"JAX {data['metadata'].get('jax_version')} | "
          f"n seeds requested: {data['metadata'].get('seeds')}\n")

    hdr = (f"{'D':>2} {'pop':>5} {'n':>2} | "
           f"{'cppn_c':>7} {'subst_c':>7} {'ask_c':>6} {'tell_c':>7} | "
           f"{'cold_ind':>8} {'warm_ind':>8} | "
           f"{'proj_1stgen':>11} {'proj_steady':>11}")
    print(hdr)
    print("-" * len(hdr))

    rows = {}
    for (depth, pop) in sorted(by):
        cs = by[(depth, pop)]
        n = len(cs)
        cppn_c = mean([comp(c, "cppn_forward") for c in cs])
        subst_c = mean([comp(c, "substrate_forward") for c in cs])
        ask_c = mean([comp(c, "neat_ask") for c in cs])
        tell_c = mean([comp(c, "neat_tell") for c in cs])
        cold = mean([c["eval_loop"]["cold_first_individual_s"] for c in cs])
        warm = mean([c["eval_loop"]["mean_warm_individual_s"] for c in cs])
        p1 = mean([c["eval_loop"]["projected_first_gen_eval_s"] for c in cs])
        psteady = mean([c["eval_loop"]["projected_steady_state_per_gen_s"] for c in cs])
        rows[(depth, pop)] = dict(cppn_c=cppn_c, subst_c=subst_c, ask_c=ask_c,
                                  tell_c=tell_c, cold=cold, warm=warm,
                                  p1=p1, psteady=psteady, n=n)
        print(f"{depth:>2} {pop:>5} {n:>2} | "
              f"{cppn_c:>7.4f} {subst_c:>7.4f} {ask_c:>6.4f} {tell_c:>7.4f} | "
              f"{cold:>8.3f} {warm:>8.3f} | "
              f"{p1:>11.1f} {psteady:>11.1f}")

    # Scaling analysis: ratio from smallest to largest population at each depth.
    print("\nSCALING (largest_pop / smallest_pop, per depth):")
    depths = sorted({d for d, _ in by})
    for d in depths:
        pops = sorted(p for dd, p in by if dd == d)
        if len(pops) < 2:
            continue
        lo, hi = pops[0], pops[-1]
        rlo, rhi = rows[(d, lo)], rows[(d, hi)]
        def ratio(a, b):
            return (b / a) if (a and a == a and b == b and a != 0) else float("nan")
        print(f"  depth {d}: pop {lo} -> {hi} ({hi/lo:.1f}x population)")
        print(f"    pure compile: cppn {ratio(rlo['cppn_c'], rhi['cppn_c']):.2f}x  "
              f"substrate {ratio(rlo['subst_c'], rhi['subst_c']):.2f}x  "
              f"ask {ratio(rlo['ask_c'], rhi['ask_c']):.2f}x  "
              f"tell {ratio(rlo['tell_c'], rhi['tell_c']):.2f}x")
        print(f"    per-individual eval: warm {ratio(rlo['warm'], rhi['warm']):.2f}x "
              f"(expected ~1.0 = pop-independent)")
        print(f"    projected steady per-gen: "
              f"{ratio(rlo['psteady'], rhi['psteady']):.2f}x "
              f"(expected ~{hi/lo:.1f}x = linear in pop)")


if __name__ == "__main__":
    main()
