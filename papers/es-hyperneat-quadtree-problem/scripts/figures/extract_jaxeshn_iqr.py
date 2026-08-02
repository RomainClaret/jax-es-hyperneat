#!/usr/bin/env python3
"""Extract JAX-ESHN total runtime IQR from consolidated ES-HyperNEAT results.

Parses es_hyperneat_population_scaling_tables.md to compute IQR (25th-75th percentile)
for total runtime at each depth across all population sizes (50-1000).

Total runtime = JIT + (post-JIT per gen × 29 generations)
Note: Generation 1 IS the JIT, so 30 total generations = JIT + 29 × post-JIT.

Output: TikZ coordinates for JAX-ESHN band in runtime comparison figure.
"""

import re
from pathlib import Path


def parse_table(content: str, section_header: str, depth_range: tuple[int, int]) -> dict:
    """Parse a markdown table section.

    Args:
        content: Full markdown content
        section_header: Section to find (e.g., "## JIT Compilation Time")
        depth_range: (min_depth, max_depth) to extract

    Returns:
        Dict[depth][population] = value
    """
    depth_min, depth_max = depth_range
    subsection_header = f"### Depths {depth_min}-{depth_max}"

    # Find section
    section_start = content.find(section_header)
    if section_start == -1:
        return {}

    # Find appropriate depth subsection
    subsection_start = content.find(subsection_header, section_start)
    if subsection_start == -1:
        return {}

    # Find next section boundary
    next_section = content.find("### ", subsection_start + len(subsection_header))
    if next_section == -1:
        next_section = content.find("## ", subsection_start + len(subsection_header))
    if next_section == -1:
        next_section = len(content)

    table_content = content[subsection_start:next_section]

    # Parse table
    data = {}
    lines = table_content.split('\n')
    header_found = False

    for line in lines:
        if '|' not in line:
            continue

        parts = [p.strip() for p in line.split('|')[1:-1]]

        if not header_found:
            if 'Population' in parts[0]:
                header_found = True
            continue

        if line.startswith('|--') or not parts:
            continue

        try:
            pop_str = parts[0].replace(',', '').strip()
            if not pop_str.isdigit():
                continue
            pop = int(pop_str)

            for i, val_str in enumerate(parts[1:], start=depth_min):
                depth = i
                if depth > depth_max:
                    break

                # Clean value (remove ~, ~*, commas)
                clean_val = val_str.replace('~*', '').replace('~', '').replace(',', '').strip()
                if not clean_val or clean_val == '0.0':
                    # For gen-1 solves, set to very small value instead of 0
                    # to indicate "no post-JIT data"
                    if depth not in data:
                        data[depth] = {}
                    data[depth][pop] = 0.0
                    continue

                try:
                    value = float(clean_val)
                    if depth not in data:
                        data[depth] = {}
                    data[depth][pop] = value
                except ValueError:
                    continue

        except (ValueError, IndexError):
            continue

    return data


def compute_total_runtime(jit_data: dict, post_jit_data: dict, generations: int = 30) -> dict:
    """Compute total runtime = JIT + (post-JIT × (generations - 1)).

    Args:
        jit_data: Dict[depth][pop] = JIT time in seconds
        post_jit_data: Dict[depth][pop] = post-JIT per-gen time in seconds
        generations: Total generations (default 30)

    Returns:
        Dict[depth][pop] = total time in seconds
    """
    total = {}
    additional_gens = generations - 1

    for depth in jit_data:
        total[depth] = {}
        for pop in jit_data[depth]:
            jit = jit_data[depth][pop]
            post_jit = post_jit_data.get(depth, {}).get(pop, 0)

            # For gen-1 solves (post_jit = 0), use JIT only
            if post_jit < 1:
                total[depth][pop] = jit
            else:
                total[depth][pop] = jit + (post_jit * additional_gens)

    return total


def compute_iqr(values: list[float]) -> tuple[float, float, float]:
    """Compute IQR (25th, 75th percentile) and median."""
    if not values:
        return (0, 0, 0)

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    q1_idx = int(n * 0.25)
    q3_idx = int(n * 0.75)
    median_idx = int(n * 0.5)

    q1 = sorted_vals[min(q1_idx, n - 1)]
    q3 = sorted_vals[min(q3_idx, n - 1)]
    median = sorted_vals[min(median_idx, n - 1)]

    return (q1, q3, median)


def main():
    # Path to consolidated results (scripts/figures/ -> paper root)
    results_path = Path(__file__).parent.parent.parent / "benchmark_checkpoints" / "es_hyperneat_population_scaling_tables.md"

    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        return

    content = results_path.read_text()

    # Parse JIT data
    jit_data = {}
    for depth_range in [(1, 7), (8, 10)]:
        data = parse_table(content, "## JIT Compilation Time", depth_range)
        for d, pops in data.items():
            jit_data[d] = pops

    # Parse post-JIT data
    post_jit_data = {}
    for depth_range in [(1, 7), (8, 10)]:
        data = parse_table(content, "## Post-JIT Per-Generation Time", depth_range)
        for d, pops in data.items():
            post_jit_data[d] = pops

    # Compute total runtime
    total_data = compute_total_runtime(jit_data, post_jit_data, generations=30)

    populations = [50, 100, 150, 200, 250, 300, 400, 500, 750, 1000]

    print("=" * 60)
    print("JAX-ESHN Total Runtime IQR (from consolidated results)")
    print(f"Source: {results_path.name}")
    print("Formula: JIT + (post-JIT × 29 generations)")
    print("=" * 60)
    print(f"{'Depth':<6} {'Q1 (25%)':<14} {'Q3 (75%)':<14} {'Median':<14} {'N':<4}")
    print("-" * 60)

    q1_coords = []
    q3_coords = []
    median_coords = []

    for depth in sorted(total_data.keys()):
        values = [total_data[depth][p] for p in populations if p in total_data[depth]]
        if not values:
            continue

        q1, q3, median = compute_iqr(values)
        q1_coords.append(f"({depth}, {q1:.1f})")
        q3_coords.append(f"({depth}, {q3:.1f})")
        median_coords.append(f"({depth}, {median:.1f})")

        # Format large numbers
        def fmt(v):
            if v > 10000:
                return f"{v/3600:.1f}h"
            return f"{v:.1f}s"

        print(f"{depth:<6} {fmt(q1):<14} {fmt(q3):<14} {fmt(median):<14} {len(values):<4}")

    # Print TikZ output
    print()
    print("TikZ coordinates for JAX-ESHN band:")
    print("-" * 60)
    print("% JAX-ESHN Total Runtime IQR (25th-75th percentile across 10 populations)")
    print("\\addplot[name path=jaxmin, color=oiblue, thick, dotted] coordinates {")
    print(f"    {' '.join(q1_coords)}")
    print("};")
    print("\\addplot[name path=jaxmax, color=oiblue, thick, dotted] coordinates {")
    print(f"    {' '.join(q3_coords)}")
    print("};")
    print("\\addplot[fill=oiblue, fill opacity=0.2] fill between[of=jaxmin and jaxmax];")
    print()
    print("% JAX-ESHN median line (optional)")
    print("\\addplot[color=oiblue, thick, mark=square*] coordinates {")
    print(f"    {' '.join(median_coords)}")
    print("};")


if __name__ == "__main__":
    main()
