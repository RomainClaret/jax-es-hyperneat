#!/usr/bin/env python3
"""Extract JIT compilation time IQR from consolidated ES-HyperNEAT results.

Parses es_hyperneat_population_scaling_tables.md to compute IQR (25th-75th percentile)
for JIT compilation time at each depth across all population sizes (50-1000).

Output: TikZ coordinates for JIT IQR band in runtime comparison figure.
"""

import re
from pathlib import Path


def parse_jit_table(content: str, depth_range: tuple[int, int]) -> dict:
    """Parse JIT compilation time table from markdown.

    Args:
        content: Full markdown content
        depth_range: (min_depth, max_depth) to extract

    Returns:
        Dict[depth][population] = JIT time in seconds
    """
    depth_min, depth_max = depth_range
    subsection_header = f"### Depths {depth_min}-{depth_max}"

    # Find JIT section
    jit_section_start = content.find("## JIT Compilation Time")
    if jit_section_start == -1:
        return {}

    # Find appropriate depth subsection
    subsection_start = content.find(subsection_header, jit_section_start)
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
                if not clean_val:
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


def compute_iqr(values: list[float]) -> tuple[float, float, float]:
    """Compute IQR (25th, 75th percentile) and median.

    Args:
        values: List of values

    Returns:
        (q1, q3, median) tuple
    """
    if not values:
        return (0, 0, 0)

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    # Use numpy-style percentile calculation
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

    # Parse JIT data (depths 1-7 and 8-10)
    jit_data_1_7 = parse_jit_table(content, (1, 7))
    jit_data_8_10 = parse_jit_table(content, (8, 10))

    # Merge
    jit_data = {**jit_data_1_7}
    for d, pops in jit_data_8_10.items():
        jit_data[d] = pops

    populations = [50, 100, 150, 200, 250, 300, 400, 500, 750, 1000]

    print("=" * 60)
    print("JIT Compilation Time IQR (from consolidated results)")
    print(f"Source: {results_path.name}")
    print("=" * 60)
    print(f"{'Depth':<6} {'Q1 (25%)':<12} {'Q3 (75%)':<12} {'Median':<12} {'N':<4}")
    print("-" * 60)

    q1_coords = []
    q3_coords = []
    median_coords = []

    for depth in sorted(jit_data.keys()):
        values = [jit_data[depth][p] for p in populations if p in jit_data[depth]]
        if not values:
            continue

        q1, q3, median = compute_iqr(values)
        q1_coords.append(f"({depth}, {q1:.1f})")
        q3_coords.append(f"({depth}, {q3:.1f})")
        median_coords.append(f"({depth}, {median:.1f})")
        print(f"{depth:<6} {q1:<12.1f} {q3:<12.1f} {median:<12.1f} {len(values):<4}")

    # Print TikZ output
    print()
    print("TikZ coordinates for JIT IQR band:")
    print("-" * 60)
    print("% JIT compilation time IQR (25th-75th percentile across 10 populations)")
    print("\\addplot[name path=jitmin, color=oigreen, thick, dotted] coordinates {")
    print(f"    {' '.join(q1_coords)}")
    print("};")
    print("\\addplot[name path=jitmax, color=oigreen, thick, dotted] coordinates {")
    print(f"    {' '.join(q3_coords)}")
    print("};")
    print("\\addplot[fill=oigreen, fill opacity=0.15] fill between[of=jitmin and jitmax];")
    print()
    print("% JIT median line (optional)")
    print("\\addplot[color=oigreen, thick, mark=triangle*] coordinates {")
    print(f"    {' '.join(median_coords)}")
    print("};")


if __name__ == "__main__":
    main()
