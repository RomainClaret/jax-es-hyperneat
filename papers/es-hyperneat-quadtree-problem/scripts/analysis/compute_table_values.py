#!/usr/bin/env python3
"""Compute table values from consolidated ES-HyperNEAT population scaling results.

Parses es_hyperneat_population_scaling_tables.md and computes:
- tab:jit_by_pop: JIT compilation time by depth and population
- tab:runtime_comparison: Total run time = JIT + (post-JIT × 29 gens)

Formula: Generation 1 IS the JIT, so 30 generations = JIT + 29 × post-JIT per gen.
"""

import re
from pathlib import Path


def parse_table(content: str, header_pattern: str, depth_range: tuple[int, int]) -> dict:
    """Parse a markdown table section.

    Args:
        content: Full markdown content
        header_pattern: Pattern to find table header (e.g., "## JIT Compilation Time")
        depth_range: (min_depth, max_depth) to extract

    Returns:
        Dict[depth][population] = value (with ~ prefix info stripped)
    """
    # Find the section
    section_start = content.find(header_pattern)
    if section_start == -1:
        return {}

    # Find the appropriate depth subsection
    depth_min, depth_max = depth_range
    subsection_header = f"### Depths {depth_min}-{depth_max}"
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

    # Parse table rows
    data = {}
    lines = table_content.split('\n')
    header_found = False

    for line in lines:
        if '|' not in line:
            continue

        parts = [p.strip() for p in line.split('|')[1:-1]]  # Remove empty first/last

        if not header_found:
            if 'Population' in parts[0]:
                header_found = True
            continue

        if line.startswith('|--') or not parts:
            continue

        try:
            # Parse population (first column, may have comma separators)
            pop_str = parts[0].replace(',', '').strip()
            if not pop_str.isdigit():
                continue
            pop = int(pop_str)

            # Parse depth values (remaining columns)
            for i, val_str in enumerate(parts[1:], start=depth_min):
                depth = i
                if depth > depth_max:
                    break

                # Clean value: remove ~, ~*, and comma separators
                clean_val = val_str.replace('~*', '').replace('~', '').replace(',', '').strip()

                if not clean_val or clean_val == '0.0':
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
    additional_gens = generations - 1  # Gen 1 IS the JIT

    for depth in jit_data:
        total[depth] = {}
        for pop in jit_data[depth]:
            jit = jit_data[depth][pop]
            post_jit = post_jit_data.get(depth, {}).get(pop, 0)

            # Handle special case: post-JIT = 0 means gen-1 solve, use JIT only
            if post_jit == 0 or post_jit < 1:
                total[depth][pop] = jit
            else:
                total[depth][pop] = jit + (post_jit * additional_gens)

    return total


def format_latex_table_v(total_data: dict, populations: list[int]) -> str:
    """Format tab:runtime_comparison (total run time) in LaTeX.

    Args:
        total_data: Dict[depth][pop] = total time in seconds
        populations: List of population sizes to include

    Returns:
        LaTeX table rows string
    """
    lines = []

    for depth in sorted(total_data.keys()):
        if depth > 7:  # tab:runtime_comparison only goes to D7
            continue

        row_values = []
        for pop in populations:
            val = total_data[depth].get(pop, None)
            if val is None:
                row_values.append("---")
            elif val < 1000:
                row_values.append(f"${int(round(val))}$")
            else:
                # Format with thousands separator
                val_int = int(round(val))
                formatted = f"${val_int:,}$".replace(",", "{,}")
                row_values.append(formatted)

        row = f"{depth} & " + " & ".join(row_values) + " \\\\"
        lines.append(row)

    return "\n".join(lines)


def format_latex_table_vi(jit_data: dict, populations: list[int]) -> str:
    """Format tab:jit_by_pop (JIT compilation time) in LaTeX.

    Args:
        jit_data: Dict[depth][pop] = JIT time in seconds
        populations: List of population sizes to include

    Returns:
        LaTeX table rows string
    """
    lines = []

    for depth in sorted(jit_data.keys()):
        if depth > 7:  # tab:jit_by_pop only goes to D7
            continue

        row_values = []
        for pop in populations:
            val = jit_data[depth].get(pop, None)
            if val is None:
                row_values.append("---")
            elif val < 1000:
                row_values.append(f"${int(round(val))}$")
            else:
                val_int = int(round(val))
                formatted = f"${val_int:,}$".replace(",", "{,}")
                row_values.append(formatted)

        row = f"{depth} & " + " & ".join(row_values) + " \\\\"
        lines.append(row)

    return "\n".join(lines)


def format_tikz_iqr(data: dict, depths: list[int], populations: list[int]) -> tuple[str, str, str]:
    """Compute IQR coordinates for TikZ plotting.

    Args:
        data: Dict[depth][pop] = value
        depths: List of depths to include
        populations: List of populations to compute IQR across

    Returns:
        (q1_coords, q3_coords, median_coords) as TikZ coordinate strings
    """
    import statistics

    q1_coords = []
    q3_coords = []
    median_coords = []

    for depth in depths:
        if depth not in data:
            continue

        values = [data[depth][p] for p in populations if p in data[depth]]
        if not values:
            continue

        values_sorted = sorted(values)
        n = len(values_sorted)

        # Compute quartiles
        q1_idx = n // 4
        q3_idx = (3 * n) // 4
        median_idx = n // 2

        q1 = values_sorted[q1_idx] if q1_idx < n else values_sorted[0]
        q3 = values_sorted[q3_idx] if q3_idx < n else values_sorted[-1]
        median = values_sorted[median_idx] if median_idx < n else values_sorted[0]

        q1_coords.append(f"({depth}, {q1:.1f})")
        q3_coords.append(f"({depth}, {q3:.1f})")
        median_coords.append(f"({depth}, {median:.1f})")

    return (" ".join(q1_coords), " ".join(q3_coords), " ".join(median_coords))


def main():
    # Path to consolidated results (scripts/analysis/ -> paper root)
    results_path = Path(__file__).parent.parent.parent / "benchmark_checkpoints" / "es_hyperneat_population_scaling_tables.md"

    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        return

    content = results_path.read_text()

    # Parse JIT data (depths 1-7)
    jit_data_1_7 = parse_table(content, "## JIT Compilation Time", (1, 7))

    # Parse JIT data (depths 8-10)
    jit_data_8_10 = parse_table(content, "## JIT Compilation Time", (8, 10))

    # Merge JIT data
    jit_data = {**jit_data_1_7}
    for d, pops in jit_data_8_10.items():
        jit_data[d] = pops

    # Parse post-JIT data (depths 1-7)
    post_jit_data_1_7 = parse_table(content, "## Post-JIT Per-Generation Time", (1, 7))

    # Parse post-JIT data (depths 8-10)
    post_jit_data_8_10 = parse_table(content, "## Post-JIT Per-Generation Time", (8, 10))

    # Merge post-JIT data
    post_jit_data = {**post_jit_data_1_7}
    for d, pops in post_jit_data_8_10.items():
        post_jit_data[d] = pops

    # Define populations for tables
    table_pops = [200, 500, 750, 1000]
    all_pops = [50, 100, 150, 200, 250, 500, 750, 1000]

    # Compute total runtime
    total_data = compute_total_runtime(jit_data, post_jit_data, generations=30)

    print("=" * 70)
    print("ES-HyperNEAT Table Value Computation")
    print(f"Source: {results_path}")
    print("=" * 70)

    # Print JIT data summary
    print("\n" + "-" * 70)
    print("JIT Compilation Time (seconds) - Depths 1-7")
    print("-" * 70)
    header = "Depth | " + " | ".join([f"P{p}" for p in table_pops])
    print(header)
    print("-" * len(header))
    for depth in range(1, 8):
        row = f"  {depth}   | "
        vals = []
        for pop in table_pops:
            val = jit_data.get(depth, {}).get(pop, None)
            if val is None:
                vals.append("---")
            else:
                vals.append(f"{val:,.1f}")
        row += " | ".join(f"{v:>8}" for v in vals)
        print(row)

    # Print total runtime summary
    print("\n" + "-" * 70)
    print("Total Runtime (seconds) = JIT + (post-JIT × 29) - Depths 1-7")
    print("-" * 70)
    print(header)
    print("-" * len(header))
    for depth in range(1, 8):
        row = f"  {depth}   | "
        vals = []
        for pop in table_pops:
            val = total_data.get(depth, {}).get(pop, None)
            if val is None:
                vals.append("---")
            else:
                vals.append(f"{val:,.1f}")
        row += " | ".join(f"{v:>8}" for v in vals)
        print(row)

    # Print LaTeX for tab:jit_by_pop
    print("\n" + "=" * 70)
    print("tab:jit_by_pop (JIT Compilation Time) - LaTeX rows:")
    print("=" * 70)
    print(format_latex_table_vi(jit_data, table_pops))

    # Print LaTeX for tab:runtime_comparison
    print("\n" + "=" * 70)
    print("tab:runtime_comparison (Total Run Time) - LaTeX rows:")
    print("=" * 70)
    print(format_latex_table_v(total_data, table_pops))

    # Compute and print TikZ coordinates for fig:baseline_iqr
    print("\n" + "=" * 70)
    print("TikZ Coordinates for fig:baseline_iqr")
    print("=" * 70)

    # JIT IQR (depths 1-7)
    jit_q1, jit_q3, jit_median = format_tikz_iqr(jit_data, list(range(1, 8)), all_pops)
    print("\n% JIT IQR (25th-75th percentile across 8 populations, D1-D7)")
    print("\\addplot[name path=jitmin, color=oigreen, thick, dotted] coordinates {")
    print(f"    {jit_q1}")
    print("};")
    print("\\addplot[name path=jitmax, color=oigreen, thick, dotted] coordinates {")
    print(f"    {jit_q3}")
    print("};")

    # Total runtime IQR (depths 1-10)
    total_q1, total_q3, total_median = format_tikz_iqr(total_data, list(range(1, 11)), all_pops)
    print("\n% JAX-ESHN Total Runtime IQR (25th-75th percentile, D1-D10)")
    print("\\addplot[name path=jaxmin, color=oiblue, thick, dotted] coordinates {")
    print(f"    {total_q1}")
    print("};")
    print("\\addplot[name path=jaxmax, color=oiblue, thick, dotted] coordinates {")
    print(f"    {total_q3}")
    print("};")

    # JIT Pop 1000 line (D1-D10)
    print("\n% JIT compilation time - Pop 1000 (solid line, circle markers)")
    jit_pop1000_coords = []
    for depth in range(1, 11):
        val = jit_data.get(depth, {}).get(1000, None)
        if val is not None:
            jit_pop1000_coords.append(f"({depth}, {int(round(val))})")
    print("\\addplot[color=oiblue, thick, mark=*, solid] coordinates {")
    print(f"    {' '.join(jit_pop1000_coords)}")
    print("};")

    # Print example computation for verification
    print("\n" + "=" * 70)
    print("Example Computations for Verification")
    print("=" * 70)
    for depth in [4, 5, 7]:
        for pop in [200, 1000]:
            jit = jit_data.get(depth, {}).get(pop, 0)
            post_jit = post_jit_data.get(depth, {}).get(pop, 0)
            total = total_data.get(depth, {}).get(pop, 0)
            print(f"D{depth} P{pop}: JIT={jit:.1f}s + (post-JIT={post_jit:.1f}s × 29) = {total:.1f}s")
            if total > 3600:
                print(f"         = {total/3600:.2f} hours")


if __name__ == "__main__":
    main()
