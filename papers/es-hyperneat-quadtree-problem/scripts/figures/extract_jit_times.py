#!/usr/bin/env python3
"""Extract JIT compilation times for Pop 1000 from consolidated results.

Parses es_hyperneat_population_scaling_tables.md to get JIT times
for the largest population size (1000) across all depths (1-10).

Output: TikZ-ready coordinates for fig:baseline_iqr.
"""

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


def format_tikz_coordinates(jit_times: dict[int, float]) -> str:
    """Format JIT times as TikZ coordinates."""
    coords = []
    for depth in sorted(jit_times.keys()):
        coords.append(f"({depth}, {int(round(jit_times[depth]))})")
    return " ".join(coords)


def main():
    # Path to consolidated results (scripts/figures/ -> paper root)
    results_path = Path(__file__).parent.parent.parent / "benchmark_checkpoints" / "es_hyperneat_population_scaling_tables.md"

    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        return

    content = results_path.read_text()

    # Parse JIT data (depths 1-7 and 8-10)
    jit_data = {}
    for depth_range in [(1, 7), (8, 10)]:
        data = parse_jit_table(content, depth_range)
        for d, pops in data.items():
            jit_data[d] = pops

    print("=" * 60)
    print("JIT Compilation Time Extraction for fig:baseline_iqr")
    print(f"Source: {results_path.name}")
    print("=" * 60)

    # Extract Pop 200 and Pop 1000
    pop200_jit = {d: jit_data[d].get(200) for d in jit_data if jit_data[d].get(200) is not None}
    pop1000_jit = {d: jit_data[d].get(1000) for d in jit_data if jit_data[d].get(1000) is not None}

    # Print Pop 200
    print("\n--- Pop 200 JIT Times ---")
    for depth in sorted(pop200_jit.keys()):
        print(f"  D{depth}: {pop200_jit[depth]:,.1f}s")
    print(f"\n  TikZ (D1-D7):")
    pop200_d1_d7 = {d: t for d, t in pop200_jit.items() if d <= 7}
    print(f"  {format_tikz_coordinates(pop200_d1_d7)}")

    # Print Pop 1000
    print("\n--- Pop 1000 JIT Times ---")
    for depth in sorted(pop1000_jit.keys()):
        print(f"  D{depth}: {pop1000_jit[depth]:,.1f}s")
    print(f"\n  TikZ (D1-D10):")
    print(f"  {format_tikz_coordinates(pop1000_jit)}")

    # Print comparison
    print("\n--- Population Scaling Factor (Pop 1000 / Pop 200) ---")
    for depth in sorted(pop200_jit.keys()):
        if depth in pop1000_jit and depth <= 7:
            factor = pop1000_jit[depth] / pop200_jit[depth]
            print(f"  D{depth}: {factor:.2f}x")

    # Print LaTeX snippet
    print("\n" + "=" * 60)
    print("LaTeX Snippet for fig:baseline_iqr:")
    print("=" * 60)
    print("""
% JIT compilation time - Pop 1000 (solid line, circle markers)
\\addplot[color=oiblue, thick, mark=*, solid] coordinates {
    """ + format_tikz_coordinates(pop1000_jit) + """
};
""")


if __name__ == "__main__":
    main()
