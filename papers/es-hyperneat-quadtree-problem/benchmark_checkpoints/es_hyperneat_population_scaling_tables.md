# ES-HyperNEAT Population Scaling Tables

**Generated:** 2026-05-14
**Depths:** 1-7 (all 10 populations); 8-10 (Pop 1,000 only)
**Populations:** 50, 100, 150, 200, 250, 300, 400, 500, 750, 1,000

## Data Sources

| Depth Range | Source |
|-------------|--------|
| 1-7 (JIT and Post-JIT) | `tensorneat_eshyperneat_pop{POP}_10gens/checkpoint.json` (this folder; 10-gen forced reruns, averaged over seeds 42/43/44) |
| 8-10 (Pop 1,000 only) | `tensorneat_eshyperneat_pop1000/checkpoint.json` (D8-D9) and `tensorneat_eshyperneat_pop1000_depth10/checkpoint.json` (D10); 300-generation campaigns, seeds 42/43/44. No other population was run at D8+. |

---

## JIT Compilation Time (seconds)

### Depths 1-7

| Population | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 5 | Depth 6 | Depth 7 |
|------------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| 50 | 18.5 | 29.1 | 55.0 | 151.8 | 436.9 | 708.0 | 739.9 |
| 100 | 22.2 | 51.7 | 101.8 | 289.2 | 857.3 | 1,240.5 | 1,301.7 |
| 150 | 25.9 | 70.5 | 149.0 | 436.0 | 1,327.0 | 2,004.3 | 2,099.5 |
| 200 | 30.5 | 91.4 | 213.3 | 633.9 | 1,903.2 | 2,538.2 | 2,826.5 |
| 250 | 98.0 | 228.2 | 385.7 | 1,112.5 | 3,753.2 | 4,447.0 | 4,610.2 |
| 300 | 35.5 | 111.6 | 265.4 | 741.0 | 2,201.8 | 3,124.4 | 3,328.0 |
| 400 | 82.1 | 235.9 | 649.5 | 1,767.3 | 5,110.5 | 5,956.2 | 6,925.5 |
| 500 | 43.5 | 169.2 | 464.4 | 1,422.8 | 4,466.7 | 6,024.0 | 6,355.8 |
| 750 | 61.3 | 236.9 | 701.1 | 2,175.0 | 6,727.6 | 9,008.0 | 9,497.8 |
| 1,000 | 85.8 | 320.7 | 975.0 | 3,110.4 | 9,985.6 | 13,700.3 | 16,151.3 |

---

## Post-JIT Per-Generation Time (seconds)

### Depths 1-7

| Population | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 5 | Depth 6 | Depth 7 |
|------------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| 50 | 4.1 | 13.0 | 37.3 | 135.4 | 470.6 | 910.4 | 687.3 |
| 100 | 6.9 | 24.4 | 79.5 | 250.2 | 779.4 | 1,522.1 | 1,645.0 |
| 150 | 9.7 | 33.6 | 119.1 | 342.8 | 1,472.7 | 1,974.7 | 2,771.8 |
| 200 | 14.6 | 49.9 | 158.0 | 499.9 | 1,876.6 | 3,389.5 | 3,235.5 |
| 250 | 19.1 | 86.3 | 245.8 | 787.1 | 3,059.6 | 7,069.3 | 6,796.8 |
| 300 | 18.4 | 69.9 | 204.9 | 504.5 | 1,806.7 | 3,541.9 | 3,517.7 |
| 400 | 31.0 | 106.2 | 399.2 | 1,230.7 | 4,100.0 | 6,673.2 | 9,036.8 |
| 500 | 29.5 | 107.8 | 357.1 | 847.9 | 3,192.7 | 4,965.0 | 7,105.8 |
| 750 | 44.0 | 158.6 | 457.0 | 1,173.7 | 3,826.6 | 9,779.2 | 8,329.4 |
| 1,000 | 69.3 | 201.9 | 645.5 | 1,752.7 | 6,988.7 | 10,747.2 | 16,680.1 |

---

## Depths 8-10 (Pop 1,000)

Only Pop 1,000 was run at depths 8-10 (separate 300-generation campaigns; the runs are
construction-dominated, so a steady-state per-generation time exists only where a post-JIT
generation actually completed).

| Depth | Construction / JIT (s, n=3) | Post-JIT Per-Generation (s) |
|-------|----------------------------:|----------------------------:|
| 8 | 14,034.1 | — (construction-only runs) |
| 9 | 16,574.8 | 17,115.5 (single seed) |
| 10 | 21,954.7 | — (construction-only runs) |

---

## Human-Readable Summary

### JIT Time

### Depths 1-7

| Population | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 5 | Depth 6 | Depth 7 |
|------------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| 50 | 18.5s | 29.1s | 55.0s | 2.5m | 7.3m | 11.8m | 12.3m |
| 100 | 22.2s | 51.7s | 1.7m | 4.8m | 14.3m | 20.7m | 21.7m |
| 150 | 25.9s | 1.2m | 2.5m | 7.3m | 22.1m | 33.4m | 35.0m |
| 200 | 30.5s | 1.5m | 3.6m | 10.6m | 31.7m | 42.3m | 47.1m |
| 250 | 1.6m | 3.8m | 6.4m | 18.5m | 1.0h | 1.2h | 1.3h |
| 300 | 35.5s | 1.9m | 4.4m | 12.4m | 36.7m | 52.1m | 55.5m |
| 400 | 1.4m | 3.9m | 10.8m | 29.5m | 1.4h | 1.7h | 1.9h |
| 500 | 43.5s | 2.8m | 7.7m | 23.7m | 1.2h | 1.7h | 1.8h |
| 750 | 1.0m | 3.9m | 11.7m | 36.3m | 1.9h | 2.5h | 2.6h |
| 1,000 | 1.4m | 5.3m | 16.3m | 51.8m | 2.8h | 3.8h | 4.5h |

### Post-JIT Per-Generation Time

### Depths 1-7

| Population | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 5 | Depth 6 | Depth 7 |
|------------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| 50 | 4.1s | 13.0s | 37.3s | 2.3m | 7.8m | 15.2m | 11.5m |
| 100 | 6.9s | 24.4s | 1.3m | 4.2m | 13.0m | 25.4m | 27.4m |
| 150 | 9.7s | 33.6s | 2.0m | 5.7m | 24.5m | 32.9m | 46.2m |
| 200 | 14.6s | 49.9s | 2.6m | 8.3m | 31.3m | 56.5m | 53.9m |
| 250 | 19.1s | 1.4m | 4.1m | 13.1m | 51.0m | 2.0h | 1.9h |
| 300 | 18.4s | 1.2m | 3.4m | 8.4m | 30.1m | 59.0m | 58.6m |
| 400 | 31.0s | 1.8m | 6.7m | 20.5m | 1.1h | 1.9h | 2.5h |
| 500 | 29.5s | 1.8m | 6.0m | 14.1m | 53.2m | 1.4h | 2.0h |
| 750 | 44.0s | 2.6m | 7.6m | 19.6m | 1.1h | 2.7h | 2.3h |
| 1,000 | 1.2m | 3.4m | 10.8m | 29.2m | 1.9h | 3.0h | 4.6h |

---

## Quadtree Position Counts by Depth

| Depth | Positions |
|-------|-----------|
| 1 | 20 |
| 2 | 84 |
| 3 | 340 |
| 4 | 1,364 |
| 5 | 5,460 |
| 6 | 21,844 |
| 7 | 87,380 |
| 8 | 349,524 |
| 9 | 1,398,100 |
| 10 | 5,592,404 |

---

## Provenance

The Depths 1-7 tables are per-cell means (seeds 42/43/44) over the per-population
`tensorneat_eshyperneat_pop{POP}_10gens/checkpoint.json` files in this folder; the Depths 8-10
values come from the Pop-1,000 campaigns (`tensorneat_eshyperneat_pop1000/`,
`tensorneat_eshyperneat_pop1000_depth10/`). `scripts/analysis/compute_table_values.py` and the
`scripts/figures/extract_*.py` helpers consume this file.
