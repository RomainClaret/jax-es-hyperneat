# PUREPLES ES-HyperNEAT Depth 6 Benchmark - PARTIAL RESULTS

**Generated:** 2026-01-02 (interrupted)
**Status:** 18/72 runs complete (25%)
**Remaining:** Runs 19-72 (pop=750, pop=1000 in Tier 1, then all Tier 2)

## Parameters

- **Generations:** 30
- **Populations:** 50, 100, 200, 300, 400, 500, 750, 1000
- **Depth (max_depth):** 6
- **Seeds:** 42, 123, 456
- **Iteration Levels:** 2, 3
- **Target Fitness:** 0.99

## Completed Runs (Tier 1 only)

| Run | Pop | Seed | Gen Time (s) | Fitness | Solved |
|-----|-----|------|--------------|---------|--------|
| 1 | 50 | 42 | 33.95 | 0.7500 | No |
| 2 | 50 | 123 | 4.51 | 0.7505 | No |
| 3 | 50 | 456 | 34.84 | 0.7500 | No |
| 4 | 100 | 42 | 75.24 | 0.8125 | No |
| 5 | 100 | 123 | 14.97 | 0.8125 | No |
| 6 | 100 | 456 | 47.02 | 0.7500 | No |
| 7 | 200 | 42 | 33.30 | 0.8119 | No |
| 8 | 200 | 123 | 190.57 | 0.7953 | No |
| 9 | 200 | 456 | 92.59 | 0.7500 | No |
| **10** | **300** | **42** | **251.90** | **1.0000** | **Yes @ Gen 14** |
| 11 | 300 | 123 | 119.28 | 0.8120 | No |
| 12 | 300 | 456 | 72.10 | 0.8056 | No |
| 13 | 400 | 42 | 169.86 | 0.8138 | No |
| 14 | 400 | 123 | 283.38 | 0.7500 | No |
| 15 | 400 | 456 | 80.48 | 0.7500 | No |
| 16 | 500 | 42 | 445.42 | 0.8125 | No |
| 17 | 500 | 123 | 85.04 | 0.8125 | No |
| 18 | 500 | 456 | 131.17 | 0.7771 | No |

## Summary Statistics (Partial)

| Metric | Value |
|--------|-------|
| Completed | 18/72 (25%) |
| Solve Rate | 1/18 (5.6%) |
| Best Fitness | 1.0000 |
| Avg Gen Time | 120.3s |
| Max Gen Time | 445.42s (pop=500 seed=42) |
| Avg Fitness (unsolved) | 0.7870 |

## Remaining Runs to Complete

### Tier 1 (runs 19-24)
- pop=750: seeds 42, 123, 456
- pop=1000: seeds 42, 123, 456

### Tier 2 iter=2 (runs 25-48)
- All populations (50-1000) × 3 seeds

### Tier 2 iter=3 (runs 49-72)
- All populations (50-1000) × 3 seeds

## Command to Resume

```bash
# Resume with remaining Tier 1 populations
PYTHONPATH=src python scripts/benchmarks/hmr_optimization/benchmark_pureples_eshyperneat_comprehensive.py \
    --depths 6 \
    --populations 750 1000 \
    --seeds 42 123 456 \
    --generations 30 \
    --iteration-levels 2 3 \
    --output scripts/benchmarks/hmr_optimization/results/pureples_depth6_remaining_t1

# Then run Tier 2 separately if needed
```

## Key Observations

1. **Depth 6 CAN solve XOR** - pop=300 seed=42 solved at generation 14
2. **Extremely slow** - gen times range from 4.5s to 445s
3. **5.6% solve rate** vs 36.7% at depths 1-5
4. **High variability** - same population can have 10x different gen times across seeds
5. **Pop=500+ is brutal** - single runs taking 1-4+ hours

## Comparison with Depth 5

| Metric | Depth 5 | Depth 6 (partial) |
|--------|---------|-------------------|
| Solve Rate | 36.7% | 5.6% |
| Avg Gen Time | 3.68s | 120.3s |
| Max Gen Time | 46.26s | 445.42s |
| Slowdown | baseline | ~33x slower |
