# TensorNEAT ES-HyperNEAT Benchmark Results

**Implementation**: `tensorneat-eshyperneat` (JAX, CPU backend)
**Hardware**: NVIDIA GeForce RTX 2080 Ti (11 GB) — JAX running on CPU
**Configuration**: Population 150, 100 generations, 30 seeds per depth. Fitness threshold: 0.95 (Sine, CartPole), 0.975 (Circle, 3-Parity).
**Depths**: 2 (84 positions), 3 (340 positions), 4 (1,364 positions)

---

## Results Files

| Problem | Re-run with | Results | Checkpoint |
|---------|-------------|---------|------------|
| Sine | `python -m jax_es_hyperneat.driver --task sine --impl jax-eshn` | `results/jax_eshn_sine_results.json` | `results/jax_eshn_sine_checkpoint.json` |
| Circle | `python -m jax_es_hyperneat.driver --task circle --impl jax-eshn` | `results/jax_eshn_circle_results.json` | `results/jax_eshn_circle_checkpoint.json` |
| 3-Parity | `python -m jax_es_hyperneat.driver --task parity3 --impl jax-eshn` | `results/jax_eshn_parity3_results.json` | `results/jax_eshn_parity3_checkpoint.json` |
| CartPole | `python -m jax_es_hyperneat.driver --task cartpole --impl jax-eshn` | `results/cartpole_d{2,3,4}/jax_eshn_cartpole_results.json` | `results/cartpole_d{2,3,4}/jax_eshn_cartpole_checkpoint.json` |

All paths relative to `papers/es-hyperneat-quadtree-problem/`. CartPole ran as 3 per-depth processes with separate output directories.

---

## Solve Rate

| Problem | Depth 2 | Depth 3 | Depth 4 |
|---------|---------|---------|---------|
| **Sine** | 29/30 (96.7%) | 30/30 (100.0%) | 30/30 (100.0%) |
| **Circle** | 0/30 (0.0%) | 0/30 (0.0%) | 0/30 (0.0%) |
| **3-Parity** | 4/30 (13.3%) | 7/30 (23.3%) | 5/30 (16.7%) |
| **CartPole** | 30/30 (100.0%) | 30/30 (100.0%) | 30/30 (100.0%) |

---

## Speed per Generation (seconds, post-JIT)

| Problem | Depth 2 | Depth 3 | Depth 4 |
|---------|---------|---------|---------|
| **Sine** | 13.1 ± 0.9 | 27.4 ± 4.1 | 65.8 ± 16.9 |
| **Circle** | 16.4 ± 1.0 | 33.5 ± 7.2 | 78.1 ± 24.9 |
| **3-Parity** | 13.6 ± 1.1 | 27.3 ± 3.7 | 65.9 ± 14.6 |
| **CartPole** | 312.6 ± 107.6 | 476.7 ± 118.5 | 964.9 ± 348.7 |

Values are mean ± std across all seeds at that depth.

---

## JIT Compilation Time (seconds)

| Problem | Depth 2 | Depth 3 | Depth 4 |
|---------|---------|---------|---------|
| **Sine** | 56.4 ± 2.2 | 117.2 ± 4.7 | 328.8 ± 36.2 |
| **Circle** | 60.3 ± 2.8 | 127.7 ± 5.0 | 384.6 ± 53.1 |
| **3-Parity** | 60.6 ± 2.8 | 128.7 ± 4.4 | 419.3 ± 49.8 |
| **CartPole** | 558.9 ± 92.9 | 849.0 ± 130.3 | 1,579.3 ± 243.0 |

---

## Total Wall Time per Run (seconds)

| Problem | Depth 2 | Depth 3 | Depth 4 |
|---------|---------|---------|---------|
| **Sine** | 1,350 (22 min) | 2,825 (47 min) | 6,841 (1.9 hr) |
| **Circle** | 1,685 (28 min) | 3,447 (57 min) | 8,113 (2.3 hr) |
| **3-Parity** | 1,406 (23 min) | 2,827 (47 min) | 6,947 (1.9 hr) |
| **CartPole** | 31,506 (8.8 hr) | 48,042 (13.3 hr) | 79,424 (22.1 hr) |

Values are mean total time (JIT + 100 generations).

---

## Fitness Distribution (unsolved runs only)

For problems with <100% solve rate, the fitness of unsolved runs:

| Problem | Depth | N (unsolved) | Mean Fitness | Min Fitness | Max Fitness |
|---------|-------|-------------|-------------|------------|------------|
| **Sine** | d2 | 1 | 0.948 | 0.948 | 0.948 |
| **Circle** | d2 | 30 | 0.867 | 0.852 | 0.887 |
| **Circle** | d3 | 30 | 0.865 | 0.850 | 0.880 |
| **Circle** | d4 | 30 | 0.862 | 0.842 | 0.871 |
| **3-Parity** | d2 | 26 | 0.861 | 0.796 | 0.955 |
| **3-Parity** | d3 | 23 | 0.870 | 0.812 | 0.936 |
| **3-Parity** | d4 | 25 | 0.877 | 0.840 | 0.965 |

---

## Solved-At Generation (successful runs only)

| Problem | Depth | Min | Median | Max |
|---------|-------|-----|--------|-----|
| **Sine** | d2 | 1 | 2 | 12 |
| **Sine** | d3 | 1 | 2 | 80 |
| **Sine** | d4 | 1 | 2 | 57 |
| **3-Parity** | d2 | 8 | 15 | 28 |
| **3-Parity** | d3 | 1 | 7 | 65 |
| **3-Parity** | d4 | 11 | 13 | 41 |
| **CartPole** | d2 | 1 | 1 | 2 |
| **CartPole** | d3 | 1 | 1 | 2 |
| **CartPole** | d4 | 1 | 1 | 3 |

---

## Depth Scaling (speed relative to depth 2)

| Problem | d2 → d3 | d2 → d4 |
|---------|---------|---------|
| **Sine** | 2.1× | 5.0× |
| **Circle** | 2.0× | 4.8× |
| **3-Parity** | 2.0× | 4.8× |
| **CartPole** | 1.5× | 3.1× |

Based on mean s/gen. Boolean tasks scale ~2× per depth level (4× more quadtree positions). CartPole scales sub-linearly due to fixed RL evaluation overhead dominating at lower depths.

---

## Key Observations

1. **Circle is unsolvable** at all depths (0/90). Max fitness plateaus at ~0.87, well below the 0.975 threshold. This is a known architecture limitation — ES-HyperNEAT's CPPN-based connectivity cannot capture the circular decision boundary.

2. **CartPole is trivially solved** — every seed solves at generation 1-3. The benchmark measures computational overhead, not algorithmic difficulty. CartPole is 15-20× slower per generation than Boolean tasks due to RL episode evaluation (200 timesteps × population).

3. **3-Parity peaks at depth 3** (23.3%) — the non-monotonic solve rate suggests depth 3 provides enough resolution for some seeds while depth 4's larger search space (1,364 positions vs 340) hurts convergence more than it helps.

4. **Sine is near-trivial** — 96.7-100% solve rate, most seeds solved by generation 2. The single depth-2 failure (fitness 0.948) was close to threshold.

5. **Speed scales ~5× from d2 to d4** for Boolean tasks (84 → 1,364 positions, a 16× increase in quadtree nodes). The sub-linear scaling suggests fixed overhead in NEAT operations dominates at lower depths.

---

*Last updated: 2026-04-19. All results FINAL — 360/360 runs complete (4 benchmarks × 3 depths × 30 seeds).*
