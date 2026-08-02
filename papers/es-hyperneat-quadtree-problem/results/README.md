# Raw results

Per-seed result files for the multi-benchmark, CartPole, and HSHG campaigns reported in the paper.

## Baseline CartPole data

The authoritative Baseline CartPole files are the per-depth set:

- `pureples_cartpole_d2_results.json`
- `pureples_cartpole_d3_results.json`
- `pureples_cartpole_d4_results.json`

These are the source for Table 6 and Section 6.6: 6.6 / 56.2 / 1175.7 seconds per generation at
depths 2 / 3 / 4, a 177x D2-to-D4 ratio.

`pureples_cartpole_results.json` covers the same three depths and the same 30 seeds but is **not
comparable to them and should not be used**. Two properties of the file establish this:

1. It produces systematically smaller substrates at every depth: 4.2 / 14.3 / 56.8 mean substrate
   nodes at depths 2 / 3 / 4, against 15.3 / 61.6 / 190.0 in the per-depth files, which is
   consistent with its much shorter runtimes.
2. Three of its 90 runs report a negative `discovered_nodes` value (-1, -2, -3), which no valid
   substrate can produce.

Analysis pointed at this file yields 1.4 / 3.4 / 39.2 seconds per generation and a 28x ratio, which
does not correspond to the configuration the paper reports.

## Failed substrate construction in the HSHG runs

In `hshg_xor_d2`, `hshg_xor_d3`, `hshg_sine_d2` and `hshg_parity3_d2`, some runs produce a substrate
so bloated that no valid network can be assembled. Those runs carry:

- `"construction_failed": true`
- `null` in place of a fitness value, both for the run's summary fields and at the corresponding
  positions in `fitness_history_best` and `fitness_history_mean`.

Every other run carries `"construction_failed": false`. The counts are 5 of 30 for XOR depth 2 and
10 of 30 for each of the other three conditions, which is the 17-33% of seeds the paper reports as
the first of HSHG's two failure modes.

These positions were originally written as the non-standard JSON literal `-Infinity`. That parses in
Python but is rejected by JavaScript, Go, Rust and R, so it was normalised to `null` for the
archive. No measured value was altered.
