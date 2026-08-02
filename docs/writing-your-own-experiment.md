# Running and writing experiments

Three ways to use the code, from highest-level to lowest:

1. The **driver CLI**: run a benchmark over depths × seeds with checkpoint/resume.
2. The **`run_single` harness**: one Python call per `(task, depth, seed)`, returns a `RunResult`.
3. The **algorithm API**: drive `initialize` / `run_generation` yourself.

Then: the **Task + Problem interface** and a recipe to **add your own task**.

> Set `JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=3` for quiet CPU runs (see
> [installation.md](installation.md)).

## 1. The driver CLI

There is no console script and no `python -m jax_es_hyperneat`; invoke the driver module directly:

```bash
python -m jax_es_hyperneat.driver --task xor  --impl jax-eshn --depths 2 --pop 150 --gens 60 --seeds 42-44
python -m jax_es_hyperneat.driver --task sine --impl jax-eshn --depths 2 3 4 --seeds 0-29
python -m jax_es_hyperneat.driver --task xor  --impl hshg     --depths 2 --pop 50 --gens 50 --seeds 0-4
```

### Flags

| Flag | Default | Choices / format | Notes |
|------|---------|------------------|-------|
| `--task` | *(required)* | `xor` `parity3` `circle` `sine` `cartpole` | the experiment task |
| `--impl` | `jax-eshn` | `jax-eshn` `pureples` `hshg` | which implementation to run |
| `--depths` | task's `default_depths` | one or more ints (`--depths 2 3 4`) | quadtree `max_depth` values |
| `--pop` | `150` | int | population size; **ignored by `pureples`** (fixed at 150 by its neat config) |
| `--gens` | `100` | int | generations per run; jax-eshn/hshg run the full count, PUREPLES stops early at its neat config `fitness_threshold` (0.975, or 0.95 for cartpole; distinct from the task solve threshold) |
| `--seeds` | `0-29` | range `"0-29"` or list `"42,43,44"` (inclusive ranges) | seeds to sweep |
| `--threshold` | task's threshold | float | override the solve threshold |
| `--backend` | `cpu` | `cpu` `gpu` | sets `JAX_PLATFORM_NAME` only if not already exported (distinct from the `JAX_PLATFORMS` used elsewhere in these docs; an exported value wins) |
| `--timeout` | `86400` | int (seconds) | per-run wall-clock cap |
| `--out` | `results` | path | output dir (relative to cwd) |
| `--no-resume` | off | flag | ignore an existing checkpoint and re-run all |
| `--quiet` | off | flag | less console output |

### Output

Written into `--out` (default `./results/`), with `prefix` = `jax_eshn` / `pureples` / `hshg`:

- `<prefix>_<task>_results.json`: final results.
  - `jax-eshn`: `{"metadata", "summary", "results"}` (`summary` is `summarize()` per depth).
  - `pureples` / `hshg`: `{"metadata", "runs"}`.
- `<prefix>_<task>_checkpoint.json`: written after every run; **resume is on by default**, so
  re-invoking skips already-completed `d{depth}_s{seed}` runs (use `--no-resume` to force a redo).

## 2. The `run_single` harness (one run in Python)

```python
from jax_es_hyperneat.tasks import TASKS
from jax_es_hyperneat.run_jax_eshn import run_single

config = {
    "population_size": 150,
    "max_generations": 60,
    "fitness_threshold": 0.98,
    "timeout_per_run_s": 86400,
}
result = run_single(TASKS["xor"], depth=2, seed=42, config=config, verbose=False)
print(result.solved, result.fitness, result.solved_at_gen)
print(result.to_dict())   # all fields as a plain dict
```

`run_single(task, depth, seed, config, verbose=True) -> RunResult`. All four `config` keys above are
**required**. The same signature works for the baseline and ablation runners
(`from jax_es_hyperneat.baseline.pureples_harness import run_single`,
`from jax_es_hyperneat.hshg.run_hshg import run_single`). Those return a plain dict
(`solved`, `best_fitness`, `solved_gen`, …) rather than a `RunResult`.

### `RunResult` fields

`implementation`, `depth`, `seed`, `problem`, `solved` (bool), `fitness` (float), `generations`,
`jit_compilation_time_s`, `time_per_gen_post_jit_s`, `total_evolution_time_s`, `total_time_s`,
`solved_at_gen` (int or `None`), `positions`, `gpu_util_avg`, `gpu_util_max`, `vram_peak_mb`,
`error` (str or `None`), `timestamp`. A failure during problem-build / initialize / evolution is
captured as `solved=False, fitness=0.0, generations=0` with `error` set, rather than propagating.
(A `config` missing `fitness_threshold` or `population_size` is read before that try/except guard
and still raises `KeyError`.)

`summarize(results)` groups a list of result dicts by depth into `{"d2": {...}, ...}` with
`depth`, `solve_rate`, `n_runs`, `n_errors`, `avg_jit_time_s`, `std_jit_time_s`,
`avg_time_per_gen_post_jit_s`, `std_time_per_gen_post_jit_s`, `avg_fitness`, `max_fitness`,
`positions` (this is what the driver stores under `summary`).

## 3. The algorithm API (drive the loop yourself)

```python
from jax_es_hyperneat import JAXESHyperNEAT          # alias of TensorNEATESHyperNEATOptimized
from jax_es_hyperneat.tasks import TASKS

task    = TASKS["xor"]
algo    = JAXESHyperNEAT()
problem = task.make_problem()
cfg     = task.build_config(algo, depth=2, population_size=150)
state   = algo.initialize(cfg, problem, seed=42)
for _ in range(60):
    state, metrics = algo.run_generation(state, problem)
print(float(metrics.best_fitness))
```

Use this when you want per-generation control (logging, custom stopping, etc.). For gym tasks the
per-generation step is `run_gym_generation(algo, state, population_size)` instead of
`run_generation` (see CartPole below).

## The Task + Problem interface

A **`Task`** (`jax_es_hyperneat/tasks/base.py`) is a dataclass:

| Field | Meaning |
|-------|---------|
| `name` | registry key; also selects the pureples config + hshg experiment |
| `input_coords` | list of `(x, y)` substrate coordinates, **one per input including the bias** |
| `output_coords` | list of `(x, y)`, one per output |
| `fitness_threshold` | `solved` when best fitness ≥ this |
| `make_problem` | the Problem **class** (called as `task.make_problem()`) |
| `default_depths` | depths used when `--depths` is omitted |
| `is_gym` | `False` for static data tasks; `True` routes through a gym episode loop |

`task.build_config(algo, depth, population_size)` assembles the algorithm config (it hardcodes
`output_activation='sigmoid'`, `hidden_activation='tanh'`, and merges the shared
`ES_PARAMS`). Convention: inputs sit at `y=-1`, the output at `y=+1`.

A **Problem** is a plain class (a protocol, no base class) with:

- class attributes `use_bias: bool`, `input_size: int`, `output_size: int`;
- `get_data() -> [(input, target), ...]` where each pair is **float32 numpy arrays**, the bias baked
  in as a trailing `1.0` column;
- a zero-arg constructor (so `task.make_problem()` works).

Fitness is computed internally as **`max(0, 1 − MSE)`** over `get_data()` (clamped at 0). Across all tasks
`len(input_coords) == input_size` and `len(output_coords) == output_size`. Shared discovery
hyperparameters live in `ES_PARAMS` (`initial_depth=0`, `variance_threshold=0.03`,
`division_threshold=0.5`, `band_threshold=0.3`, `max_weight=8.0`, `iteration_level=1`);
`DEPTH_POSITIONS` maps `max_depth → quadtree node count = (4^(d+2)-4)/3`.

## Recipe: add your own static task (JAX-ESHN)

A static, data-based task on the primary `jax-eshn` implementation needs **three edits**.

**1. Create `jax_es_hyperneat/tasks/<name>.py`**, a Problem class and a module-level `TASK`
(this is the `xor.py` shape, abbreviated):

```python
import numpy as np
from .base import Task

class XORProblem:
    use_bias = True
    input_size = 3   # 2 bits + bias
    output_size = 1

    def get_data(self):
        inputs = np.array([[0.0, 0.0, 1.0],
                           [0.0, 1.0, 1.0],
                           [1.0, 0.0, 1.0],
                           [1.0, 1.0, 1.0]], dtype=np.float32)
        targets = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)
        return [(inputs[i], targets[i]) for i in range(len(inputs))]

TASK = Task(
    name="xor",
    input_coords=[(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)],  # 3 inputs at y=-1 (bias included)
    output_coords=[(0.0, 1.0)],                              # 1 output at y=+1
    fitness_threshold=0.98,
    make_problem=XORProblem,
    default_depths=[1, 2, 3, 4, 5, 6, 7],
)
```

**2. Register it in `jax_es_hyperneat/tasks/__init__.py`**: add `from . import <name>` and
`"<name>": <name>.TASK` to the `TASKS` dict.

**3. Add `"<name>"` to the driver's `--task` choices** in `jax_es_hyperneat/driver.py`: this list
is hardcoded (not derived from `TASKS`), so the CLI rejects an unregistered name otherwise.

Then run it:

```bash
python -m jax_es_hyperneat.driver --task <name> --impl jax-eshn --depths 2 3 4 --pop 150 --gens 100 --seeds 0-29
```

## Extending to the other implementations

- **PUREPLES baseline.** The PUREPLES harness does not use your `Problem.get_data()`; it carries
  PUREPLES-faithful data in `baseline/pureples_harness.py::_pureples_data` and reads a neat-python
  config `baseline/configs/config_cppn_<name>`. To run a new task on `--impl pureples` you must add
  both (and population is fixed at 150 by the neat config).
- **HSHG ablation.** `hshg/run_hshg.py` dispatches by `task.name` through `_EXPERIMENT_MODULES` to a
  per-task `hshg/experiments/<name>.py` exposing `HSHGESHyperNEATExperiment` + a `<Name>Config`. A
  new task on `--impl hshg` needs that module and a registry entry.
- **Gym tasks.** The `is_gym=True` path is **CartPole-specific**: `run_jax_eshn.py`
  hardcodes `from .tasks.cartpole import run_gym_generation`, and that function is wired to
  `gym.make('CartPole-v1')`. A genuinely new gym task requires its own `run_gym_generation`/evaluator
  and an edit to `run_jax_eshn.py`. Static tasks have no such coupling; they go through the generic
  `algo.run_generation(state, problem)`. CartPole's pattern (for reference,
  `jax_es_hyperneat/tasks/cartpole.py`):

```python
class CartPoleProblem:
    use_bias = True
    input_size = 5   # 4 observations + bias
    output_size = 1
    def get_data(self):
        return [(np.zeros(5, dtype=np.float32), np.zeros(1, dtype=np.float32))]  # gym, not static data

TASK = Task(name="cartpole",
            input_coords=[(-2.,-1.),(-1.,-1.),(0.,-1.),(1.,-1.),(2.,-1.)],
            output_coords=[(0.0, 1.0)], fitness_threshold=0.95,
            make_problem=CartPoleProblem, default_depths=[2, 3, 4], is_gym=True)
```

## Task-specific gotchas (already in the code)

- **Sine** regresses `(sin(πx)+1)/2 ∈ [0,1]` on the JAX-ESHN side but raw `sin(πx)` on the PUREPLES
  side. This is deliberate; do not harmonize (it is why PUREPLES plateaus ~0.73 on sine).
- **Circle** seeds its 100 points with `np.random.RandomState(42)` so the dataset is fixed across
  runs; label is `x²+y² < 0.25` (radius 0.5).

## Next

- Reproduce the paper's experiments: [reproducing-the-paper.md](reproducing-the-paper.md)
- The class internals at a glance: [architecture.md](architecture.md)
