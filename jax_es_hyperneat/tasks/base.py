"""Task definitions for the JAX-ESHN benchmarks.

Each task provides the substrate input/output coordinates, the success (fitness)
threshold, and a ``Problem`` object exposing ``get_data() -> [(input, target), ...]``
of float32 numpy arrays. These are the exact definitions used by the benchmark
runners that produced the paper's results; the algorithm computes fitness internally
as ``1 - MSE`` over ``get_data()``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

CoordList = List[Tuple[float, float]]

# Quadtree child-node count by max_depth at initial_depth=0, summed over levels 1..d+1
# (= (4^(d+2) - 4) / 3). This is the `positions` value the runners record in results JSON.
DEPTH_POSITIONS = {1: 20, 2: 84, 3: 340, 4: 1364, 5: 5460, 6: 21844, 7: 87380}

# Shared ES-HyperNEAT substrate-discovery hyperparameters, code-true (as used by the
# benchmark runners that produced the published JAX-ESHN results).
#
# NOTE: max_weight=8.0 and band_threshold=0.3 are the values the JAX-ESHN runs actually
# used. The paper's reproducibility table lists 5.0 / 0.2 for some columns; those apply
# to the PUREPLES baseline / a different configuration. See papers/.../REPRODUCIBILITY.md.
ES_PARAMS = dict(
    initial_depth=0,
    variance_threshold=0.03,
    division_threshold=0.5,
    band_threshold=0.3,
    max_weight=8.0,
    iteration_level=1,
)


@dataclass
class Task:
    """A benchmark task: substrate geometry, success threshold, and data source."""

    name: str
    input_coords: CoordList
    output_coords: CoordList
    fitness_threshold: float
    make_problem: Callable[[], object]
    default_depths: List[int]
    is_gym: bool = False  # CartPole uses a gym episode loop instead of run_generation

    def build_config(self, algo, depth: int, population_size: int):
        """Build the algorithm config dict consumed by ``algo.create_config``."""
        return algo.create_config({
            'algorithm_params': {
                'eshyperneat': {
                    'population_size': population_size,
                    'es_hyperneat': {'max_depth': depth, **ES_PARAMS},
                    'substrate': {
                        'input_coords': self.input_coords,
                        'output_coords': self.output_coords,
                        'output_activation': 'sigmoid',
                        'hidden_activation': 'tanh',
                    },
                }
            }
        })
