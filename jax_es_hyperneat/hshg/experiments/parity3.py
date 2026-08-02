"""
HSHG ES-HyperNEAT Parity-3 experiment.
Uses HSHG-based ES-HyperNEAT reimplementation with TensorNEAT for Parity-3 problem.

Identical to tensorneat_parity3 except substrate discovery uses HSHG instead of quadtree.
"""

import os
import sys
import time
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

# Force CPU for fair comparison - ES-HyperNEAT discovery is CPU-based
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

# CRITICAL: Force CPU in JAX config before any JAX operations
# This prevents Metal backend initialization and memory leaks
jax.config.update('jax_platform_name', 'cpu')
print(f"[CPU-ONLY MODE] JAX devices: {jax.devices()}")
print(f"[CPU-ONLY MODE] JAX default backend: {jax.default_backend()}")

# Import TensorNEAT components
from tensorneat import algorithm, genome, problem
from tensorneat.common import ACT, AGG, State
from tensorneat.pipeline import Pipeline

# HSHG ES-HyperNEAT components (one package up).
from ..eshyperneat_hshg import ESHyperNEATHSHG
from ..substrate import ESHyperNEATSubstrate


@dataclass
class Parity3Config:
    """Configuration for Parity-3 ES-HyperNEAT experiment matching PUREPLES."""
    # NEAT parameters for evolving CPPN (matching PUREPLES config_cppn_parity3)
    pop_size: int = 150
    species_size: int = 15
    survival_threshold: float = 0.2
    max_nodes: int = 20

    # ES-HyperNEAT specific parameters (matching PUREPLES)
    band_threshold: float = 0.2  # PUREPLES hardcodes 0.2 in query_cppn
    variance_threshold: float = 0.03
    division_threshold: float = 0.5
    initial_depth: int = 0  # Small version
    max_depth: int = 1  # Small version
    iteration_level: int = 1
    max_weight: float = 5.0
    activation: str = 'sigmoid'

    # Training parameters
    max_generations: int = 300
    fitness_target: float = -0.025  # TensorNEAT uses negative MSE
    seed: int = 42

    # Substrate coordinates matching PUREPLES
    # Four inputs: x1, x2, x3, bias (evenly spaced)
    input_coors: Tuple[Tuple[float, float], ...] = ((-1.5, -1.0), (-0.5, -1.0), (0.5, -1.0), (1.5, -1.0))
    output_coors: Tuple[Tuple[float, float], ...] = ((0.0, 1.0),)

    # CPPN activation functions matching PUREPLES
    cppn_activations: Tuple[str, ...] = ('tanh', 'sin', 'gauss')


class Parity3Problem(problem.FuncFit):
    """Parity-3 problem implementation with bias input to match PUREPLES."""

    def __init__(self):
        super().__init__(error_method="mse")

    @property
    def inputs(self):
        # Parity-3 inputs with bias column to match PUREPLES
        # 3 binary inputs + 1 bias = 4 inputs per pattern, 8 patterns total
        return np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 1],
                [0, 1, 0, 1],
                [0, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.float32,
        )

    @property
    def targets(self):
        # Output is 1 if odd number of 1s in the 3 inputs, else 0
        return np.array(
            [[0], [1], [1], [0], [1], [0], [0], [1]],
            dtype=np.float32,
        )

    @property
    def input_shape(self):
        return 8, 4  # 8 samples, 4 inputs (x1, x2, x3, bias)

    @property
    def output_shape(self):
        return 8, 1  # 8 samples, 1 output


class HSHGESHyperNEATExperiment:
    """HSHG ES-HyperNEAT experiment for Parity-3."""

    def __init__(self, config: Optional[Parity3Config] = None,
                 save_checkpoints: bool = False,
                 checkpoint_dir: Optional[str] = None,
                 version: str = 'S'):
        """
        Initialize the experiment.

        Args:
            config: Experiment configuration
            save_checkpoints: Whether to save checkpoints
            checkpoint_dir: Directory for checkpoints
            version: 'S' (small), 'M' (medium), or 'L' (large) - affects ES-HyperNEAT depth
        """
        self.config = config or Parity3Config()
        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = checkpoint_dir
        self.version = version

        # Adjust ES-HyperNEAT parameters based on version
        if version == 'M':
            self.config.initial_depth = 1
            self.config.max_depth = 2
        elif version == 'L':
            self.config.initial_depth = 2
            self.config.max_depth = 3

        # Create checkpoint directory if needed
        if save_checkpoints and checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize tracking
        self.generation_history = []
        self.best_genomes = []
        self.es_metrics_history = []
        self.start_time = None
        self.algorithm = None
        self.problem = None
        self._setup_experiment()

    def _setup_experiment(self):
        """Set up HSHG ES-HyperNEAT algorithm and problem."""
        # Create substrate matching PUREPLES
        substrate = ESHyperNEATSubstrate(
            input_coordinates=list(self.config.input_coors),
            hidden_coordinates=[],  # ES-HyperNEAT discovers these
            output_coordinates=list(self.config.output_coors),
            include_bias=False,  # Bias is already in input coordinates
            band_threshold=self.config.band_threshold,  # Now 0.2 to match PUREPLES
            variance_threshold=self.config.variance_threshold,
        )

        # Create CPPN genome (for evolving connection patterns)
        cppn_genome = genome.DefaultGenome(
            num_inputs=5,  # x1, y1, x2, y2, bias
            num_outputs=1,  # connection weight
            max_nodes=self.config.max_nodes,
            output_transform=ACT.tanh,  # CPPN output activation
        )

        # Create NEAT algorithm for evolving CPPN
        neat_algo = algorithm.NEAT(
            genome=cppn_genome,
            pop_size=self.config.pop_size,
            species_size=self.config.species_size,
            survival_threshold=self.config.survival_threshold,
            verbose=False,
        )

        # Create HSHG ES-HyperNEAT algorithm (HSHG instead of quadtree)
        self.algorithm = ESHyperNEATHSHG(
            substrate=substrate,
            neat=neat_algo,
            max_weight=self.config.max_weight,
            band_threshold=self.config.band_threshold,
            initial_depth=self.config.initial_depth,
            max_depth=self.config.max_depth,
            variance_threshold=self.config.variance_threshold,
            division_threshold=self.config.division_threshold,
            iteration_level=self.config.iteration_level,
            activation=ACT.sigmoid if self.config.activation == 'sigmoid' else ACT.tanh,
            activate_time=1,
            output_transform=ACT.identity,
        )

        # Create Parity-3 problem
        self.problem = Parity3Problem()

    def run(self):
        """
        Run the HSHG ES-HyperNEAT experiment.

        Returns:
            Tuple of (final_state, statistics_dict)
        """
        self.start_time = time.time()

        # Initialize state with random key
        state = State(randkey=jax.random.PRNGKey(self.config.seed))
        state = self.algorithm.setup(state)

        # Initialize tracking
        best_fitness_history = []
        mean_fitness_history = []
        min_fitness_history = []
        es_metrics_history = []
        cppn_nodes_history = []
        cppn_connections_history = []

        print(f"\nRunning HSHG ES-HyperNEAT Parity-3...")
        print(f"Population size: {self.config.pop_size}")
        print(f"ES-HyperNEAT parameters:")
        print(f"  Initial depth: {self.config.initial_depth}")
        print(f"  Max depth: {self.config.max_depth}")
        print(f"  Band threshold: {self.config.band_threshold}")
        print(f"  Variance threshold: {self.config.variance_threshold}")

        best_fitness_overall = float('-inf')
        best_genome = None
        best_metrics = None

        for generation in range(self.config.max_generations):
            # Get current population
            pop = self.algorithm.ask(state)

            # Extract metrics about discovered topology
            discovered_nodes_list = []
            substrate_nodes_list = []
            total_connections_list = []
            cppn_nodes_list = []
            cppn_connections_list = []

            # Transform population and evaluate
            fitnesses = []
            for i in range(len(pop[0])):
                individual = (pop[0][i], pop[1][i])

                # Transform CPPN to substrate via HSHG ES-HyperNEAT discovery
                try:
                    transformed = self.algorithm.transform(state, individual)

                    # Evaluate fitness
                    fitness = self.problem.evaluate(state, None, self.algorithm.forward, transformed)
                    fitnesses.append(fitness)

                    # Extract metrics from the discovery process
                    cppn_nodes = int(jnp.sum(~jnp.isnan(individual[0][:, 0])))
                    cppn_connections = int(jnp.sum(~jnp.isnan(individual[1][:, 0])))

                    cppn_nodes_list.append(cppn_nodes)
                    cppn_connections_list.append(cppn_connections)

                    # Extract ES-HyperNEAT discovery metrics
                    if hasattr(self.algorithm, 'last_discovery_metrics'):
                        discovered_nodes_list.append(self.algorithm.last_discovery_metrics['discovered_nodes'])
                        substrate_nodes_list.append(self.algorithm.last_discovery_metrics['substrate_nodes'])
                        total_connections_list.append(self.algorithm.last_discovery_metrics['total_connections'])
                    else:
                        discovered_nodes_list.append(0)
                        substrate_nodes_list.append(len(self.config.input_coors) + len(self.config.output_coors))
                        total_connections_list.append(0)

                except Exception as e:
                    print(f"Warning: Transform failed for individual {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    fitnesses.append(float('-inf'))
                    cppn_nodes_list.append(0)
                    cppn_connections_list.append(0)

            fitnesses = jnp.array(fitnesses)

            # Update state with fitness
            state = self.algorithm.tell(state, fitnesses)

            # Convert negative MSE to positive accuracy
            accuracies = 1.0 + fitnesses  # Since fitness is -MSE

            # Track statistics
            best_idx = jnp.argmax(fitnesses)
            best_fitness = float(accuracies[best_idx])
            mean_fitness = float(jnp.mean(accuracies))
            min_fitness = float(jnp.min(accuracies))

            best_fitness_history.append(best_fitness)
            mean_fitness_history.append(mean_fitness)
            min_fitness_history.append(min_fitness)

            # Track best overall
            if best_fitness > best_fitness_overall:
                best_fitness_overall = best_fitness
                best_genome = (pop[0][best_idx], pop[1][best_idx])
                best_metrics = {
                    'cppn_nodes': cppn_nodes_list[best_idx],
                    'cppn_connections': cppn_connections_list[best_idx],
                    'discovered_nodes': discovered_nodes_list[best_idx] if discovered_nodes_list else 0,
                    'substrate_nodes': substrate_nodes_list[best_idx] if substrate_nodes_list else len(self.config.input_coors) + len(self.config.output_coors),
                    'total_connections': total_connections_list[best_idx] if total_connections_list else 0
                }

            # ES-HyperNEAT specific metrics
            es_metrics = {
                'generation': generation,
                'discovered_nodes': int(np.mean(discovered_nodes_list)) if discovered_nodes_list else 0,
                'total_connections': int(np.mean(total_connections_list)) if total_connections_list else 0,
                'substrate_nodes': int(np.mean(substrate_nodes_list)) if substrate_nodes_list else len(self.config.input_coors) + len(self.config.output_coors),
                'cppn_nodes': int(np.mean(cppn_nodes_list)) if cppn_nodes_list else 0,
                'cppn_connections': int(np.mean(cppn_connections_list)) if cppn_connections_list else 0
            }
            es_metrics_history.append(es_metrics)

            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.6f}, "
                      f"Mean = {mean_fitness:.6f}, CPPN nodes = {es_metrics['cppn_nodes']}")

            # Garbage collection to prevent memory leak
            import gc
            gc.collect()

            # Check for early termination
            if best_fitness >= 0.975:
                print(f"Target fitness reached at generation {generation}")
                break

        # Extract detailed metrics from best genome
        if best_genome is not None:
            try:
                # Transform best genome to get topology info
                best_transformed = self.algorithm.transform(state, best_genome)

                discovered_nodes = 0  # Default if we can't extract
                total_connections = 0

                substrate_nodes = len(self.config.input_coors) + discovered_nodes + len(self.config.output_coors)

                best_metrics.update({
                    'discovered_nodes': discovered_nodes,
                    'substrate_nodes': substrate_nodes,
                    'total_connections': total_connections
                })

            except Exception as e:
                print(f"Warning: Could not extract detailed metrics: {e}")

        # Compile statistics
        generation_stats = {
            'fitness': {
                'best': best_fitness_history,
                'mean': mean_fitness_history,
                'min': min_fitness_history
            },
            'generations': len(best_fitness_history),
            'total_time': time.time() - self.start_time,
            'es_metrics': es_metrics_history,
            'final_metrics': {
                'fitness': best_fitness_overall,
                'cppn_nodes': best_metrics.get('cppn_nodes', 0) if best_metrics else 0,
                'cppn_connections': best_metrics.get('cppn_connections', 0) if best_metrics else 0,
                'discovered_nodes': best_metrics.get('discovered_nodes', 0) if best_metrics else 0,
                'substrate_nodes': best_metrics.get('substrate_nodes', 0) if best_metrics else 0
            }
        }

        return state, generation_stats
