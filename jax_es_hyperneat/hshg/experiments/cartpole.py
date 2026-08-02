"""
HSHG ES-HyperNEAT CartPole experiment.
Uses HSHG-based ES-HyperNEAT reimplementation with TensorNEAT for CartPole-v1 control task.

Network: 4 inputs (cart pos, cart vel, pole angle, pole angular vel) + 1 bias -> 1 output (action).
Fitness: average steps / 500 over 5 episodes. Solve threshold: >= 0.95 (475+ steps).

Identical to tensorneat_cartpole except substrate discovery uses HSHG instead of quadtree.
"""

import os
import sys
import time
import gc
import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

import gymnasium as gym

# Force CPU for fair comparison; ES-HyperNEAT discovery is CPU-based
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
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


# Observation normalization ranges for CartPole-v1
CART_POS_SCALE = 2.4
CART_VEL_SCALE = 3.0
POLE_ANGLE_SCALE = 0.2095
POLE_VEL_SCALE = 3.0

NUM_EPISODES = 5
MAX_STEPS = 500  # CartPole-v1 max


@dataclass
class CartPoleConfig:
    """Configuration for CartPole ES-HyperNEAT experiment matching PUREPLES."""
    # NEAT parameters for evolving CPPN
    pop_size: int = 150
    species_size: int = 15
    survival_threshold: float = 0.2
    max_nodes: int = 20

    # ES-HyperNEAT specific parameters (matching PUREPLES)
    band_threshold: float = 0.2
    variance_threshold: float = 0.03
    division_threshold: float = 0.5
    initial_depth: int = 0  # Small version
    max_depth: int = 1  # Small version
    iteration_level: int = 1
    max_weight: float = 5.0
    activation: str = 'sigmoid'

    # Training parameters
    max_generations: int = 300
    fitness_target: float = 0.95  # Average 475+ steps
    seed: int = 42

    # Substrate coordinates: 4 inputs + 1 bias at y=-1, 1 output at y=1
    input_coors: Tuple[Tuple[float, float], ...] = (
        (-2.0, -1.0),  # cart position
        (-1.0, -1.0),  # cart velocity
        (0.0, -1.0),   # pole angle
        (1.0, -1.0),   # pole angular velocity
        (2.0, -1.0),   # bias
    )
    output_coors: Tuple[Tuple[float, float], ...] = ((0.0, 1.0),)

    # CPPN activation functions matching PUREPLES
    cppn_activations: Tuple[str, ...] = ('tanh', 'sin', 'gauss')


def evaluate_substrate_on_cartpole(forward_fn, state, params, num_episodes=NUM_EPISODES):
    """Evaluate a TensorNEAT substrate network on CartPole-v1.

    The substrate forward function maps 5 inputs -> 1 output.
    We run gymnasium episodes and return average fitness.

    Args:
        forward_fn: The algorithm's forward function (state, params, inputs) -> outputs.
        state: Algorithm state.
        params: Transformed individual parameters (substrate weights).
        num_episodes: Number of episodes to average.

    Returns:
        Fitness in [0, 1].
    """
    total_fitness = 0.0
    for _ in range(num_episodes):
        env = gym.make('CartPole-v1')
        obs, _ = env.reset()
        steps = 0
        done = False
        while not done:
            inputs = jnp.array([
                obs[0] / CART_POS_SCALE,
                obs[1] / CART_VEL_SCALE,
                obs[2] / POLE_ANGLE_SCALE,
                obs[3] / POLE_VEL_SCALE,
                1.0,  # bias
            ])
            output = forward_fn(state, params, inputs)
            action = 1 if float(output[0]) > 0.5 else 0
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        total_fitness += steps / MAX_STEPS
        env.close()
    return total_fitness / num_episodes


class HSHGESHyperNEATExperiment:
    """HSHG ES-HyperNEAT experiment for CartPole-v1."""

    def __init__(self, config: Optional[CartPoleConfig] = None,
                 save_checkpoints: bool = False,
                 checkpoint_dir: Optional[str] = None,
                 version: str = 'S'):
        """
        Initialize the experiment.

        Args:
            config: Experiment configuration.
            save_checkpoints: Whether to save checkpoints.
            checkpoint_dir: Directory for checkpoints.
            version: 'S' (small), 'M' (medium), or 'L' (large); affects ES-HyperNEAT depth.
        """
        self.config = config or CartPoleConfig()
        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = checkpoint_dir
        self.version = version

        # Adjust depth based on version
        if version == 'M':
            self.config.initial_depth = 1
            self.config.max_depth = 2
        elif version == 'L':
            self.config.initial_depth = 2
            self.config.max_depth = 3

        if save_checkpoints and checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Tracking
        self.generation_history = []
        self.best_genomes = []
        self.es_metrics_history = []
        self.start_time = None
        self.algorithm = None
        self._setup_experiment()

    def _setup_experiment(self):
        """Set up HSHG ES-HyperNEAT algorithm."""
        substrate = ESHyperNEATSubstrate(
            input_coordinates=list(self.config.input_coors),
            hidden_coordinates=[],  # ES-HyperNEAT discovers these
            output_coordinates=list(self.config.output_coors),
            include_bias=False,  # Bias already in input coordinates
            band_threshold=self.config.band_threshold,
            variance_threshold=self.config.variance_threshold,
        )

        cppn_genome = genome.DefaultGenome(
            num_inputs=5,  # x1, y1, x2, y2, bias
            num_outputs=1,  # connection weight
            max_nodes=self.config.max_nodes,
            output_transform=ACT.tanh,
        )

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

    def run(self):
        """Run the HSHG ES-HyperNEAT CartPole experiment.

        Returns:
            Tuple of (final_state, statistics_dict).
        """
        self.start_time = time.time()

        state = State(randkey=jax.random.PRNGKey(self.config.seed))
        state = self.algorithm.setup(state)

        best_fitness_history = []
        mean_fitness_history = []
        min_fitness_history = []
        es_metrics_history = []

        print(f"\nRunning HSHG ES-HyperNEAT CartPole...")
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
            pop = self.algorithm.ask(state)

            discovered_nodes_list = []
            substrate_nodes_list = []
            total_connections_list = []
            cppn_nodes_list = []
            cppn_connections_list = []

            fitnesses = []
            for i in range(len(pop[0])):
                individual = (pop[0][i], pop[1][i])

                try:
                    transformed = self.algorithm.transform(state, individual)

                    # Evaluate on CartPole episodes instead of FuncFit
                    fitness = evaluate_substrate_on_cartpole(
                        self.algorithm.forward, state, transformed,
                    )
                    fitnesses.append(fitness)

                    cppn_nodes = int(jnp.sum(~jnp.isnan(individual[0][:, 0])))
                    cppn_connections = int(jnp.sum(~jnp.isnan(individual[1][:, 0])))
                    cppn_nodes_list.append(cppn_nodes)
                    cppn_connections_list.append(cppn_connections)

                    if hasattr(self.algorithm, 'last_discovery_metrics'):
                        discovered_nodes_list.append(self.algorithm.last_discovery_metrics['discovered_nodes'])
                        substrate_nodes_list.append(self.algorithm.last_discovery_metrics['substrate_nodes'])
                        total_connections_list.append(self.algorithm.last_discovery_metrics['total_connections'])
                    else:
                        discovered_nodes_list.append(0)
                        substrate_nodes_list.append(
                            len(self.config.input_coors) + len(self.config.output_coors)
                        )
                        total_connections_list.append(0)

                except Exception as e:
                    print(f"Warning: Transform failed for individual {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    fitnesses.append(0.0)
                    cppn_nodes_list.append(0)
                    cppn_connections_list.append(0)

            fitnesses = jnp.array(fitnesses)
            state = self.algorithm.tell(state, fitnesses)

            # Track statistics; fitness is already in [0, 1]
            best_idx = int(jnp.argmax(fitnesses))
            best_fitness = float(fitnesses[best_idx])
            mean_fitness = float(jnp.mean(fitnesses))
            min_fitness = float(jnp.min(fitnesses))

            best_fitness_history.append(best_fitness)
            mean_fitness_history.append(mean_fitness)
            min_fitness_history.append(min_fitness)

            if best_fitness > best_fitness_overall:
                best_fitness_overall = best_fitness
                best_genome = (pop[0][best_idx], pop[1][best_idx])
                best_metrics = {
                    'cppn_nodes': cppn_nodes_list[best_idx],
                    'cppn_connections': cppn_connections_list[best_idx],
                    'discovered_nodes': discovered_nodes_list[best_idx] if discovered_nodes_list else 0,
                    'substrate_nodes': substrate_nodes_list[best_idx] if substrate_nodes_list else len(self.config.input_coors) + len(self.config.output_coors),
                    'total_connections': total_connections_list[best_idx] if total_connections_list else 0,
                }

            es_metrics = {
                'generation': generation,
                'discovered_nodes': int(np.mean(discovered_nodes_list)) if discovered_nodes_list else 0,
                'total_connections': int(np.mean(total_connections_list)) if total_connections_list else 0,
                'substrate_nodes': int(np.mean(substrate_nodes_list)) if substrate_nodes_list else len(self.config.input_coors) + len(self.config.output_coors),
                'cppn_nodes': int(np.mean(cppn_nodes_list)) if cppn_nodes_list else 0,
                'cppn_connections': int(np.mean(cppn_connections_list)) if cppn_connections_list else 0,
            }
            es_metrics_history.append(es_metrics)

            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.6f}, "
                      f"Mean = {mean_fitness:.6f}, CPPN nodes = {es_metrics['cppn_nodes']}")

            # Prevent memory leak
            gc.collect()

            if best_fitness >= self.config.fitness_target:
                print(f"Target fitness reached at generation {generation}")
                break

        # Final metrics
        generation_stats = {
            'fitness': {
                'best': best_fitness_history,
                'mean': mean_fitness_history,
                'min': min_fitness_history,
            },
            'generations': len(best_fitness_history),
            'total_time': time.time() - self.start_time,
            'es_metrics': es_metrics_history,
            'final_metrics': {
                'fitness': best_fitness_overall,
                'cppn_nodes': best_metrics.get('cppn_nodes', 0) if best_metrics else 0,
                'cppn_connections': best_metrics.get('cppn_connections', 0) if best_metrics else 0,
                'discovered_nodes': best_metrics.get('discovered_nodes', 0) if best_metrics else 0,
                'substrate_nodes': best_metrics.get('substrate_nodes', 0) if best_metrics else 0,
            },
        }

        return state, generation_stats
