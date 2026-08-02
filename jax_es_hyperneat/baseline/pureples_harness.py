"""PUREPLES ES-HyperNEAT baseline runner (CPU quadtree).

Runs the per-task PUREPLES ES-HyperNEAT experiments that produced the paper's
``pureples_*`` result JSONs.

Pipeline per run:
    1. neat-python evolves a CPPN population (config in ``configs/config_cppn_<task>``).
    2. For each genome, PUREPLES' ``ESNetwork`` discovers the substrate from the CPPN
       via the sequential quadtree, then builds a recurrent phenotype network.
    3. Fitness is ``1 - MSE`` over the task data (clamped to 0 for the regression /
       classification tasks), or the normalized CartPole return over 5 episodes.

The substrate input/output coordinates and the ES-HyperNEAT parameters
(``max_weight=5.0``, ``band_threshold=0.3``, etc.) are the exact values the PUREPLES
baseline used, which differ from the JAX-ESHN side (``max_weight=8.0``). They are kept
verbatim here for faithful reproduction.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import neat
import neat.nn
from pureples.es_hyperneat.es_hyperneat import ESNetwork
from pureples.shared.substrate import Substrate

CONFIG_DIR = Path(__file__).resolve().parent / "configs"

# ES-HyperNEAT parameters as used by the PUREPLES baseline (NOT the JAX-ESHN side).
# initial_depth is always 0; max_depth is the swept depth. max_weight=5.0 and
# band_threshold=0.3 are the PUREPLES baseline values.
PUREPLES_ES_PARAMS = dict(
    initial_depth=0,
    variance_threshold=0.03,
    band_threshold=0.3,
    iteration_level=1,
    division_threshold=0.5,
    max_weight=5.0,
    activation="sigmoid",
)

# CartPole observation normalization (CartPole-v1 state bounds) and episode budget.
CART_POS_SCALE = 2.4
CART_VEL_SCALE = 3.0
POLE_ANGLE_SCALE = 0.2095
POLE_VEL_SCALE = 3.0
CARTPOLE_EPISODES = 5
CARTPOLE_MAX_STEPS = 500


@dataclass
class _GenStats:
    """Per-generation fitness record collected by the reporter."""

    best: List[float] = field(default_factory=list)
    mean: List[float] = field(default_factory=list)
    minimum: List[float] = field(default_factory=list)


class _Reporter(neat.reporting.BaseReporter):
    """Records per-generation best/mean/min fitness and the best topology metrics."""

    def __init__(self):
        self.stats = _GenStats()
        self.best_fitness = None
        self.best_metrics: Dict[str, Any] = {}
        self.start_time = time.time()

    def start_generation(self, generation):
        self.best_fitness = None  # reset per-generation best tracker (see eval loop)

    def post_evaluate(self, config, population, species_set, best_genome):
        fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
        if not fitnesses:
            return
        self.stats.best.append(max(fitnesses))
        self.stats.mean.append(sum(fitnesses) / len(fitnesses))
        self.stats.minimum.append(min(fitnesses))


# --------------------------------------------------------------------------- #
# PUREPLES-faithful task data
# --------------------------------------------------------------------------- #
#
# The data here mirrors the per-task PUREPLES experiments EXACTLY, because that
# is what produced the frozen pureples_* JSONs. It deliberately does NOT read
# ``task.make_problem().get_data()``: that data source is calibrated for the JAX-ESHN
# side, where the sine target is rescaled to [0, 1]. The PUREPLES baseline instead
# regresses the RAW ``sin(pi*x)`` in [-1, 1] against a sigmoid output (so it plateaus
# near 0.735 -- the published result). XOR / parity / circle data are identical on both
# sides; only sine differs. Each (input, target) already includes the bias column, so the
# phenotype is activated directly on the input vector.

def _pureples_data(task) -> Tuple[List[Tuple[float, ...]], List[float]]:
    """Return (inputs_with_bias, targets) for ``task``, faithful to the PUREPLES baseline."""
    name = task.name
    if name == "xor":
        rows = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
        targets = [0.0, 1.0, 1.0, 0.0]
        inputs = [r + (1.0,) for r in rows]
        return inputs, targets
    if name == "parity3":
        rows = [(a, b, c) for a in (0.0, 1.0) for b in (0.0, 1.0) for c in (0.0, 1.0)]
        targets = [float(int(a + b + c) % 2) for (a, b, c) in rows]
        inputs = [r + (1.0,) for r in rows]
        return inputs, targets
    if name == "circle":
        rng = np.random.RandomState(42)
        points = rng.uniform(-1.0, 1.0, size=(100, 2))
        labels = ((points[:, 0] ** 2 + points[:, 1] ** 2) <= 0.5 ** 2).astype(float)
        inputs = [(float(x), float(y), 1.0) for (x, y) in points]
        return inputs, [float(l) for l in labels]
    if name == "sine":
        x_vals = np.linspace(-1.0, 1.0, 20)
        y_vals = np.sin(np.pi * x_vals)  # RAW sin(pi*x) in [-1, 1] (NOT rescaled)
        inputs = [(float(x), 1.0) for x in x_vals]
        return inputs, [float(y) for y in y_vals]
    raise ValueError(f"no PUREPLES static data for task {name!r}")


# --------------------------------------------------------------------------- #
# Fitness evaluation
# --------------------------------------------------------------------------- #

def _make_eval_fitness(task, es_params, reporter):
    """Build the neat-python ``eval_fitness(genomes, config)`` callback for a task.

    Static tasks: fitness is ``1 - MSE`` over the PUREPLES-faithful data. The regression /
    classification tasks (circle, sine) clamp at 0; XOR and parity do not (matching the
    PUREPLES baseline experiments exactly). The bias column is already part of the inputs,
    so the phenotype is activated on the raw input vector.
    """
    substrate = Substrate(list(task.input_coords), list(task.output_coords))
    clamp = task.name in ("circle", "sine")
    inputs_list, targets = _pureples_data(task)
    data = list(zip(inputs_list, targets))
    n_samples = len(data)

    def eval_fitness(genomes, config):
        # Track the best genome's discovered-topology metrics across this generation.
        gen_best = -float("inf")
        for _genome_id, genome in genomes:
            cppn = neat.nn.FeedForwardNetwork.create(genome, config)
            es_network = ESNetwork(substrate, cppn, es_params)
            network = es_network.create_phenotype_network()

            sse = 0.0
            output = [0.0]
            for inputs, target in data:
                network.reset()
                for _ in range(es_network.activations):
                    output = network.activate(inputs)
                sse += (output[0] - target) ** 2.0

            mse = sse / n_samples
            fitness = max(0.0, 1.0 - mse) if clamp else 1.0 - mse
            genome.fitness = fitness

            if fitness > gen_best:
                gen_best = fitness
                reporter.best_metrics = _topology_metrics(network, cppn, genome, task)

        if gen_best > (reporter.best_fitness or -float("inf")):
            reporter.best_fitness = gen_best

    return eval_fitness


def _topology_metrics(network, cppn, genome, task) -> Dict[str, Any]:
    """Discovered/substrate/CPPN counts for the best genome, as PUREPLES reports them."""
    n_io = len(task.input_coords) + len(task.output_coords)
    return {
        "discovered_nodes": max(0, len(network.node_evals) - n_io),
        "substrate_nodes": len(network.node_evals),
        "cppn_nodes": len(cppn.node_evals),
        "cppn_connections": len(genome.connections),
    }


# --------------------------------------------------------------------------- #
# CartPole (gym episode loop)
# --------------------------------------------------------------------------- #

def _evaluate_cartpole(network, es_network, num_episodes=CARTPOLE_EPISODES):
    """Mean(steps)/500 over ``num_episodes`` of CartPole-v1 (matches PUREPLES baseline)."""
    import gymnasium as gym

    total = 0.0
    for _ in range(num_episodes):
        env = gym.make("CartPole-v1")
        obs, _ = env.reset()
        steps = 0
        done = False
        output = [0.0]
        while not done:
            inputs = (
                obs[0] / CART_POS_SCALE,
                obs[1] / CART_VEL_SCALE,
                obs[2] / POLE_ANGLE_SCALE,
                obs[3] / POLE_VEL_SCALE,
                1.0,  # bias
            )
            network.reset()
            for _ in range(es_network.activations):
                output = network.activate(inputs)
            action = 1 if output[0] > 0.5 else 0
            obs, _reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        total += steps / CARTPOLE_MAX_STEPS
        env.close()
    return total / num_episodes


def _make_eval_cartpole(task, es_params, reporter):
    """Build the neat-python eval callback for the CartPole gym task."""
    substrate = Substrate(list(task.input_coords), list(task.output_coords))

    def eval_fitness(genomes, config):
        gen_best = -float("inf")
        for _genome_id, genome in genomes:
            cppn = neat.nn.FeedForwardNetwork.create(genome, config)
            es_network = ESNetwork(substrate, cppn, es_params)
            network = es_network.create_phenotype_network()
            fitness = _evaluate_cartpole(network, es_network)
            genome.fitness = fitness
            if fitness > gen_best:
                gen_best = fitness
                reporter.best_metrics = _topology_metrics(network, cppn, genome, task)
        if gen_best > (reporter.best_fitness or -float("inf")):
            reporter.best_fitness = gen_best

    return eval_fitness


# --------------------------------------------------------------------------- #
# Driver entry point
# --------------------------------------------------------------------------- #

def run_single(task, depth: int, seed: int, config: Dict[str, Any],
               verbose: bool = True) -> Dict[str, Any]:
    """Run one (depth, seed) PUREPLES baseline experiment for ``task``.

    Returns a run dict matching the frozen ``pureples_*`` result schema. The static
    tasks (xor, parity3, circle, sine) use the ``solved_gen`` / ``total_time_s`` /
    ``final_*_nodes`` schema; CartPole uses the ``solve_generation`` / ``wall_time_s`` /
    ``final_metrics`` / ``final_mean_fitness`` / ``timestamp`` schema, exactly as the
    benchmark drivers wrote them.
    """
    threshold = config["fitness_threshold"]
    max_generations = config["max_generations"]
    is_gym = task.is_gym

    es_params = dict(PUREPLES_ES_PARAMS, max_depth=depth)
    config_path = str(CONFIG_DIR / f"config_cppn_{task.name}")

    # neat-python seeds itself from the numpy global RNG state.
    np.random.seed(seed)

    neat_config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path,
    )

    population = neat.Population(neat_config)
    if verbose:
        population.add_reporter(neat.StdOutReporter(False))
    reporter = _Reporter()
    population.add_reporter(reporter)

    if is_gym:
        eval_fn = _make_eval_cartpole(task, es_params, reporter)
    else:
        eval_fn = _make_eval_fitness(task, es_params, reporter)

    start = time.time()
    population.run(eval_fn, max_generations)
    elapsed = time.time() - start

    best_list = reporter.stats.best
    mean_list = reporter.stats.mean
    best_fitness = float(max(best_list)) if best_list else 0.0
    solved = any(f >= threshold for f in best_list)
    solve_gen = next((i for i, f in enumerate(best_list) if f >= threshold), None)
    metrics = reporter.best_metrics or {
        "discovered_nodes": 0,
        "substrate_nodes": len(task.input_coords) + len(task.output_coords),
        "cppn_nodes": 0,
        "cppn_connections": 0,
    }
    generations_run = len(best_list)

    if is_gym:
        # CartPole schema (benchmark_cartpole.run_pureples + main()'s timestamp).
        return {
            "implementation": "pureples",
            "depth": depth,
            "seed": seed,
            "generations_run": generations_run,
            "wall_time_s": elapsed,
            "best_fitness": best_fitness,
            "final_mean_fitness": float(mean_list[-1]) if mean_list else 0.0,
            "solved": solved,
            "solve_generation": solve_gen,
            "final_metrics": {
                "discovered_nodes": metrics["discovered_nodes"],
                "substrate_nodes": metrics["substrate_nodes"],
                "cppn_nodes": metrics["cppn_nodes"],
                "cppn_connections": metrics["cppn_connections"],
                "fitness": best_fitness,
            },
            "fitness_history_best": [float(f) for f in best_list],
            "fitness_history_mean": [float(f) for f in mean_list],
            "timestamp": datetime.now().isoformat(),
        }

    # Static-task schema (benchmark_sine.run_pureples_sine).
    return {
        "implementation": "pureples",
        "depth": depth,
        "seed": seed,
        "generations_run": generations_run,
        "best_fitness": best_fitness,
        "solved": solved,
        "solved_gen": solve_gen,
        "total_time_s": elapsed,
        "final_discovered_nodes": metrics["discovered_nodes"],
        "final_substrate_nodes": metrics["substrate_nodes"],
        "final_cppn_nodes": metrics["cppn_nodes"],
        "final_cppn_connections": metrics["cppn_connections"],
        "fitness_history_best": [float(f) for f in best_list],
        "fitness_history_mean": [float(f) for f in mean_list],
    }
