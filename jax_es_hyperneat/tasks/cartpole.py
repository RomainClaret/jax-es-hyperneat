"""CartPole-v1 reinforcement-learning control. Threshold 0.95.

CartPole fitness is the normalized return (mean steps / 500) over 5 episodes, so it is
evaluated through a gym episode loop rather than the algorithm's static-data
``run_generation``. The generation loop here is the manual ask -> discover -> build ->
evaluate -> tell pattern the paper's runner used.
"""
import numpy as np
import jax.numpy as jnp

from .base import Task

# Observation normalization scales (CartPole-v1 state bounds).
CART_POS_SCALE = 2.4
CART_VEL_SCALE = 3.0
POLE_ANGLE_SCALE = 0.2095
POLE_VEL_SCALE = 3.0
NUM_EPISODES = 5
MAX_STEPS = 500


class CartPoleProblem:
    """Minimal problem interface for Pipeline initialization (fitness uses gym)."""

    use_bias = True
    input_size = 5   # 4 observations + bias
    output_size = 1

    def get_data(self):
        # Dummy data: CartPole uses gym evaluation, not static input/target pairs.
        return [(np.zeros(5, dtype=np.float32), np.zeros(1, dtype=np.float32))]


def evaluate_on_cartpole(algo, substrate_net, num_episodes=NUM_EPISODES):
    """Return fitness in [0, 1] = mean(steps) / MAX_STEPS over ``num_episodes``."""
    import gymnasium as gym

    if substrate_net is None:
        return 0.0

    nodes, conns = substrate_net
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
            output = algo._forward_hyperneat_style(nodes, conns, inputs)
            action = 1 if float(output[0]) > 0.5 else 0
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        total_fitness += steps / MAX_STEPS
        env.close()
    return total_fitness / num_episodes


def run_gym_generation(algo, state, population_size):
    """One generation: ask -> per-CPPN discover/build/evaluate -> tell.

    Returns (new_state, best_fitness, mean_fitness).
    """
    cppn_pop = algo._compiled_ask(state)
    cppns_transformed = algo._compiled_transform_batch(state, cppn_pop)

    fitnesses = []
    for i in range(population_size):
        try:
            cppn = tuple(cppns_transformed[j][i] for j in range(4))
            hidden_nodes, connections, _ = algo._discover_substrate_es(state, cppn)
            substrate_net = algo._build_tensorneat_substrate(hidden_nodes, connections, state, cppn)
            fitnesses.append(evaluate_on_cartpole(algo, substrate_net))
        except Exception:
            fitnesses.append(0.0)

    new_state = algo._compiled_tell(state, jnp.array(fitnesses))
    return new_state, float(max(fitnesses)), float(np.mean(fitnesses))


TASK = Task(
    name="cartpole",
    input_coords=[(-2.0, -1.0), (-1.0, -1.0), (0.0, -1.0), (1.0, -1.0), (2.0, -1.0)],
    output_coords=[(0.0, 1.0)],
    fitness_threshold=0.95,
    make_problem=CartPoleProblem,
    default_depths=[2, 3, 4],
    is_gym=True,
)
