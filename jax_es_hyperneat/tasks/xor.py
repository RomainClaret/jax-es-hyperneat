"""XOR (2-input exclusive-or, with bias). Threshold 0.98.

The headline depth x population scaling study (paper Section 6) runs on XOR.
"""
import numpy as np

from .base import Task


class XORProblem:
    """XOR truth table with an explicit bias input column."""

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
    input_coords=[(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)],  # 3 inputs at y=-1
    output_coords=[(0.0, 1.0)],
    fitness_threshold=0.98,
    make_problem=XORProblem,
    default_depths=[1, 2, 3, 4, 5, 6, 7],
)
