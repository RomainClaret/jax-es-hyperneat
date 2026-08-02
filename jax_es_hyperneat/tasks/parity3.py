"""3-bit parity (odd number of 1s among 3 bits, with bias). Threshold 0.975."""
import numpy as np

from .base import Task


class Parity3Problem:
    """3-bit parity: output = 1 if an odd number of the 3 input bits are 1."""

    use_bias = True
    input_size = 4   # 3 bits + bias
    output_size = 1

    def get_data(self):
        return [
            (np.array([0, 0, 0, 1], dtype=np.float32), np.array([0], dtype=np.float32)),
            (np.array([0, 0, 1, 1], dtype=np.float32), np.array([1], dtype=np.float32)),
            (np.array([0, 1, 0, 1], dtype=np.float32), np.array([1], dtype=np.float32)),
            (np.array([0, 1, 1, 1], dtype=np.float32), np.array([0], dtype=np.float32)),
            (np.array([1, 0, 0, 1], dtype=np.float32), np.array([1], dtype=np.float32)),
            (np.array([1, 0, 1, 1], dtype=np.float32), np.array([0], dtype=np.float32)),
            (np.array([1, 1, 0, 1], dtype=np.float32), np.array([0], dtype=np.float32)),
            (np.array([1, 1, 1, 1], dtype=np.float32), np.array([1], dtype=np.float32)),
        ]


TASK = Task(
    name="parity3",
    input_coords=[(-1.5, -1.0), (-0.5, -1.0), (0.5, -1.0), (1.5, -1.0)],
    output_coords=[(0.0, 1.0)],
    fitness_threshold=0.975,
    make_problem=Parity3Problem,
    default_depths=[2, 3, 4],
)
