"""Sine regression. Threshold 0.95.

20 evenly spaced points in [-1, 1]; target = (sin(pi*x) + 1) / 2, scaled to [0, 1]
for the sigmoid output.
"""
import numpy as np

from .base import Task


class SineProblem:
    """Sine regression: target = (sin(pi*x) + 1) / 2 over 20 points in [-1, 1]."""

    use_bias = True
    input_size = 2   # x, bias
    output_size = 1

    def __init__(self):
        self._x = np.linspace(-1, 1, 20, dtype=np.float32)
        self._targets = ((np.sin(np.pi * self._x) + 1.0) / 2.0).astype(np.float32)

    def get_data(self):
        data = []
        for i in range(len(self._x)):
            inp = np.array([self._x[i], 1.0], dtype=np.float32)
            target = np.array([self._targets[i]], dtype=np.float32)
            data.append((inp, target))
        return data


TASK = Task(
    name="sine",
    input_coords=[(-1.0, -1.0), (1.0, -1.0)],  # x + bias
    output_coords=[(0.0, 1.0)],
    fitness_threshold=0.95,
    make_problem=SineProblem,
    default_depths=[2, 3, 4],
)
