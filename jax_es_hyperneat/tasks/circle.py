"""2D circle classification (inside radius 0.5). Threshold 0.975.

100 fixed points drawn with a seeded RNG so the dataset is identical across runs.
"""
import numpy as np

from .base import Task


class CircleProblem:
    """Label = 1 if x^2 + y^2 < 0.25, over 100 fixed points in [-1, 1]^2."""

    use_bias = True
    input_size = 3   # x, y, bias
    output_size = 1

    def __init__(self):
        rng = np.random.RandomState(42)
        self._points = rng.uniform(-1, 1, (100, 2)).astype(np.float32)
        self._labels = (self._points[:, 0] ** 2 + self._points[:, 1] ** 2 < 0.25).astype(np.float32)

    def get_data(self):
        data = []
        for i in range(len(self._points)):
            inp = np.array([self._points[i, 0], self._points[i, 1], 1.0], dtype=np.float32)
            target = np.array([self._labels[i]], dtype=np.float32)
            data.append((inp, target))
        return data


TASK = Task(
    name="circle",
    input_coords=[(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)],
    output_coords=[(0.0, 1.0)],
    fitness_threshold=0.975,
    make_problem=CircleProblem,
    default_depths=[2, 3, 4],
)
