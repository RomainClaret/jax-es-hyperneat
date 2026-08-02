"""Standalone benchmark task registry.

Each task module exposes a ``TASK`` (see ``base.Task``). CartPole additionally
exposes ``run_gym_generation`` for its episode-based fitness loop.
"""
from .base import Task, DEPTH_POSITIONS, ES_PARAMS
from . import xor, parity3, circle, sine, cartpole

TASKS = {
    "xor": xor.TASK,
    "parity3": parity3.TASK,
    "circle": circle.TASK,
    "sine": sine.TASK,
    "cartpole": cartpole.TASK,
}

__all__ = ["TASKS", "Task", "DEPTH_POSITIONS", "ES_PARAMS"]
