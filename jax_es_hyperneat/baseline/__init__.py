"""PUREPLES ES-HyperNEAT baseline harness.

The CPU quadtree baseline the paper compares against: neat-python evolves a CPPN,
PUREPLES' ``ESNetwork`` discovers the substrate via the sequential quadtree, and the
phenotype is evaluated on the task data. See ``pureples_harness.run_single``.
"""
from .pureples_harness import run_single

__all__ = ["run_single"]
