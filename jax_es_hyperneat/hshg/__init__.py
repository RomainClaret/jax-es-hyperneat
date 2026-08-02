"""HSHG ES-HyperNEAT subsystem.

The HSHG ablation: ES-HyperNEAT with the sequential quadtree replaced by a
Hierarchical Spatial Hash Grid for substrate discovery. The paper's finding is that
HSHG over-discovers nodes and solves 0% of the tasks; this code reproduces that
faithful failure.

Layout:
    eshyperneat.py        ES-HyperNEAT base class (quadtree), depends only on tensorneat
    substrate.py          Substrate / QuadPoint / Connection, depends only on tensorneat
    eshyperneat_hshg.py   ESHyperNEATHSHG subclass; uses the local ``hshg_core`` package
    hshg_core/            Self-contained HSHG primitives (JAX + NumPy only)
    experiments/          Per-task experiment classes (HSHGESHyperNEATExperiment)

``run_hshg.run_single`` is the driver entry point.
"""
from .run_hshg import run_single

__all__ = ["run_single"]
