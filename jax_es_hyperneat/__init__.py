"""JAX-ESHN: a JAX/TensorNEAT implementation of ES-HyperNEAT.

ES-HyperNEAT evolves substrate topology through adaptive quadtree subdivision: each
CPPN discovers its own variable-cardinality set of substrate positions. This package
provides a JAX/GPU implementation built on TensorNEAT, used to diagnose why that
adaptive quadtree is structurally incompatible with population-level ``vmap``/XLA
batching (each genome produces a different tensor shape). It is the standing diagnostic
companion to EMR-HyperNEAT, which resolves the bottleneck.

Public API
----------
    from jax_es_hyperneat import JAXESHyperNEAT

The implementation is the lazy-quadtree ES-HyperNEAT with the batched optimizations
(O1-O8) described in the paper. See ``jax_es_hyperneat.driver`` for the unified
benchmark CLI and ``jax_es_hyperneat.tasks`` for the standalone task definitions.
"""
from .eshyperneat import TensorNEATESHyperNEATOptimized
from .eshyperneat import TensorNEATESHyperNEATOptimized as JAXESHyperNEAT

__all__ = ["JAXESHyperNEAT", "TensorNEATESHyperNEATOptimized"]
