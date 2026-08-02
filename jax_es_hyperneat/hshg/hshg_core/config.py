"""HSHG Configuration for ES-HyperNEAT spatial indexing.

This module defines the configuration dataclass for Hierarchical Spatial Hash Grids,
replacing the quadtree-based spatial indexing with O(1) average-case spatial queries.
"""

from dataclasses import dataclass
from typing import Tuple
import jax.numpy as jnp


@dataclass(frozen=True)
class HSHGConfig:
    """Static HSHG configuration - determines compiled function shapes.

    Attributes:
        cell_size: Size of each hash cell at the finest level. Smaller = more precision,
                   larger = fewer cells to check. Typical: 0.25 for [-1, 1] coordinate space.
        max_nodes: Maximum number of nodes that can be stored. Must be static for JAX JIT.
        cell_capacity: Maximum nodes per cell. Aligned to 32 for GPU warp efficiency.
        num_cells: Size of hash table. Use prime number for better distribution.
        capacity_multiplier: Safety margin for density fluctuations (JAX MD pattern).
        hierarchy_levels: Cell sizes for multi-level hierarchy (fine to coarse).
    """
    cell_size: float = 0.25
    max_nodes: int = 1000
    cell_capacity: int = 32
    num_cells: int = 1009  # Prime number for good hash distribution
    capacity_multiplier: float = 1.25
    hierarchy_levels: Tuple[float, ...] = (0.125, 0.25, 0.5)

    # Hash function primes (Wang hash optimized for spatial data)
    hash_prime_x: int = 73856093
    hash_prime_y: int = 19349663

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.cell_size <= 0:
            raise ValueError(f"cell_size must be positive, got {self.cell_size}")
        if self.max_nodes <= 0:
            raise ValueError(f"max_nodes must be positive, got {self.max_nodes}")
        if self.cell_capacity <= 0:
            raise ValueError(f"cell_capacity must be positive, got {self.cell_capacity}")
        if self.num_cells <= 0:
            raise ValueError(f"num_cells must be positive, got {self.num_cells}")
        if len(self.hierarchy_levels) == 0:
            raise ValueError("hierarchy_levels cannot be empty")

    @property
    def num_levels(self) -> int:
        """Number of hierarchy levels."""
        return len(self.hierarchy_levels)

    @property
    def hash_primes(self) -> jnp.ndarray:
        """Hash primes as JAX array."""
        return jnp.array([self.hash_prime_x, self.hash_prime_y], dtype=jnp.int32)

    @classmethod
    def for_es_hyperneat(cls, max_depth: int = 3, variance_threshold: float = 0.03) -> 'HSHGConfig':
        """Create configuration optimized for ES-HyperNEAT substrate discovery.

        Args:
            max_depth: Maximum quadtree depth (determines grid resolution).
            variance_threshold: ES-HyperNEAT variance threshold (affects expected node count).

        Returns:
            HSHGConfig tuned for ES-HyperNEAT.
        """
        # Calculate expected positions based on max_depth
        # At each level: 4^level positions. Total = sum(4^i for i in 1..max_depth)
        max_positions = sum(4 ** i for i in range(1, max_depth + 1))

        # Add margin for multiple input coordinates and iteration levels
        # Typical: 3 inputs * 2 outputs * 2 iterations = 12x multiplier
        max_nodes = max_positions * 20  # Conservative estimate

        # Cell size based on finest resolution
        # At max_depth, grid step = 2.0 / (2^max_depth)
        finest_step = 2.0 / (2 ** max_depth)
        cell_size = finest_step * 1.5  # Slightly larger than grid step

        # Hierarchy levels from fine to coarse
        hierarchy_levels = tuple(cell_size * (2 ** i) for i in range(3))

        return cls(
            cell_size=cell_size,
            max_nodes=max_nodes,
            cell_capacity=32,
            num_cells=2017,  # Prime number larger than expected cells
            capacity_multiplier=1.25,
            hierarchy_levels=hierarchy_levels,
        )
