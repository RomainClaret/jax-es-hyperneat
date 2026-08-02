"""HSHG State for ES-HyperNEAT spatial indexing.

This module defines the state dataclass for Hierarchical Spatial Hash Grids.
All arrays have fixed shapes for JAX JIT compatibility.
"""

from typing import NamedTuple
import jax.numpy as jnp


class HSHGState(NamedTuple):
    """JAX-compatible HSHG state with fixed-shape arrays.

    All arrays have static shapes for JIT compilation. Variable-size data
    is handled using validity masks and count fields.

    Attributes:
        positions: Node positions, shape [max_nodes, 2]. (x, y) in [-1, 1] space.
        weights: CPPN weights at each position, shape [max_nodes].
        levels: Quadtree level at which each node was discovered, shape [max_nodes].
        valid_mask: Boolean mask for valid nodes, shape [max_nodes].
        num_nodes: Current number of valid nodes, scalar array shape [].

        cells: Node indices per cell, shape [num_levels, num_cells, cell_capacity].
               Contains -1 for empty slots.
        cell_counts: Number of valid nodes per cell, shape [num_levels, num_cells].

        did_buffer_overflow: Flag indicating capacity exceeded, scalar array shape [].
                            Following JAX MD pattern for overflow detection.
    """
    # Node storage
    positions: jnp.ndarray      # [max_nodes, 2]
    weights: jnp.ndarray        # [max_nodes]
    levels: jnp.ndarray         # [max_nodes] - quadtree level (1, 2, 3, ...)
    valid_mask: jnp.ndarray     # [max_nodes]
    num_nodes: jnp.ndarray      # [] scalar

    # Cell storage (per hierarchy level)
    cells: jnp.ndarray          # [num_levels, num_cells, cell_capacity]
    cell_counts: jnp.ndarray    # [num_levels, num_cells]

    # Overflow detection (JAX MD pattern)
    did_buffer_overflow: jnp.ndarray  # [] scalar boolean


class QueryResult(NamedTuple):
    """Fixed-size query result with padding.

    Attributes:
        neighbor_indices: Indices of neighbors, shape [max_neighbors].
                         Padded with -1 for invalid entries.
        neighbor_distances: Euclidean distances to neighbors, shape [max_neighbors].
                           Padded with inf for invalid entries.
        neighbor_count: Number of valid neighbors found, scalar array shape [].
    """
    neighbor_indices: jnp.ndarray    # [max_neighbors]
    neighbor_distances: jnp.ndarray  # [max_neighbors]
    neighbor_count: jnp.ndarray      # [] scalar


def create_empty_state(
    max_nodes: int,
    num_levels: int,
    num_cells: int,
    cell_capacity: int
) -> HSHGState:
    """Create an empty HSHG state with pre-allocated arrays.

    Args:
        max_nodes: Maximum number of nodes.
        num_levels: Number of hierarchy levels.
        num_cells: Number of cells in hash table.
        cell_capacity: Maximum nodes per cell.

    Returns:
        Empty HSHGState ready for insertions.
    """
    return HSHGState(
        # Node storage - initialized to zeros/False
        positions=jnp.zeros((max_nodes, 2), dtype=jnp.float32),
        weights=jnp.zeros(max_nodes, dtype=jnp.float32),
        levels=jnp.zeros(max_nodes, dtype=jnp.int32),
        valid_mask=jnp.zeros(max_nodes, dtype=bool),
        num_nodes=jnp.array(0, dtype=jnp.int32),

        # Cell storage - initialized to -1 (invalid index)
        cells=jnp.full((num_levels, num_cells, cell_capacity), -1, dtype=jnp.int32),
        cell_counts=jnp.zeros((num_levels, num_cells), dtype=jnp.int32),

        # Overflow flag
        did_buffer_overflow=jnp.array(False, dtype=bool),
    )


def create_empty_query_result(max_neighbors: int) -> QueryResult:
    """Create an empty query result with padding.

    Args:
        max_neighbors: Maximum number of neighbors to return.

    Returns:
        Empty QueryResult with padding values.
    """
    return QueryResult(
        neighbor_indices=jnp.full(max_neighbors, -1, dtype=jnp.int32),
        neighbor_distances=jnp.full(max_neighbors, jnp.inf, dtype=jnp.float32),
        neighbor_count=jnp.array(0, dtype=jnp.int32),
    )
