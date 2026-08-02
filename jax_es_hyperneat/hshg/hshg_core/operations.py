"""Core HSHG operations for ES-HyperNEAT spatial indexing.

This module implements O(1) average-case spatial operations using
Hierarchical Spatial Hash Grids, replacing O(log n) quadtree operations.

All functions are designed to be JAX JIT-compatible with pure functional updates.
"""

import functools
from typing import Tuple
import jax
import jax.numpy as jnp

from .config import HSHGConfig
from .state import HSHGState, QueryResult, create_empty_state, create_empty_query_result


# ==============================================================================
# Hash Functions
# ==============================================================================

def spatial_hash(
    x: jnp.ndarray,
    y: jnp.ndarray,
    cell_size: float,
    num_cells: int,
    hash_prime_x: int,
    hash_prime_y: int
) -> jnp.ndarray:
    """Compute spatial hash for 2D coordinates.

    Uses Wang hash with primes optimized for spatial coherence.
    Vectorizable over arbitrary batch dimensions.

    Args:
        x: X coordinates, any shape.
        y: Y coordinates, same shape as x.
        cell_size: Size of each hash cell.
        num_cells: Total number of cells in hash table.
        hash_prime_x: Prime for x-coordinate hashing.
        hash_prime_y: Prime for y-coordinate hashing.

    Returns:
        Hash indices, same shape as inputs.
    """
    # Quantize to cell coordinates
    cx = jnp.floor(x / cell_size).astype(jnp.int32)
    cy = jnp.floor(y / cell_size).astype(jnp.int32)

    # Wang hash with spatial primes
    hash_val = cx * hash_prime_x ^ cy * hash_prime_y

    # Map to valid range
    return jnp.abs(hash_val) % num_cells


def spatial_hash_2d(
    positions: jnp.ndarray,
    cell_size: float,
    num_cells: int,
    hash_prime_x: int,
    hash_prime_y: int
) -> jnp.ndarray:
    """Compute spatial hash for array of 2D positions.

    Args:
        positions: Positions array, shape [..., 2].
        cell_size: Size of each hash cell.
        num_cells: Total number of cells in hash table.
        hash_prime_x: Prime for x-coordinate hashing.
        hash_prime_y: Prime for y-coordinate hashing.

    Returns:
        Hash indices, shape [...].
    """
    return spatial_hash(
        positions[..., 0],
        positions[..., 1],
        cell_size,
        num_cells,
        hash_prime_x,
        hash_prime_y
    )


# ==============================================================================
# Insertion Operations
# ==============================================================================

def insert_single(
    state: HSHGState,
    position: jnp.ndarray,
    weight: float,
    level: int,
    cell_size: float,
    num_cells: int,
    hash_prime_x: int,
    hash_prime_y: int,
    cell_capacity: int,
    hierarchy_level: int = 0
) -> HSHGState:
    """Insert a single node into HSHG.

    Pure functional - returns new state without modifying input.

    Args:
        state: Current HSHG state.
        position: 2D position [x, y].
        weight: CPPN weight at position.
        level: Quadtree level at which node was discovered.
        cell_size: Cell size for this hierarchy level.
        num_cells: Number of cells in hash table.
        hash_prime_x: Prime for x-coordinate hashing.
        hash_prime_y: Prime for y-coordinate hashing.
        cell_capacity: Maximum nodes per cell.
        hierarchy_level: Which hierarchy level to insert into.

    Returns:
        Updated HSHGState.
    """
    node_idx = state.num_nodes

    # Check for node overflow
    node_overflow = node_idx >= state.positions.shape[0]

    # Compute cell hash
    cell_idx = spatial_hash(
        position[0], position[1],
        cell_size, num_cells, hash_prime_x, hash_prime_y
    )

    # Get current cell count
    current_count = state.cell_counts[hierarchy_level, cell_idx]

    # Check for cell overflow
    cell_overflow = current_count >= cell_capacity

    # Combined overflow flag
    any_overflow = node_overflow | cell_overflow

    # Update positions (only if no overflow)
    new_positions = jax.lax.cond(
        any_overflow,
        lambda: state.positions,
        lambda: state.positions.at[node_idx].set(position)
    )

    # Update weights
    new_weights = jax.lax.cond(
        any_overflow,
        lambda: state.weights,
        lambda: state.weights.at[node_idx].set(weight)
    )

    # Update levels
    new_levels = jax.lax.cond(
        any_overflow,
        lambda: state.levels,
        lambda: state.levels.at[node_idx].set(level)
    )

    # Update valid mask
    new_valid = jax.lax.cond(
        any_overflow,
        lambda: state.valid_mask,
        lambda: state.valid_mask.at[node_idx].set(True)
    )

    # Update cell (insert node index into cell)
    slot_idx = current_count  # Next available slot
    new_cells = jax.lax.cond(
        any_overflow,
        lambda: state.cells,
        lambda: state.cells.at[hierarchy_level, cell_idx, slot_idx].set(node_idx)
    )

    # Update cell count
    new_cell_counts = jax.lax.cond(
        any_overflow,
        lambda: state.cell_counts,
        lambda: state.cell_counts.at[hierarchy_level, cell_idx].set(current_count + 1)
    )

    # Update node count
    new_num_nodes = jax.lax.cond(
        any_overflow,
        lambda: state.num_nodes,
        lambda: state.num_nodes + 1
    )

    # Update overflow flag
    new_overflow = state.did_buffer_overflow | any_overflow

    return HSHGState(
        positions=new_positions,
        weights=new_weights,
        levels=new_levels,
        valid_mask=new_valid,
        num_nodes=new_num_nodes,
        cells=new_cells,
        cell_counts=new_cell_counts,
        did_buffer_overflow=new_overflow,
    )


def insert_batch(
    state: HSHGState,
    positions: jnp.ndarray,
    weights: jnp.ndarray,
    levels: jnp.ndarray,
    config: HSHGConfig
) -> HSHGState:
    """Insert a batch of nodes into HSHG.

    Uses jax.lax.scan for efficient sequential insertion with JIT.

    Args:
        state: Current HSHG state.
        positions: Positions array, shape [N, 2].
        weights: Weights array, shape [N].
        levels: Levels array, shape [N].
        config: HSHG configuration.

    Returns:
        Updated HSHGState with all nodes inserted.
    """
    def scan_fn(carry_state, x):
        pos, weight, level = x
        # Insert into hierarchy level 0 (finest)
        new_state = insert_single(
            carry_state, pos, weight, level,
            config.cell_size, config.num_cells,
            config.hash_prime_x, config.hash_prime_y,
            config.cell_capacity, hierarchy_level=0
        )
        return new_state, None

    final_state, _ = jax.lax.scan(
        scan_fn,
        state,
        (positions, weights, levels)
    )

    return final_state


# ==============================================================================
# Query Operations
# ==============================================================================

@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8, 9))
def query_radius_single(
    state: HSHGState,
    center: jnp.ndarray,
    radius: float,
    hierarchy_level: int,
    cell_size: float,
    num_cells: int,
    hash_prime_x: int,
    hash_prime_y: int,
    cell_capacity: int,
    max_neighbors: int = 100
) -> QueryResult:
    """Query all nodes within radius of center point.

    O(1) average case complexity via spatial hashing.

    Args:
        state: Current HSHG state.
        center: Query center point [x, y].
        radius: Query radius.
        hierarchy_level: Which hierarchy level to query.
        cell_size: Cell size for this hierarchy level.
        num_cells: Number of cells in hash table.
        hash_prime_x: Prime for x-coordinate hashing.
        hash_prime_y: Prime for y-coordinate hashing.
        cell_capacity: Maximum nodes per cell.
        max_neighbors: Maximum neighbors to return.

    Returns:
        QueryResult with neighbor indices, distances, and count.
    """
    # Generate all candidate cell offsets (5x5 grid covers most cases)
    offset_range = jnp.arange(-2, 3)  # [-2, -1, 0, 1, 2]
    offset_x, offset_y = jnp.meshgrid(offset_range, offset_range)
    offsets = jnp.stack([offset_x.ravel(), offset_y.ravel()], axis=1)  # [25, 2]

    # Get center cell coordinates
    center_cx = jnp.floor(center[0] / cell_size).astype(jnp.int32)
    center_cy = jnp.floor(center[1] / cell_size).astype(jnp.int32)

    # Compute hash for all candidate cells
    candidate_cx = center_cx + offsets[:, 0]
    candidate_cy = center_cy + offsets[:, 1]
    candidate_hashes = (candidate_cx * hash_prime_x ^ candidate_cy * hash_prime_y)
    candidate_hashes = jnp.abs(candidate_hashes) % num_cells

    # Gather all node indices from candidate cells
    # Shape: [num_candidate_cells, cell_capacity]
    candidate_node_indices = state.cells[hierarchy_level, candidate_hashes]

    # Flatten to [num_candidates * cell_capacity]
    all_candidates = candidate_node_indices.ravel()

    # Filter by validity and distance
    def check_candidate(node_idx):
        # Get position (use zeros for invalid indices)
        safe_idx = jnp.maximum(node_idx, 0)
        pos = state.positions[safe_idx]

        # Compute distance
        dist = jnp.sqrt(jnp.sum((pos - center) ** 2))

        # Check validity: valid node AND within radius
        is_valid = (
            (node_idx >= 0) &
            state.valid_mask[safe_idx] &
            (dist <= radius)
        )

        return jnp.where(is_valid, node_idx, -1), jnp.where(is_valid, dist, jnp.inf)

    # Vectorized check over all candidates
    neighbor_indices, neighbor_distances = jax.vmap(check_candidate)(all_candidates)

    # Sort by distance and take top max_neighbors
    sort_idx = jnp.argsort(neighbor_distances)[:max_neighbors]
    neighbor_indices = neighbor_indices[sort_idx]
    neighbor_distances = neighbor_distances[sort_idx]

    # Count valid neighbors (distance < inf)
    neighbor_count = jnp.sum(neighbor_indices >= 0)

    return QueryResult(
        neighbor_indices=neighbor_indices,
        neighbor_distances=neighbor_distances,
        neighbor_count=neighbor_count,
    )


def query_radius_batch(
    state: HSHGState,
    centers: jnp.ndarray,
    radius: float,
    config: HSHGConfig,
    max_neighbors: int = 100
) -> QueryResult:
    """Batch query using vmap over multiple centers.

    Args:
        state: Current HSHG state.
        centers: Query centers, shape [N, 2].
        radius: Query radius (same for all queries).
        config: HSHG configuration.
        max_neighbors: Maximum neighbors per query.

    Returns:
        QueryResult with batched results, shapes [N, max_neighbors].
    """
    # Create a wrapper that captures all static args
    def single_query(center):
        return query_radius_single(
            state, center, radius,
            hierarchy_level=0,
            cell_size=config.cell_size,
            num_cells=config.num_cells,
            hash_prime_x=config.hash_prime_x,
            hash_prime_y=config.hash_prime_y,
            cell_capacity=config.cell_capacity,
            max_neighbors=max_neighbors,
        )

    # vmap over centers
    return jax.vmap(single_query)(centers)


# ==============================================================================
# Neighbor Finding (for band detection)
# ==============================================================================

@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def find_directional_neighbors(
    state: HSHGState,
    position: jnp.ndarray,
    step_size: float,
    cell_size: float,
    num_cells: int,
    hash_prime_x: int,
    hash_prime_y: int,
    cell_capacity: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Find the 4 directional neighbors (left, right, top, bottom).

    Used for band detection in ES-HyperNEAT pruning extraction.

    Args:
        state: Current HSHG state.
        position: Query position [x, y].
        step_size: Distance to look for neighbors (parent cell width).
        cell_size: Cell size for hashing.
        num_cells: Number of cells in hash table.
        hash_prime_x: Prime for x-coordinate hashing.
        hash_prime_y: Prime for y-coordinate hashing.
        cell_capacity: Maximum nodes per cell.

    Returns:
        Tuple of (neighbor_indices [4], neighbor_weights [4]).
        Indices are -1 if no neighbor found.
    """
    # 4 neighbor positions: left, right, top, bottom
    neighbor_positions = jnp.array([
        [position[0] - step_size, position[1]],  # left
        [position[0] + step_size, position[1]],  # right
        [position[0], position[1] - step_size],  # top (y decreases)
        [position[0], position[1] + step_size],  # bottom (y increases)
    ])

    def find_nearest_in_cell(query_pos):
        """Find nearest node to query_pos using HSHG."""
        # Hash the query position
        cell_idx = spatial_hash(
            query_pos[0], query_pos[1],
            cell_size, num_cells, hash_prime_x, hash_prime_y
        )

        # Get nodes in this cell
        cell_nodes = state.cells[0, cell_idx]  # [cell_capacity]

        # Find nearest valid node
        def check_node(node_idx):
            safe_idx = jnp.maximum(node_idx, 0)
            is_valid = (node_idx >= 0) & state.valid_mask[safe_idx]
            pos = state.positions[safe_idx]
            dist = jnp.sqrt(jnp.sum((pos - query_pos) ** 2))
            return jnp.where(is_valid, dist, jnp.inf), node_idx

        distances, indices = jax.vmap(check_node)(cell_nodes)
        min_idx = jnp.argmin(distances)
        min_dist = distances[min_idx]

        # Return node index and weight if found within step_size
        found = min_dist < step_size * 1.5  # Allow some tolerance
        result_idx = jnp.where(found, indices[min_idx], -1)
        safe_result_idx = jnp.maximum(result_idx, 0)
        result_weight = jnp.where(
            found & (result_idx >= 0),
            state.weights[safe_result_idx],
            0.0
        )

        return result_idx, result_weight

    # Find neighbor for each direction
    neighbor_indices, neighbor_weights = jax.vmap(find_nearest_in_cell)(neighbor_positions)

    return neighbor_indices, neighbor_weights


# ==============================================================================
# HSHG Factory Functions
# ==============================================================================

def allocate(config: HSHGConfig) -> HSHGState:
    """Allocate empty HSHG state based on configuration.

    Args:
        config: HSHG configuration.

    Returns:
        Empty HSHGState ready for insertions.
    """
    return create_empty_state(
        max_nodes=config.max_nodes,
        num_levels=config.num_levels,
        num_cells=config.num_cells,
        cell_capacity=config.cell_capacity,
    )


def clear(state: HSHGState) -> HSHGState:
    """Clear all nodes from HSHG state while preserving array shapes.

    Args:
        state: Current HSHG state.

    Returns:
        Cleared HSHGState.
    """
    return HSHGState(
        positions=jnp.zeros_like(state.positions),
        weights=jnp.zeros_like(state.weights),
        levels=jnp.zeros_like(state.levels),
        valid_mask=jnp.zeros_like(state.valid_mask),
        num_nodes=jnp.array(0, dtype=jnp.int32),
        cells=jnp.full_like(state.cells, -1),
        cell_counts=jnp.zeros_like(state.cell_counts),
        did_buffer_overflow=jnp.array(False, dtype=bool),
    )
