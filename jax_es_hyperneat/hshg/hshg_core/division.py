"""Dense Grid Division Initialization for ES-HyperNEAT with HSHG.

This module replaces quadtree-based adaptive subdivision with dense grid
pre-generation combined with HSHG spatial indexing.

Key difference from quadtree:
- Quadtree: Recursively subdivides, only exploring high-variance regions
- HSHG: Pre-generates ALL positions, batch queries CPPN, filters by variance

This trades adaptivity for massive parallelism - better for JAX/GPU.
"""

from typing import List, Tuple, Set, NamedTuple, Any, Optional
import jax
import jax.numpy as jnp
import numpy as np

from .config import HSHGConfig
from .state import HSHGState
from .operations import allocate, insert_batch, query_radius_batch, clear


def sparsify_and_scale_weight(raw_weight: float, max_weight: float = 8.0,
                               threshold: float = 0.2) -> float:
    """Apply sparsification and scaling matching compiled.py behavior.

    CPPN outputs are in [-1, 1]. This function:
    1. Zeros out weak weights (|w| <= threshold)
    2. Normalizes remaining weights to [-1, 1] after removing dead zone
    3. Scales to [-max_weight, max_weight]

    Args:
        raw_weight: Raw CPPN output in [-1, 1]
        max_weight: Target weight range (default 8.0)
        threshold: Sparsification threshold (default 0.2)

    Returns:
        Scaled weight in [-max_weight, max_weight] or 0.0
    """
    if abs(raw_weight) > threshold:
        if raw_weight > 0:
            normalized = (raw_weight - threshold) / (1.0 - threshold)
        else:
            normalized = (raw_weight + threshold) / (1.0 - threshold)
        return max(-max_weight, min(normalized * max_weight, max_weight))
    return 0.0


class GridPosition(NamedTuple):
    """A position on the dense grid with its metadata."""
    x: float
    y: float
    level: int  # Quadtree-equivalent level (1, 2, 3, ...)
    width: float  # Cell width at this level


class DiscoveredNode(NamedTuple):
    """A node discovered through division initialization."""
    x: float
    y: float
    weight: float
    level: int


class DivisionResult(NamedTuple):
    """Result of HSHG-based division initialization."""
    discovered_nodes: List[DiscoveredNode]
    hshg_state: HSHGState
    num_cppn_queries: int


def generate_all_grid_positions(
    max_depth: int,
    initial_depth: int = 1,
    center: Tuple[float, float] = (0.0, 0.0),
    initial_width: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate ALL positions for all depth levels (LEGACY - spatial variance).

    Unlike quadtree which adaptively subdivides, we pre-generate all
    positions at all levels from initial_depth to max_depth.

    Args:
        max_depth: Maximum quadtree-equivalent depth.
        initial_depth: Starting depth (default 1).
        center: Center of the search space (x, y).
        initial_width: Width at level 1.

    Returns:
        Tuple of (positions, levels, widths):
        - positions: Array of [x, y] coordinates, shape [N, 2]
        - levels: Array of depth levels, shape [N]
        - widths: Array of cell widths, shape [N]
    """
    all_positions = []
    all_levels = []
    all_widths = []

    cx, cy = center

    for level in range(initial_depth, max_depth + 1):
        # At each level, grid has 2^level cells per dimension
        n_cells = 2 ** level
        width = initial_width / n_cells

        # Generate grid for this level
        # Positions are at cell centers
        for i in range(n_cells):
            for j in range(n_cells):
                # Map grid indices to [-1, 1] coordinate space
                x = -initial_width + width + 2 * i * width + cx
                y = -initial_width + width + 2 * j * width + cy

                all_positions.append([x, y])
                all_levels.append(level)
                all_widths.append(width * 2)  # Parent cell width

    positions = jnp.array(all_positions, dtype=jnp.float32)
    levels = jnp.array(all_levels, dtype=jnp.int32)
    widths = jnp.array(all_widths, dtype=jnp.float32)

    return positions, levels, widths


def generate_all_grid_positions_hierarchical(
    max_depth: int,
    center: Tuple[float, float] = (0.0, 0.0),
    initial_width: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate ALL positions ORDERED BY LEVEL for O(1) child indexing.

    Key insight: If positions are ordered [level0, level1, level2, ...],
    then parent at index i (in level L) has children at indices:
        child_base = level_offsets[L+1] + (i - level_offsets[L]) * 4
        children = child_base + [0, 1, 2, 3]

    This enables hierarchical variance computation matching quadtree semantics.

    Args:
        max_depth: Maximum quadtree-equivalent depth.
        center: Center of the search space (x, y).
        initial_width: Width at level 0 (typically 1.0 for [-1, 1] space).

    Returns:
        Tuple of (positions, levels, widths, level_offsets):
        - positions: Array of [x, y] coordinates, shape [N, 2]
        - levels: Array of depth levels, shape [N]
        - widths: Array of cell widths, shape [N]
        - level_offsets: Start index of each level, shape [max_depth+2]
    """
    all_positions = []
    all_levels = []
    all_widths = []
    level_offsets = [0]

    cx, cy = center

    for level in range(max_depth + 1):
        # At each level, grid has 2^level cells per dimension
        n_cells = 2 ** level
        # Width of each cell at this level (in coordinate space)
        cell_width = (2.0 * initial_width) / n_cells

        # Generate grid for this level in ROW-MAJOR order (consistent for parent-child mapping)
        for i in range(n_cells):
            for j in range(n_cells):
                # Position at cell center in [-initial_width, initial_width] space
                x = -initial_width + cell_width / 2 + i * cell_width + cx
                y = -initial_width + cell_width / 2 + j * cell_width + cy

                all_positions.append([x, y])
                all_levels.append(level)
                all_widths.append(cell_width)

        level_offsets.append(len(all_positions))

    positions = jnp.array(all_positions, dtype=jnp.float32)
    levels = jnp.array(all_levels, dtype=jnp.int32)
    widths = jnp.array(all_widths, dtype=jnp.float32)
    level_offsets = jnp.array(level_offsets, dtype=jnp.int32)

    return positions, levels, widths, level_offsets


def batch_query_cppn_all_pairs(
    cppn_forward_fn: Any,
    cppn_state: Any,
    source_coord: Tuple[float, float],
    target_positions: jnp.ndarray,
    outgoing: bool = True
) -> jnp.ndarray:
    """Batch query CPPN for all (source, target) pairs at once.

    This is the key optimization: instead of querying one position at a time,
    we query ALL positions in a single batched operation.

    Args:
        cppn_forward_fn: Function to evaluate CPPN, signature (state, inputs) -> outputs.
        cppn_state: CPPN state/weights.
        source_coord: Source coordinate (x, y) - fixed for all queries.
        target_positions: Target coordinates, shape [N, 2].
        outgoing: If True, source->target; if False, target->source.

    Returns:
        Weights for each target position, shape [N].
    """
    n_targets = target_positions.shape[0]

    # Create source array matching target shape
    source_x = jnp.full(n_targets, source_coord[0], dtype=jnp.float32)
    source_y = jnp.full(n_targets, source_coord[1], dtype=jnp.float32)

    target_x = target_positions[:, 0]
    target_y = target_positions[:, 1]

    if outgoing:
        # Source -> Target: CPPN input is (x1, y1, x2, y2)
        cppn_inputs = jnp.stack([source_x, source_y, target_x, target_y], axis=1)
    else:
        # Target -> Source: CPPN input is (x2, y2, x1, y1)
        cppn_inputs = jnp.stack([target_x, target_y, source_x, source_y], axis=1)

    # Batch evaluate CPPN
    # Assumes cppn_forward_fn can handle batched inputs
    outputs = cppn_forward_fn(cppn_state, cppn_inputs)

    # Extract weight output (typically first output)
    if outputs.ndim == 2:
        weights = outputs[:, 0]
    else:
        weights = outputs

    return weights


def compute_region_variances(
    positions: jnp.ndarray,
    weights: jnp.ndarray,
    levels: jnp.ndarray,
    widths: jnp.ndarray,
    hshg_state: HSHGState,
    config: HSHGConfig
) -> jnp.ndarray:
    """LEGACY: Compute variance using SPATIAL neighbors (WRONG for ES-HyperNEAT).

    NOTE: This function computes variance from spatial neighbors within a radius,
    which is NOT the correct ES-HyperNEAT quadtree semantics. Use
    compute_hierarchical_variance_hshg() instead.

    For each position, we find its neighbors within a radius and compute variance.

    Args:
        positions: Grid positions, shape [N, 2].
        weights: CPPN weights at each position, shape [N].
        levels: Depth level for each position, shape [N].
        widths: Cell width for each position, shape [N].
        hshg_state: HSHG state with positions inserted.
        config: HSHG configuration.

    Returns:
        Variance for each position, shape [N].
    """
    n_positions = positions.shape[0]

    # Query radius should be slightly larger than cell width to find neighbors
    # Use the maximum width to ensure we find all relevant neighbors
    max_width = float(jnp.max(widths))
    query_radius = max_width * 1.5

    # Batch query all positions
    query_results = query_radius_batch(
        hshg_state, positions, radius=query_radius, config=config, max_neighbors=16
    )

    # Compute variance for each position based on neighbors
    def compute_single_variance(args):
        neighbor_indices, neighbor_count, pos_idx = args

        # Get weights of valid neighbors
        valid_mask = neighbor_indices >= 0

        # Gather weights (use 0 for invalid indices)
        safe_indices = jnp.maximum(neighbor_indices, 0)
        neighbor_weights = weights[safe_indices]

        # Mask out invalid neighbors
        neighbor_weights = jnp.where(valid_mask, neighbor_weights, 0.0)
        count = jnp.sum(valid_mask.astype(jnp.float32))

        # Compute variance (avoid division by zero)
        mean = jnp.sum(neighbor_weights) / jnp.maximum(count, 1.0)
        sq_diff = jnp.where(valid_mask, (neighbor_weights - mean) ** 2, 0.0)
        variance = jnp.sum(sq_diff) / jnp.maximum(count, 1.0)

        return variance

    # vmap over all positions
    position_indices = jnp.arange(n_positions)
    variances = jax.vmap(compute_single_variance)(
        (query_results.neighbor_indices, query_results.neighbor_count, position_indices)
    )

    return variances


def compute_hierarchical_variance_hshg(
    weights: jnp.ndarray,
    level_offsets: jnp.ndarray,
    max_depth: int
) -> jnp.ndarray:
    """Compute variance using HIERARCHICAL children (correct ES-HyperNEAT semantics).

    This is the KEY FIX. For each non-leaf node, variance is computed from
    its 4 quadtree children at level+1, NOT from spatial neighbors.

    Algorithm (bottom-up):
    1. Start from level max_depth-1 (parents of leaves)
    2. For each node, find its 4 children using index arithmetic
    3. Compute variance of children's weights
    4. Propagate mean upward for parent computation

    The key insight is that with positions ordered by level:
        child_base = level_offsets[level+1] + (node_idx_in_level) * 4
        children = child_base + [0, 1, 2, 3]

    Args:
        weights: CPPN weights for all positions, shape [N].
        level_offsets: Start index of each level, shape [max_depth+2].
        max_depth: Maximum quadtree depth.

    Returns:
        Variance for each position, shape [N].
    """
    num_positions = len(weights)
    variances = jnp.zeros(num_positions, dtype=jnp.float32)
    node_values = weights.astype(jnp.float32)  # Will hold means for propagation

    # Process levels bottom-up (max_depth-1 down to 0)
    for level in range(max_depth - 1, -1, -1):
        level_start = int(level_offsets[level])
        next_level_start = int(level_offsets[level + 1])
        level_size = next_level_start - level_start

        if level_size == 0:
            continue

        # Indices of nodes at this level
        node_indices = jnp.arange(level_size) + level_start

        # Child indices for each node at this level
        # Each node at level L has 4 children at level L+1
        # The children are at: next_level_start + (node_index_in_level) * 4 + [0,1,2,3]
        node_indices_in_level = node_indices - level_start
        child_base = next_level_start + node_indices_in_level * 4

        # Get all 4 children: base + [0,1,2,3]
        child_offsets = jnp.arange(4)
        child_indices = child_base[:, None] + child_offsets[None, :]  # [level_size, 4]

        # Clamp to valid range (for leaf level, there are no children)
        safe_indices = jnp.clip(child_indices, 0, num_positions - 1)
        child_values = node_values[safe_indices]  # [level_size, 4]

        # Compute variance and mean for each node
        node_means = jnp.mean(child_values, axis=1)
        node_vars = jnp.var(child_values, axis=1)

        # Update arrays
        variances = variances.at[node_indices].set(node_vars)
        node_values = node_values.at[node_indices].set(node_means)

    return variances


def division_initialization_hshg(
    cppn_forward_fn: Any,
    cppn_state: Any,
    source_coord: Tuple[float, float],
    outgoing: bool,
    max_depth: int,
    initial_depth: int,
    division_threshold: float,
    variance_threshold: float,
    max_weight: float = 8.0,
    config: Optional[HSHGConfig] = None
) -> DivisionResult:
    """HSHG-based division initialization with HIERARCHICAL variance (FIXED).

    This function uses correct ES-HyperNEAT quadtree semantics:
    1. Pre-generate ALL positions at all depth levels (ordered by level)
    2. Batch query CPPN for all positions at once
    3. Compute HIERARCHICAL variance (parent = var of 4 children)
    4. A leaf is active if its PARENT's variance is below threshold
    5. Extract active leaves with significant weight as discovered nodes

    Args:
        cppn_forward_fn: CPPN forward function.
        cppn_state: CPPN state/weights.
        source_coord: Source coordinate for CPPN queries.
        outgoing: True for outgoing connections, False for incoming.
        max_depth: Maximum exploration depth.
        initial_depth: Initial exploration depth (not used in hierarchical mode).
        division_threshold: Variance threshold (not used - kept for API compat).
        variance_threshold: Variance threshold for node discovery.
        max_weight: Maximum weight for scaling CPPN outputs (default 8.0).
        config: HSHG configuration (auto-created if None).

    Returns:
        DivisionResult with discovered nodes, HSHG state, and query count.
    """
    # Create config if not provided
    if config is None:
        config = HSHGConfig.for_es_hyperneat(max_depth=max_depth)

    # Step 1: Generate all grid positions with hierarchical ordering
    # Includes ALL levels from 0 to max_depth for proper parent-child indexing
    positions, levels, widths, level_offsets = generate_all_grid_positions_hierarchical(
        max_depth=max_depth
    )

    # Step 2: Batch query CPPN for all positions
    raw_weights = batch_query_cppn_all_pairs(
        cppn_forward_fn, cppn_state, source_coord, positions, outgoing
    )
    num_queries = len(positions)

    # Step 2.5: CRITICAL - Sparsify weights BEFORE computing variance
    # This matches JAX-optimized behavior (vectorized_weight_sparsification)
    # Sparsification sets small values (~0) to 0, making variance more meaningful
    sparsified_weights = jnp.where(
        jnp.abs(raw_weights) < 0.2,
        0.0,  # Small values become 0
        jnp.sign(raw_weights) * (jnp.abs(raw_weights) - 0.2) / 0.8 * max_weight
    )

    # Step 3: Compute HIERARCHICAL variance on SPARSIFIED weights (THE KEY FIX)
    # Variance is computed from 4 children, not spatial neighbors
    variances = compute_hierarchical_variance_hshg(sparsified_weights, level_offsets, max_depth)

    # Step 4: Extract ALL leaves with significant weight - let pruning handle band detection
    #
    # Filtering leaves by parent variance would cause underdiscovery here. The quadtree
    # implementation uses variance to decide whether to RECURSE DEEPER (high variance) or
    # CHECK BAND DISCONTINUITY (low variance). Since HSHG pre-generates all positions at
    # max_depth, the recursion decision does not apply: pass ALL significant-weight leaves
    # to pruning, whose band detection filters out nodes without sufficient weight
    # discontinuity. With max_depth=1, both implementations discover ~3 hidden nodes (not 1).
    leaf_mask = levels == max_depth
    leaf_indices = jnp.where(leaf_mask)[0]

    # Step 5: Extract discovered nodes - pass ALL significant-weight leaves to pruning
    discovered_nodes = []

    for i, leaf_idx in enumerate(leaf_indices):
        leaf_idx = int(leaf_idx)

        # Use already-sparsified weight
        weight = float(sparsified_weights[leaf_idx])

        # Only include if weight is significant (non-zero after sparsification)
        # Let pruning's band detection handle the rest
        if abs(weight) > 0.0:
            discovered_nodes.append(DiscoveredNode(
                x=float(positions[leaf_idx, 0]),
                y=float(positions[leaf_idx, 1]),
                weight=weight,
                level=int(levels[leaf_idx])
            ))

    # Step 6: Build HSHG state for potential band detection use
    # Use sparsified weights for consistency
    hshg_state = allocate(config)
    hshg_state = insert_batch(hshg_state, positions, sparsified_weights, levels, config)

    return DivisionResult(
        discovered_nodes=discovered_nodes,
        hshg_state=hshg_state,
        num_cppn_queries=num_queries
    )


def filter_by_weight_threshold(
    nodes: List[DiscoveredNode],
    weight_threshold: float
) -> List[DiscoveredNode]:
    """Filter discovered nodes by weight threshold.

    Args:
        nodes: List of discovered nodes.
        weight_threshold: Minimum absolute weight to keep.

    Returns:
        Filtered list of nodes.
    """
    return [n for n in nodes if abs(n.weight) > weight_threshold]
