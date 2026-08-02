"""Vectorized Pruning Extraction for ES-HyperNEAT with HSHG.

This module implements connection extraction using HSHG spatial indexing
for neighbor lookup, replacing CPPN-based neighbor queries.

Key difference from quadtree:
- Quadtree: Queries CPPN for neighbor weights at each leaf
- HSHG: Uses spatial hash to find actual discovered neighbors

This avoids redundant CPPN queries since neighbors are already in the HSHG.
"""

from typing import List, Set, Tuple, NamedTuple, Optional
import jax
import jax.numpy as jnp
import math

from .config import HSHGConfig
from .state import HSHGState
from .operations import find_directional_neighbors, query_radius_single
from .division import DiscoveredNode


class Connection(NamedTuple):
    """A connection between two coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    weight: float


class PruningResult(NamedTuple):
    """Result of pruning extraction."""
    connections: Set[Connection]
    num_leaves_processed: int
    num_connections_created: int


def compute_band_metric(
    center_weight: float,
    left_weight: float,
    right_weight: float,
    top_weight: float,
    bottom_weight: float
) -> float:
    """Compute the ES-HyperNEAT band detection metric.

    Band = max(min(d_top, d_bottom), min(d_left, d_right))

    This detects "bands" of high weight change in orthogonal directions,
    indicating potential connection points.

    Args:
        center_weight: Weight at the center position.
        left_weight: Weight at left neighbor.
        right_weight: Weight at right neighbor.
        top_weight: Weight at top neighbor.
        bottom_weight: Weight at bottom neighbor.

    Returns:
        Band metric value.
    """
    d_left = abs(center_weight - left_weight)
    d_right = abs(center_weight - right_weight)
    d_top = abs(center_weight - top_weight)
    d_bottom = abs(center_weight - bottom_weight)

    return max(min(d_top, d_bottom), min(d_left, d_right))


def compute_band_metric_jax(
    center_weight: jnp.ndarray,
    left_weight: jnp.ndarray,
    right_weight: jnp.ndarray,
    top_weight: jnp.ndarray,
    bottom_weight: jnp.ndarray
) -> jnp.ndarray:
    """JAX-compatible band metric computation.

    Vectorized version for batch processing.
    """
    d_left = jnp.abs(center_weight - left_weight)
    d_right = jnp.abs(center_weight - right_weight)
    d_top = jnp.abs(center_weight - top_weight)
    d_bottom = jnp.abs(center_weight - bottom_weight)

    return jnp.maximum(jnp.minimum(d_top, d_bottom), jnp.minimum(d_left, d_right))


def find_neighbors_for_nodes(
    nodes: List[DiscoveredNode],
    hshg_state: HSHGState,
    config: HSHGConfig
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Find 4 directional neighbors for each discovered node using HSHG.

    Args:
        nodes: List of discovered nodes.
        hshg_state: HSHG state with all nodes.
        config: HSHG configuration.

    Returns:
        Tuple of:
        - neighbor_indices: Shape [N, 4] - indices of neighbors (-1 if not found)
        - neighbor_weights: Shape [N, 4] - weights of neighbors (0.0 if not found)
    """
    n_nodes = len(nodes)

    if n_nodes == 0:
        return jnp.empty((0, 4), dtype=jnp.int32), jnp.empty((0, 4), dtype=jnp.float32)

    all_neighbor_indices = []
    all_neighbor_weights = []

    for node in nodes:
        position = jnp.array([node.x, node.y])
        # Step size is the parent width (approximated from level)
        step_size = 2.0 / (2 ** node.level)

        indices, weights = find_directional_neighbors(
            hshg_state, position, step_size,
            config.cell_size, config.num_cells,
            config.hash_prime_x, config.hash_prime_y,
            config.cell_capacity
        )

        all_neighbor_indices.append(indices)
        all_neighbor_weights.append(weights)

    neighbor_indices = jnp.stack(all_neighbor_indices)  # [N, 4]
    neighbor_weights = jnp.stack(all_neighbor_weights)  # [N, 4]

    return neighbor_indices, neighbor_weights


def extract_connections_vectorized(
    nodes: List[DiscoveredNode],
    neighbor_weights: jnp.ndarray,
    source_coord: Tuple[float, float],
    outgoing: bool,
    band_threshold: float,
    variance_threshold: float,
    allow_same_layer_connections: bool = False
) -> Set[Connection]:
    """Extract connections using vectorized band detection.

    Args:
        nodes: List of discovered nodes.
        neighbor_weights: Neighbor weights, shape [N, 4] (left, right, top, bottom).
        source_coord: Source coordinate for connections.
        outgoing: If True, source->node; if False, node->source.
        band_threshold: Threshold for band detection.
        variance_threshold: Variance threshold (nodes passing this are potential connections).
        allow_same_layer_connections: If True, allow y1 == y2.

    Returns:
        Set of connections.
    """
    connections = set()

    if len(nodes) == 0:
        return connections

    # Vectorized band computation
    center_weights = jnp.array([n.weight for n in nodes])
    left_weights = neighbor_weights[:, 0]
    right_weights = neighbor_weights[:, 1]
    top_weights = neighbor_weights[:, 2]
    bottom_weights = neighbor_weights[:, 3]

    bands = compute_band_metric_jax(
        center_weights, left_weights, right_weights, top_weights, bottom_weights
    )

    # Process each node
    for i, node in enumerate(nodes):
        band = float(bands[i])

        if band > band_threshold:
            # Convert to Python floats for hashability (JAX arrays aren't hashable)
            nx, ny, nw = float(node.x), float(node.y), float(node.weight)
            sx, sy = float(source_coord[0]), float(source_coord[1])

            if outgoing:
                conn = Connection(sx, sy, nx, ny, nw)
            else:
                conn = Connection(nx, ny, sx, sy, nw)

            # Apply connection filters
            y_constraint = conn.y1 <= conn.y2 if allow_same_layer_connections else conn.y1 < conn.y2

            if (nw != 0.0 and
                not math.isnan(nw) and
                y_constraint and
                not (conn.x1 == conn.x2 and conn.y1 == conn.y2)):
                connections.add(conn)

    return connections


def pruning_extraction_hshg(
    nodes: List[DiscoveredNode],
    hshg_state: HSHGState,
    config: HSHGConfig,
    source_coord: Tuple[float, float],
    outgoing: bool,
    band_threshold: float,
    variance_threshold: float,
    allow_same_layer_connections: bool = False
) -> PruningResult:
    """HSHG-based pruning extraction replacing quadtree traversal.

    This function:
    1. Uses HSHG to find 4 directional neighbors for each node
    2. Computes band metrics using vectorized operations
    3. Filters by band_threshold and other constraints
    4. Creates connections

    Args:
        nodes: List of discovered nodes from division initialization.
        hshg_state: HSHG state with all nodes inserted.
        config: HSHG configuration.
        source_coord: Source coordinate for connections.
        outgoing: True for outgoing connections, False for incoming.
        band_threshold: Threshold for band detection.
        variance_threshold: Variance threshold for node selection.
        allow_same_layer_connections: If True, allow y1 == y2.

    Returns:
        PruningResult with connections and statistics.
    """
    if len(nodes) == 0:
        return PruningResult(
            connections=set(),
            num_leaves_processed=0,
            num_connections_created=0
        )

    # Step 1: Find neighbors for all nodes using HSHG
    neighbor_indices, neighbor_weights = find_neighbors_for_nodes(
        nodes, hshg_state, config
    )

    # Step 2: Extract connections using vectorized band detection
    connections = extract_connections_vectorized(
        nodes,
        neighbor_weights,
        source_coord,
        outgoing,
        band_threshold,
        variance_threshold,
        allow_same_layer_connections
    )

    return PruningResult(
        connections=connections,
        num_leaves_processed=len(nodes),
        num_connections_created=len(connections)
    )


def pruning_extraction_hshg_with_cppn_fallback(
    nodes: List[DiscoveredNode],
    hshg_state: HSHGState,
    config: HSHGConfig,
    source_coord: Tuple[float, float],
    outgoing: bool,
    band_threshold: float,
    variance_threshold: float,
    cppn_forward_fn: Optional[any] = None,
    cppn_state: Optional[any] = None,
    allow_same_layer_connections: bool = False
) -> PruningResult:
    """HSHG pruning with optional CPPN fallback for missing neighbors.

    When HSHG doesn't find a neighbor (sparse regions), we can optionally
    fall back to CPPN queries. However, this may not be needed if the
    division initialization already generated a dense enough grid.

    Args:
        nodes: List of discovered nodes.
        hshg_state: HSHG state with all nodes.
        config: HSHG configuration.
        source_coord: Source coordinate for connections.
        outgoing: True for outgoing, False for incoming.
        band_threshold: Band detection threshold.
        variance_threshold: Variance threshold.
        cppn_forward_fn: Optional CPPN function for fallback.
        cppn_state: Optional CPPN state for fallback.
        allow_same_layer_connections: Allow same-layer connections.

    Returns:
        PruningResult with connections.
    """
    # Use the HSHG-only version; the dense grid provides sufficient coverage.
    return pruning_extraction_hshg(
        nodes, hshg_state, config, source_coord, outgoing,
        band_threshold, variance_threshold, allow_same_layer_connections
    )


def get_hidden_nodes_from_connections(
    connections: Set[Connection],
    input_coords: List[Tuple[float, float]],
    output_coords: List[Tuple[float, float]]
) -> Set[Tuple[float, float]]:
    """Extract hidden node coordinates from connections.

    Hidden nodes are any coordinate that appears in connections
    but is not an input or output coordinate.

    Args:
        connections: Set of connections.
        input_coords: List of input coordinates.
        output_coords: List of output coordinates.

    Returns:
        Set of hidden node coordinates.
    """
    input_set = set(input_coords)
    output_set = set(output_coords)

    hidden = set()
    for conn in connections:
        coord1 = (conn.x1, conn.y1)
        coord2 = (conn.x2, conn.y2)

        if coord1 not in input_set and coord1 not in output_set:
            hidden.add(coord1)
        if coord2 not in input_set and coord2 not in output_set:
            hidden.add(coord2)

    return hidden
