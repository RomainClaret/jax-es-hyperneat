"""Hierarchical Spatial Hash Grid (HSHG) module for ES-HyperNEAT.

This module provides O(1) average-case spatial indexing, replacing the
O(log n) quadtree operations in traditional ES-HyperNEAT.

Key Components:
    - HSHGConfig: Configuration dataclass for HSHG parameters
    - HSHGState: JAX-compatible state with fixed-shape arrays
    - QueryResult: Fixed-size query result with padding

Core Operations:
    - spatial_hash: Compute spatial hash for 2D coordinates
    - insert_single: Insert a single node
    - insert_batch: Insert multiple nodes using lax.scan
    - query_radius_single: Query nodes within radius
    - query_radius_batch: Batch query using vmap
    - find_directional_neighbors: Find 4 neighbors for band detection

Division Initialization:
    - generate_all_grid_positions: Pre-generate all positions at all depths
    - batch_query_cppn_all_pairs: Mega-batch CPPN queries
    - compute_region_variances: Variance computation using HSHG
    - division_initialization_hshg: Main division replacement

Factory Functions:
    - allocate: Create empty HSHG state from config
    - clear: Reset HSHG state while preserving shapes

Example Usage:
    ```python
    from hshg import HSHGConfig, allocate, insert_batch, query_radius_batch

    # Create configuration for ES-HyperNEAT
    config = HSHGConfig.for_es_hyperneat(max_depth=3)

    # Allocate state
    state = allocate(config)

    # Insert nodes
    state = insert_batch(state, positions, weights, levels, config)

    # Query neighbors
    results = query_radius_batch(state, centers, radius=0.5, config=config)
    ```
"""

from .config import HSHGConfig
from .state import HSHGState, QueryResult, create_empty_state, create_empty_query_result
from .operations import (
    # Hash functions
    spatial_hash,
    spatial_hash_2d,
    # Insertion
    insert_single,
    insert_batch,
    # Queries
    query_radius_single,
    query_radius_batch,
    find_directional_neighbors,
    # Factory
    allocate,
    clear,
)
from .division import (
    # Types
    GridPosition,
    DiscoveredNode,
    DivisionResult,
    # Functions
    generate_all_grid_positions,
    batch_query_cppn_all_pairs,
    compute_region_variances,
    division_initialization_hshg,
    filter_by_weight_threshold,
)
from .pruning import (
    # Types
    Connection,
    PruningResult,
    # Functions
    compute_band_metric,
    compute_band_metric_jax,
    find_neighbors_for_nodes,
    extract_connections_vectorized,
    pruning_extraction_hshg,
    get_hidden_nodes_from_connections,
)

__all__ = [
    # Config
    'HSHGConfig',
    # State
    'HSHGState',
    'QueryResult',
    'create_empty_state',
    'create_empty_query_result',
    # Hash functions
    'spatial_hash',
    'spatial_hash_2d',
    # Insertion
    'insert_single',
    'insert_batch',
    # Queries
    'query_radius_single',
    'query_radius_batch',
    'find_directional_neighbors',
    # Factory
    'allocate',
    'clear',
    # Division types
    'GridPosition',
    'DiscoveredNode',
    'DivisionResult',
    # Division functions
    'generate_all_grid_positions',
    'batch_query_cppn_all_pairs',
    'compute_region_variances',
    'division_initialization_hshg',
    'filter_by_weight_threshold',
    # Pruning types
    'Connection',
    'PruningResult',
    # Pruning functions
    'compute_band_metric',
    'compute_band_metric_jax',
    'find_neighbors_for_nodes',
    'extract_connections_vectorized',
    'pruning_extraction_hshg',
    'get_hidden_nodes_from_connections',
]
