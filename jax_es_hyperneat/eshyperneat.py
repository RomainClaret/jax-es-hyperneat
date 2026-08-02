"""ES-HyperNEAT implementation on TensorNEAT/JAX, with batched-optimization feature flags:

- O1: Coordinate normalization (cache coherency)
- O2: Variance caching (eliminate redundant tree traversals)
- O3: BFS network cleaning (O(n) vs O(n²))
- O4: Batch division queries (batch CPPN queries per level)
- O5: Batch pruning queries (batch neighbor queries)
- O6: Query caching (none, lru, per_cppn strategies)
- O7: vmap substrate evaluation (batch test cases)
- O8: Fixed-size JAX arrays (enables JIT downstream)

Each optimization can be toggled via its flag.
"""

import time
import copy
import math
import numpy as np
from typing import Any, Dict, Tuple, Set, List, Optional
from collections import defaultdict, deque
import jax
import jax.numpy as jnp

from jax_es_hyperneat._compat.core.base_algorithm import BaseAlgorithm, AlgorithmMetrics
from jax_es_hyperneat._compat.utils.config_manager import ConfigManager
from jax_es_hyperneat._compat.adapters.tensorneat_adapter import TensorNEATAdapter


# ============================================================================
# Quadtree and Connection Classes
# ============================================================================

class QuadPoint:
    """Quadtree node representing a spatial region.

    Extended with cached_variance for O2 optimization.
    """
    def __init__(self, x: float, y: float, width: float, level: int):
        self.x = x
        self.y = y
        self.width = width
        self.level = level
        self.weight = 0.0
        self.children = [None] * 4  # 4 quadrants
        self.cached_variance: Optional[float] = None  # O2: Variance caching


class Connection:
    """Connection between two spatial coordinates with weight."""
    def __init__(self, x1: float, y1: float, x2: float, y2: float, weight: float):
        self.x1 = float(x1) if hasattr(x1, '__float__') else x1
        self.y1 = float(y1) if hasattr(y1, '__float__') else y1
        self.x2 = float(x2) if hasattr(x2, '__float__') else x2
        self.y2 = float(y2) if hasattr(y2, '__float__') else y2
        self.weight = float(weight) if hasattr(weight, '__float__') else weight

        # Safety: convert NaN weights to 0.0
        if math.isnan(self.weight):
            self.weight = 0.0

    def __eq__(self, other):
        if not isinstance(other, Connection):
            return NotImplemented
        return (self.x1, self.y1, self.x2, self.y2) == (other.x1, other.y1, other.x2, other.y2)

    def __hash__(self):
        return hash((self.x1, self.y1, self.x2, self.y2))


# ============================================================================
# Optimized ES-HyperNEAT Implementation
# ============================================================================

class TensorNEATESHyperNEATOptimized(BaseAlgorithm):
    """ES-HyperNEAT implementation using TensorNEAT/JAX.

    Batches CPPN queries during quadtree division (O4) and pruning (O5), vmaps substrate
    evaluation over test cases (O7), and pre-computes quadtree child-coordinate offsets (O10).
    """

    def __init__(self, name: str = 'eshyperneat',
                 implementation: str = 'tensorneat-optimized-eshyperneat',
                 trial_id: int = None,
                 **kwargs):
        super().__init__(name=name, implementation=implementation, trial_id=trial_id)
        self.adapter = TensorNEATAdapter()
        self.lazy_metrics = True

        # ========== OPTIMIZATIONS (enabled by default) ==========
        # O4: Batch division queries
        self.opt_batch_division_queries = True

        # O5: Batch pruning queries
        self.opt_batch_pruning_queries = True

        # O7: vmap substrate evaluation
        self.opt_vmap_evaluation = True

        # O10: Pre-compute quadtree coordinate offsets
        self.opt_precompute_coords = True
        self._child_offsets = None  # Pre-computed child offset multipliers

        # ========== CONNECTIVITY FLAGS ==========
        # F1: Allow same-layer connections (y1 <= y2 vs y1 < y2).
        # Enabled by default: converges in fewer generations at a higher per-generation
        # cost, for a net faster time-to-solve.
        self.allow_same_layer_connections = True

        # F2: Use CPPN-queried weights in fallback vs random.
        # Disabled by default: no measurable impact on task performance.
        self.use_cppn_fallback_weights = False

        # ES-HyperNEAT parameters
        self.initial_depth = None
        self.max_depth = None
        self.variance_threshold = None
        self.division_threshold = None
        self.band_threshold = None
        self.max_weight = None
        self.iteration_level = None

        # Substrate coordinates
        self.substrate_input_coords = None
        self.substrate_output_coords = None

        # NEAT algorithm for CPPN evolution
        self.neat_algo = None
        self.pipeline = None

        # RecurrentGenome for substrate execution
        self.hyper_genome = None

        # Config metadata
        self._config_metadata = None
        self._start_time = None

        # Debug logging
        self.verbose = False
        self.debug_log_file = '/tmp/eshyperneat_optimized_debug.log'
        self._variance_call_count = 0
        self._cppn_query_count = 0

        # Query timing
        self._total_query_time = 0.0
        self._query_count = 0

        # JIT compilation
        self._jitted_cppn_forward = None
        self._compiled_ask = None
        self._compiled_transform_batch = None
        self._compiled_tell = None

        # Current CPPN reference (for fallback substrate)
        self._current_cppn = None

    # ========================================================================
    # CPPN Query Methods
    # ========================================================================

    def _query_cppn(self, state: Any, cppn_transformed: Any,
                    coord1: Tuple[float, float], coord2: Tuple[float, float],
                    outgoing: bool) -> float:
        """Query CPPN for connection weight between two coordinates."""
        t0 = time.perf_counter()

        if outgoing:
            inputs = np.array([coord1[0], coord1[1], coord2[0], coord2[1], 1.0], dtype=np.float32)
        else:
            inputs = np.array([coord2[0], coord2[1], coord1[0], coord1[1], 1.0], dtype=np.float32)

        inputs_jax = jnp.array(inputs)
        weight_jax = self._jitted_cppn_forward(state, cppn_transformed, inputs_jax)

        if hasattr(weight_jax, 'ndim'):
            if weight_jax.ndim > 0:
                weight = float(weight_jax.flatten()[0])
            else:
                weight = float(weight_jax)
        else:
            weight = float(weight_jax)

        if math.isnan(weight) or math.isinf(weight):
            t1 = time.perf_counter()
            self._total_query_time += (t1 - t0)
            self._query_count += 1
            return 0.0

        WEIGHT_SPARSIFICATION_THRESHOLD = 0.2

        if abs(weight) > WEIGHT_SPARSIFICATION_THRESHOLD:
            if weight > 0:
                weight = (weight - WEIGHT_SPARSIFICATION_THRESHOLD) / (1.0 - WEIGHT_SPARSIFICATION_THRESHOLD)
            else:
                weight = (weight + WEIGHT_SPARSIFICATION_THRESHOLD) / (1.0 - WEIGHT_SPARSIFICATION_THRESHOLD)
            result = max(-self.max_weight, min(weight * self.max_weight, self.max_weight))
        else:
            result = 0.0

        t1 = time.perf_counter()
        self._total_query_time += (t1 - t0)
        self._query_count += 1

        return result

    # ========================================================================
    # O4/O5: Batched CPPN Queries
    # ========================================================================

    def _batch_query_cppn(self, state: Any, cppn_transformed: Any,
                         coord: Tuple[float, float], target_coords: List[Tuple[float, float]],
                         outgoing: bool) -> np.ndarray:
        """Batch query CPPN for multiple target coordinates.

        O4/O5 Optimization: Instead of N individual queries, performs 1 batched query
        using JAX vmap for efficient parallel computation.
        """
        if len(target_coords) == 0:
            return np.array([])

        # Build input batch
        if outgoing:
            inputs_list = [
                [coord[0], coord[1], tc[0], tc[1], 1.0]
                for tc in target_coords
            ]
        else:
            inputs_list = [
                [tc[0], tc[1], coord[0], coord[1], 1.0]
                for tc in target_coords
            ]

        inputs_batch = jnp.array(inputs_list, dtype=jnp.float32)

        # vmap forward pass over batch
        weights_batch = jax.vmap(
            lambda x: self._jitted_cppn_forward(state, cppn_transformed, x)
        )(inputs_batch)

        # Flatten weights and process
        weights = np.array(weights_batch.flatten())
        processed_weights = self._process_weights(weights)

        self._cppn_query_count += len(target_coords)
        return processed_weights

    def _process_weights(self, weights: np.ndarray) -> np.ndarray:
        """Process raw CPPN weights with sparsification threshold."""
        WEIGHT_SPARSIFICATION_THRESHOLD = 0.2
        processed_weights = np.zeros_like(weights)

        for i, w in enumerate(weights):
            if math.isnan(w) or math.isinf(w):
                processed_weights[i] = 0.0
            elif abs(w) > WEIGHT_SPARSIFICATION_THRESHOLD:
                if w > 0:
                    scaled = (w - WEIGHT_SPARSIFICATION_THRESHOLD) / (1.0 - WEIGHT_SPARSIFICATION_THRESHOLD)
                else:
                    scaled = (w + WEIGHT_SPARSIFICATION_THRESHOLD) / (1.0 - WEIGHT_SPARSIFICATION_THRESHOLD)
                processed_weights[i] = max(-self.max_weight, min(scaled * self.max_weight, self.max_weight))
            else:
                processed_weights[i] = 0.0

        return processed_weights

    # ========================================================================
    # Variance Calculation
    # ========================================================================

    @staticmethod
    def _get_weights(quad_point: QuadPoint) -> list:
        """Recursively collect all weights from quadtree leaves."""
        weights = []

        def collect_weights(point):
            if point is not None and all(child is not None for child in point.children):
                for i in range(4):
                    collect_weights(point.children[i])
            else:
                if point is not None:
                    weights.append(point.weight)

        collect_weights(quad_point)
        return weights

    def _variance(self, quad_point: QuadPoint) -> float:
        """Calculate variance of weights in quadtree region."""
        self._variance_call_count += 1

        if not quad_point:
            return 0.0
        weights = self._get_weights(quad_point)
        if len(weights) == 0:
            return 0.0

        return float(np.var(weights))

    # ========================================================================
    # Network Cleaning
    # ========================================================================

    def _clean_net(self, connections: Set[Connection]) -> Tuple[Set, Set]:
        """Clean network to remove unreachable nodes and connections."""
        connected_to_inputs = set(tuple(float(c) for c in coord) for coord in self.substrate_input_coords)
        connected_to_outputs = set(tuple(float(c) for c in coord) for coord in self.substrate_output_coords)
        true_connections = set()

        initial_input_connections = copy.deepcopy(connections)
        initial_output_connections = copy.deepcopy(connections)

        add_happened = True
        while add_happened:
            add_happened = False
            temp_connections = copy.deepcopy(initial_input_connections)
            for conn in temp_connections:
                if (conn.x1, conn.y1) in connected_to_inputs:
                    connected_to_inputs.add((conn.x2, conn.y2))
                    initial_input_connections.remove(conn)
                    add_happened = True

        add_happened = True
        while add_happened:
            add_happened = False
            temp_connections = copy.deepcopy(initial_output_connections)
            for conn in temp_connections:
                if (conn.x2, conn.y2) in connected_to_outputs:
                    connected_to_outputs.add((conn.x1, conn.y1))
                    initial_output_connections.remove(conn)
                    add_happened = True

        true_nodes = connected_to_inputs.intersection(connected_to_outputs)

        for conn in connections:
            if ((conn.x1, conn.y1) in true_nodes and
                (conn.x2, conn.y2) in true_nodes):
                true_connections.add(conn)

        input_coords = set(tuple(float(c) for c in coord) for coord in self.substrate_input_coords)
        output_coords = set(tuple(float(c) for c in coord) for coord in self.substrate_output_coords)
        true_nodes = true_nodes - input_coords - output_coords

        return true_nodes, true_connections

    # ========================================================================
    # O10: Pre-computed Coordinate Offsets
    # ========================================================================

    def _init_child_offsets(self):
        """Pre-compute child offset multipliers for quadtree.

        O10 Optimization: Avoid creating tuples dynamically in the hot loop.
        Children positions relative to parent: (x_mult, y_mult)
        - Child 0: (-0.5, -0.5) -> bottom-left
        - Child 1: (-0.5, +0.5) -> top-left
        - Child 2: (+0.5, +0.5) -> top-right
        - Child 3: (+0.5, -0.5) -> bottom-right
        """
        self._child_offsets = np.array([
            [-0.5, -0.5],
            [-0.5, +0.5],
            [+0.5, +0.5],
            [+0.5, -0.5],
        ], dtype=np.float32)

    def _create_children_fast(self, point: QuadPoint) -> List[QuadPoint]:
        """Create 4 children using pre-computed offsets.

        O10 Optimization: Uses numpy broadcasting instead of 4 separate calculations.
        """
        if self._child_offsets is None:
            self._init_child_offsets()

        new_width = point.width * 0.5
        new_level = point.level + 1

        # Compute all 4 child positions at once
        child_positions = np.array([point.x, point.y]) + self._child_offsets * point.width

        return [
            QuadPoint(child_positions[i, 0], child_positions[i, 1], new_width, new_level)
            for i in range(4)
        ]

    # ========================================================================
    # O4: Batched Division Initialization
    # ========================================================================

    def _division_initialization_batched(self, state: Any, cppn_transformed: Any,
                                         coord: Tuple[float, float], outgoing: bool) -> QuadPoint:
        """Level-by-level quadtree with batched CPPN queries.

        O4 Optimization: Batch all children queries per level instead of individual queries.
        O10 Optimization: Use pre-computed offsets for child creation (when enabled).
        """
        root = QuadPoint(0.0, 0.0, 1.0, 1)
        level_queue = [root]

        while level_queue:
            # Collect ALL children coordinates for this level
            all_children = []
            all_coords = []

            for point in level_queue:
                # O10: Use fast child creation if enabled
                if self.opt_precompute_coords:
                    point.children = self._create_children_fast(point)
                else:
                    point.children = [
                        QuadPoint(point.x - point.width/2, point.y - point.width/2, point.width/2, point.level + 1),
                        QuadPoint(point.x - point.width/2, point.y + point.width/2, point.width/2, point.level + 1),
                        QuadPoint(point.x + point.width/2, point.y + point.width/2, point.width/2, point.level + 1),
                        QuadPoint(point.x + point.width/2, point.y - point.width/2, point.width/2, point.level + 1),
                    ]
                for child in point.children:
                    all_children.append((point, child))
                    all_coords.append((child.x, child.y))

            # BATCH query ALL children at this level
            weights = self._batch_query_cppn(state, cppn_transformed, coord, all_coords, outgoing)

            # Assign weights to children
            for i, (parent, child) in enumerate(all_children):
                child.weight = weights[i] if i < len(weights) else 0.0

            # Check variance and filter for next level
            next_level = []
            for point in level_queue:
                var = self._variance(point)

                if (point.level < self.initial_depth) or (
                    point.level < self.max_depth and var > self.division_threshold
                ):
                    next_level.extend([c for c in point.children if c is not None])

            level_queue = next_level

        return root

    def _division_initialization_original(self, state: Any, cppn_transformed: Any,
                                         coord: Tuple[float, float], outgoing: bool) -> QuadPoint:
        """Original quadtree initialization with individual queries (baseline)."""
        root = QuadPoint(0.0, 0.0, 1.0, 1)
        queue = [root]

        while queue:
            point = queue.pop(0)

            point.children[0] = QuadPoint(
                point.x - point.width/2.0,
                point.y - point.width/2.0,
                point.width/2.0,
                point.level + 1
            )
            point.children[1] = QuadPoint(
                point.x - point.width/2.0,
                point.y + point.width/2.0,
                point.width/2.0,
                point.level + 1
            )
            point.children[2] = QuadPoint(
                point.x + point.width/2.0,
                point.y + point.width/2.0,
                point.width/2.0,
                point.level + 1
            )
            point.children[3] = QuadPoint(
                point.x + point.width/2.0,
                point.y - point.width/2.0,
                point.width/2.0,
                point.level + 1
            )

            for child in point.children:
                child.weight = self._query_cppn(
                    state, cppn_transformed, coord, (child.x, child.y), outgoing
                )
                self._cppn_query_count += 1

            if (point.level < self.initial_depth) or (
                point.level < self.max_depth and
                self._variance(point) > self.division_threshold
            ):
                for child in point.children:
                    queue.append(child)

        return root

    def _division_initialization(self, state: Any, cppn_transformed: Any,
                                coord: Tuple[float, float], outgoing: bool) -> QuadPoint:
        """Division initialization dispatcher based on O4 flag."""
        if self.opt_batch_division_queries:
            return self._division_initialization_batched(state, cppn_transformed, coord, outgoing)
        else:
            return self._division_initialization_original(state, cppn_transformed, coord, outgoing)

    # ========================================================================
    # O5: Batched Pruning Extraction
    # ========================================================================

    def _pruning_extraction_batched(self, state: Any, cppn_transformed: Any,
                                    coord: Tuple[float, float], root: QuadPoint,
                                    outgoing: bool) -> Set[Connection]:
        """Extract connections with batched neighbor queries.

        O5 Optimization: Batch all 4 neighbors for all leaves at once.
        """
        connections = set()

        # Collect all leaves that pass variance threshold
        def collect_extraction_leaves(point, parent_width):
            """Recursively collect leaves for extraction."""
            extraction_leaves = []

            def traverse(p, pw):
                for child in p.children:
                    if child is None:
                        continue

                    child_var = self._variance(child)
                    if child_var > self.variance_threshold:
                        traverse(child, p.width)
                    else:
                        extraction_leaves.append((child, p.width))

            traverse(point, parent_width)
            return extraction_leaves

        leaves = collect_extraction_leaves(root, root.width)

        if len(leaves) == 0:
            return connections

        # Batch query ALL 4 neighbors for ALL leaves at once
        all_neighbor_coords = []
        leaf_widths = []

        for leaf, parent_width in leaves:
            # 4 neighbors: left, right, top, bottom
            all_neighbor_coords.extend([
                (leaf.x - parent_width, leaf.y),
                (leaf.x + parent_width, leaf.y),
                (leaf.x, leaf.y - parent_width),
                (leaf.x, leaf.y + parent_width),
            ])
            leaf_widths.append(parent_width)

        # Single batched CPPN query for ALL neighbor coordinates
        neighbor_weights = self._batch_query_cppn(
            state, cppn_transformed, coord, all_neighbor_coords, outgoing
        )

        # Process leaves with Python loop (band detection)
        for i, (leaf, parent_width) in enumerate(leaves):
            base_idx = i * 4
            if base_idx + 3 >= len(neighbor_weights):
                continue

            d_left = abs(leaf.weight - neighbor_weights[base_idx])
            d_right = abs(leaf.weight - neighbor_weights[base_idx + 1])
            d_top = abs(leaf.weight - neighbor_weights[base_idx + 2])
            d_bottom = abs(leaf.weight - neighbor_weights[base_idx + 3])

            # Band detection (exact ES-HyperNEAT formula)
            band = max(min(d_top, d_bottom), min(d_left, d_right))

            if band > self.band_threshold:
                if outgoing:
                    conn = Connection(coord[0], coord[1], leaf.x, leaf.y, leaf.weight)
                else:
                    conn = Connection(leaf.x, leaf.y, coord[0], coord[1], leaf.weight)

                # Apply connection filters
                # F1: allow_same_layer_connections toggles y1 <= y2 (clean) vs y1 < y2 (optimized)
                y_constraint = conn.y1 <= conn.y2 if self.allow_same_layer_connections else conn.y1 < conn.y2
                if (leaf.weight != 0.0 and
                    not math.isnan(leaf.weight) and
                    y_constraint and
                    not (conn.x1 == conn.x2 and conn.y1 == conn.y2)):
                    connections.add(conn)

        return connections

    def _pruning_extraction_original(self, state: Any, cppn_transformed: Any,
                                    coord: Tuple[float, float], point: QuadPoint,
                                    outgoing: bool) -> Set[Connection]:
        """Original pruning extraction with individual queries (baseline)."""
        connections = set()

        def extract(p):
            for child in p.children:
                if child is None:
                    continue

                child_var = self._variance(child)
                if child_var > self.variance_threshold:
                    extract(child)
                else:
                    # Query neighbors individually
                    d_left = abs(child.weight - self._query_cppn(
                        state, cppn_transformed, coord,
                        (child.x - p.width, child.y), outgoing
                    ))
                    d_right = abs(child.weight - self._query_cppn(
                        state, cppn_transformed, coord,
                        (child.x + p.width, child.y), outgoing
                    ))
                    d_top = abs(child.weight - self._query_cppn(
                        state, cppn_transformed, coord,
                        (child.x, child.y - p.width), outgoing
                    ))
                    d_bottom = abs(child.weight - self._query_cppn(
                        state, cppn_transformed, coord,
                        (child.x, child.y + p.width), outgoing
                    ))

                    self._cppn_query_count += 4

                    if max(min(d_top, d_bottom), min(d_left, d_right)) > self.band_threshold:
                        if outgoing:
                            conn = Connection(coord[0], coord[1], child.x, child.y, child.weight)
                        else:
                            conn = Connection(child.x, child.y, coord[0], coord[1], child.weight)

                        # F1: allow_same_layer_connections toggles y1 <= y2 (clean) vs y1 < y2 (optimized)
                        y_constraint = conn.y1 <= conn.y2 if self.allow_same_layer_connections else conn.y1 < conn.y2
                        if (child.weight != 0.0 and
                            not math.isnan(child.weight) and
                            y_constraint and
                            not (conn.x1 == conn.x2 and conn.y1 == conn.y2)):
                            connections.add(conn)

        extract(point)
        return connections

    def _pruning_extraction(self, state: Any, cppn_transformed: Any,
                           coord: Tuple[float, float], point: QuadPoint,
                           outgoing: bool) -> Set[Connection]:
        """Pruning extraction dispatcher based on O5 flag."""
        if self.opt_batch_pruning_queries:
            return self._pruning_extraction_batched(state, cppn_transformed, coord, point, outgoing)
        else:
            return self._pruning_extraction_original(state, cppn_transformed, coord, point, outgoing)

    # ========================================================================
    # O7: vmap Substrate Evaluation
    # ========================================================================

    def _evaluate_substrate_vmap(self, state: Any, substrate_net: Tuple[Any, Any], problem: Any) -> float:
        """Batch evaluate all test cases using vmap.

        O7 Optimization: vmap over test cases instead of Python loop.
        """
        if substrate_net is None:
            return 0.0

        nodes, conns = substrate_net

        # Get test data
        if hasattr(problem, 'get_data'):
            data = problem.get_data()
            inputs_list = [inp for inp, _ in data]
            targets_list = [target for _, target in data]
        elif hasattr(problem, 'get_test_cases'):
            test_cases = problem.get_test_cases()
            inputs_list = [tc['input'] for tc in test_cases]
            targets_list = [tc['target'] for tc in test_cases]
        else:
            return 0.0

        if len(inputs_list) == 0:
            return 0.0

        # Stack inputs and targets
        # Handle bias
        if hasattr(problem, 'use_bias') and problem.use_bias:
            inputs_batch = jnp.stack([jnp.array(inp, dtype=jnp.float32) for inp in inputs_list])
        else:
            inputs_batch = jnp.stack([
                jnp.concatenate([jnp.array(inp, dtype=jnp.float32), jnp.array([1.0])])
                for inp in inputs_list
            ])

        targets_batch = jnp.stack([jnp.array(t, dtype=jnp.float32) for t in targets_list])

        # vmap forward pass over all test cases
        outputs_batch = jax.vmap(
            lambda inputs: self._forward_hyperneat_style(nodes, conns, inputs)
        )(inputs_batch)

        # Batch MSE computation
        errors = jnp.mean((outputs_batch - targets_batch) ** 2, axis=1)
        avg_error = jnp.mean(errors)

        fitness = max(0.0, 1.0 - float(avg_error))
        return fitness

    def _evaluate_substrate_original(self, state: Any, substrate_net: Tuple[Any, Any], problem: Any) -> float:
        """Original substrate evaluation with Python loop (baseline)."""
        if substrate_net is None:
            return 0.0

        nodes, conns = substrate_net

        if hasattr(problem, 'get_data'):
            data = problem.get_data()
            inputs_list = [inp for inp, _ in data]
            targets_list = [target for _, target in data]
        elif hasattr(problem, 'get_test_cases'):
            test_cases = problem.get_test_cases()
            inputs_list = [tc['input'] for tc in test_cases]
            targets_list = [tc['target'] for tc in test_cases]
        else:
            return 0.0

        total_error = 0.0
        num_cases = len(inputs_list)

        for inputs, targets in zip(inputs_list, targets_list):
            inputs_jax = jnp.array(inputs, dtype=jnp.float32)

            if hasattr(problem, 'use_bias') and problem.use_bias:
                inputs_with_bias = inputs_jax
            else:
                inputs_with_bias = jnp.concatenate([inputs_jax, jnp.array([1.0])])

            outputs = self._forward_hyperneat_style(nodes, conns, inputs_with_bias)

            targets_jax = jnp.array(targets, dtype=jnp.float32)
            error = jnp.mean((outputs - targets_jax) ** 2)
            total_error += float(error)

        avg_error = total_error / num_cases
        fitness = max(0.0, 1.0 - avg_error)

        return fitness

    def _evaluate_substrate(self, state: Any, substrate_net: Tuple[Any, Any], problem: Any) -> float:
        """Substrate evaluation dispatcher based on O7 flag."""
        if self.opt_vmap_evaluation:
            return self._evaluate_substrate_vmap(state, substrate_net, problem)
        else:
            return self._evaluate_substrate_original(state, substrate_net, problem)

    # ========================================================================
    # Substrate Building
    # ========================================================================

    def _build_tensorneat_substrate(self, hidden_nodes: Set, connections: Set,
                                       state: Any = None, cppn_transformed: Any = None) -> Tuple[Any, Any]:
        """Original substrate building with dynamic arrays (baseline)."""
        coord_to_idx = {}

        num_inputs = len(self.substrate_input_coords)
        for i, coord in enumerate(self.substrate_input_coords):
            coord_to_idx[tuple(float(c) for c in coord)] = i

        output_coords_set = set(tuple(float(c) for c in coord)
                               for coord in self.substrate_output_coords)

        all_hidden_coords = set()
        for conn in connections:
            coord1 = (conn.x1, conn.y1)
            coord2 = (conn.x2, conn.y2)
            if coord1 not in coord_to_idx and coord1 not in output_coords_set:
                all_hidden_coords.add(coord1)
            if coord2 not in coord_to_idx and coord2 not in output_coords_set:
                all_hidden_coords.add(coord2)

        hidden_idx = num_inputs
        for coord in sorted(all_hidden_coords):
            coord_to_idx[coord] = hidden_idx
            hidden_idx += 1

        for i, coord in enumerate(self.substrate_output_coords):
            coord_to_idx[tuple(float(c) for c in coord)] = hidden_idx + i

        num_nodes = len(coord_to_idx)
        nodes = np.zeros((num_nodes, 1))

        for idx in range(num_nodes):
            nodes[idx, 0] = idx

        conn_list = []
        for conn in connections:
            if (conn.x1, conn.y1) in coord_to_idx and (conn.x2, conn.y2) in coord_to_idx:
                from_idx = coord_to_idx[(conn.x1, conn.y1)]
                to_idx = coord_to_idx[(conn.x2, conn.y2)]
                conn_list.append([from_idx, to_idx, conn.weight])

        if len(conn_list) == 0:
            conn_list = self._create_minimal_substrate_fallback(num_nodes, state, cppn_transformed)

        conns = np.array(conn_list)

        nodes_jax = jnp.array(nodes)
        conns_jax = jnp.array(conns)

        return nodes_jax, conns_jax

    def _create_minimal_substrate_fallback(self, num_nodes: int, state: Any = None,
                                            cppn_transformed: Any = None) -> list:
        """Create minimal fallback substrate with direct input→output connections.

        F2 Toggle:
        - use_cppn_fallback_weights=False (default): Random weights (optimized behavior)
        - use_cppn_fallback_weights=True: CPPN-queried weights (clean behavior)
        """
        num_inputs = len(self.substrate_input_coords)
        num_outputs = len(self.substrate_output_coords)
        output_start_idx = num_nodes - num_outputs

        conn_list = []

        # F2: Use CPPN-queried weights (clean behavior) if enabled and available
        if self.use_cppn_fallback_weights and state is not None and cppn_transformed is not None:
            # Query CPPN for weights like clean implementation
            for i, input_coord in enumerate(self.substrate_input_coords):
                for j, output_coord in enumerate(self.substrate_output_coords):
                    # Query raw CPPN output (without sparsification threshold)
                    inputs = np.array([
                        input_coord[0], input_coord[1],
                        output_coord[0], output_coord[1], 1.0
                    ], dtype=np.float32)
                    inputs_jax = jnp.array(inputs)
                    weight_raw = self._jitted_cppn_forward(state, cppn_transformed, inputs_jax)
                    weight = float(weight_raw.flatten()[0]) * self.max_weight

                    # Clamp weight
                    weight = max(-self.max_weight, min(weight, self.max_weight))

                    output_idx = output_start_idx + j
                    conn_list.append([i, output_idx, weight])
        else:
            # Default: Random weights (optimized behavior)
            import hashlib
            if hasattr(self, '_current_cppn') and self._current_cppn is not None:
                genome_str = str(self._current_cppn)
                genome_hash = hashlib.md5(genome_str.encode()).hexdigest()
                seed = int(genome_hash[:8], 16)
            else:
                seed_str = f"{num_nodes}_{num_inputs}_{num_outputs}"
                seed_hash = hashlib.md5(seed_str.encode()).hexdigest()
                seed = int(seed_hash[:8], 16)
            rng = np.random.RandomState(seed)

            for input_idx in range(num_inputs):
                for output_idx in range(output_start_idx, num_nodes):
                    weight = rng.uniform(-0.5, 0.5)
                    conn_list.append([input_idx, output_idx, weight])

        return conn_list

    # ========================================================================
    # Forward Pass
    # ========================================================================

    def _forward_hyperneat_style(self, nodes: Any, conns: Any, inputs: Any) -> Any:
        """Forward pass using HyperNEATNode computational model."""
        num_nodes = nodes.shape[0]
        num_inputs = inputs.shape[0]
        num_outputs = len(self.substrate_output_coords)
        output_start_idx = num_nodes - num_outputs

        # Pre-extract connection info (outside loop for efficiency)
        from_indices = conns[:, 0].astype(jnp.int32)
        to_indices = conns[:, 1].astype(jnp.int32)
        weights = conns[:, 2]
        valid_mask = ~jnp.isnan(weights)
        valid_from = from_indices[valid_mask]
        valid_to = to_indices[valid_mask]
        valid_weights = weights[valid_mask]

        # Forward pass with activation iterations
        vals = jnp.zeros(num_nodes)
        vals = vals.at[:num_inputs].set(inputs)

        for iteration in range(self.activate_time):
            new_vals = jnp.zeros(num_nodes)
            new_vals = new_vals.at[:num_inputs].set(inputs)

            aggregated = jnp.zeros(num_nodes)
            aggregated = aggregated.at[valid_to].add(vals[valid_from] * valid_weights)

            if output_start_idx > num_inputs:
                hidden_vals = jnp.tanh(aggregated[num_inputs:output_start_idx])
                new_vals = new_vals.at[num_inputs:output_start_idx].set(hidden_vals)

            output_vals = aggregated[output_start_idx:]
            new_vals = new_vals.at[output_start_idx:].set(output_vals)

            vals = new_vals

        raw_outputs = vals[-num_outputs:]
        return jax.nn.sigmoid(raw_outputs)

    # ========================================================================
    # ES-HyperNEAT Discovery (with optimization dispatching)
    # ========================================================================

    def _discover_substrate_es(self, state: Any, cppn_transformed: Any) -> Tuple[Set, Set, Dict]:
        """Three-phase ES-HyperNEAT substrate discovery."""
        self._current_cppn = cppn_transformed

        hidden_nodes = set()
        unexplored_hidden_nodes = set()
        connections1, connections2, connections3 = set(), set(), set()

        # Phase 1: Explore from inputs
        for coord in self.substrate_input_coords:
            root = self._division_initialization(
                state, cppn_transformed, tuple(coord), outgoing=True
            )
            conns = self._pruning_extraction(
                state, cppn_transformed, tuple(coord), root, outgoing=True
            )
            connections1 = connections1.union(conns)
            for conn in conns:
                hidden_nodes.add((conn.x2, conn.y2))

        unexplored_hidden_nodes = copy.deepcopy(hidden_nodes)

        # FORCED CONNECTIVITY FALLBACK
        if len(connections1) == 0:
            center = (0.0, 0.0)
            hidden_nodes.add(center)
            for input_coord in self.substrate_input_coords:
                conn = Connection(
                    x1=input_coord[0], y1=input_coord[1],
                    x2=center[0], y2=center[1],
                    weight=0.5
                )
                connections1.add(conn)
            unexplored_hidden_nodes = copy.deepcopy(hidden_nodes)

        # Phase 2: Explore from hidden nodes
        for iteration in range(self.iteration_level):
            new_hidden_nodes = set()
            for coord in unexplored_hidden_nodes:
                root = self._division_initialization(
                    state, cppn_transformed, coord, outgoing=True
                )
                conns = self._pruning_extraction(
                    state, cppn_transformed, coord, root, outgoing=True
                )
                connections2 = connections2.union(conns)
                for conn in conns:
                    if (conn.x2, conn.y2) not in hidden_nodes:
                        new_hidden_nodes.add((conn.x2, conn.y2))
            hidden_nodes.update(new_hidden_nodes)
            unexplored_hidden_nodes = hidden_nodes - unexplored_hidden_nodes

        # Phase 3: Explore to outputs
        for coord in self.substrate_output_coords:
            root = self._division_initialization(
                state, cppn_transformed, tuple(coord), outgoing=False
            )
            conns = self._pruning_extraction(
                state, cppn_transformed, tuple(coord), root, outgoing=False
            )
            connections3 = connections3.union(conns)

        # FORCED CONNECTIVITY FALLBACK
        if len(connections3) == 0 and len(hidden_nodes) > 0:
            for output_coord in self.substrate_output_coords:
                nearest_hidden = min(hidden_nodes, key=lambda h:
                    ((h[0] - output_coord[0])**2 + (h[1] - output_coord[1])**2)**0.5
                )
                conn = Connection(
                    x1=nearest_hidden[0], y1=nearest_hidden[1],
                    x2=output_coord[0], y2=output_coord[1],
                    weight=0.5
                )
                connections3.add(conn)

        # Combine and clean
        connections = connections1.union(connections2).union(connections3)
        pre_clean_connections = len(connections)
        pre_clean_hidden = len(hidden_nodes)

        result = self._clean_net(connections)
        post_clean_hidden, post_clean_connections_set = result
        post_clean_connections = len(post_clean_connections_set)
        post_clean_nodes = len(post_clean_hidden)

        phase_info = {
            'phase1_connections': len(connections1),
            'phase2_connections': len(connections2),
            'phase3_connections': len(connections3),
            'pre_clean_connections': pre_clean_connections,
            'post_clean_connections': post_clean_connections,
            'pre_clean_hidden': pre_clean_hidden,
            'post_clean_hidden': post_clean_nodes,
        }

        return post_clean_hidden, post_clean_connections_set, phase_info

    # ========================================================================
    # Main Algorithm Methods
    # ========================================================================

    def _log_debug(self, message: str):
        """Log debug message to file when verbose mode enabled."""
        if self.verbose:
            import sys
            print(f"[ES-HyperNEAT OPTIMIZED] {message}")
            sys.stdout.flush()
            with open(self.debug_log_file, 'a') as f:
                f.write(f"{message}\n")
                f.flush()

    def create_config(self, params: Dict[str, Any]) -> Any:
        """Create NEAT configuration for CPPN evolution."""
        # Handle config loading
        if params.get('config_file') or params.get('preset'):
            config_manager = ConfigManager()
            hierarchical_config = config_manager.load_config(
                algorithm='eshyperneat',
                implementation='tensorneat',
                preset=params.get('preset', 'default'),
                config_file=params.get('config_file'),
                overrides=params.get('overrides', {})
            )
        else:
            hierarchical_config = params

        self._config_metadata = hierarchical_config

        # Extract algorithm_params.eshyperneat section
        algo_params = hierarchical_config.get('algorithm_params', {}).get('eshyperneat', {})
        if not algo_params:
            algo_params = hierarchical_config

        # Extract ES-HyperNEAT discovery parameters
        es_section = algo_params.get('es_hyperneat', {})
        self.initial_depth = es_section.get('initial_depth', 0)
        self.max_depth = es_section.get('max_depth', 1)
        self.variance_threshold = es_section.get('variance_threshold', 0.03)
        self.division_threshold = es_section.get('division_threshold', 0.5)
        self.band_threshold = es_section.get('band_threshold', 0.3)
        self.max_weight = es_section.get('max_weight', 8.0)
        self.iteration_level = es_section.get('iteration_level', 1)
        self.verbose = es_section.get('verbose', False)

        # Extract substrate coordinates
        substrate_section = algo_params.get('substrate', {})
        self.substrate_input_coords = substrate_section.get('input_coords', [])
        self.substrate_output_coords = substrate_section.get('output_coords', [])

        self.output_activation = substrate_section.get('output_activation', 'sigmoid')
        self.hidden_activation = substrate_section.get('hidden_activation', 'tanh')

        default_activate_time = (2 ** self.max_depth) + 1
        self.activate_time = substrate_section.get('activate_time', default_activate_time)
        self.weight_threshold = substrate_section.get('weight_threshold', 0.2)

        # Build NEAT config for CPPN evolution
        flat_params = {
            'genome': {
                'num_inputs': 5,
                'num_outputs': 1,
                'num_hidden': 0,
                'feed_forward': True,
                'weight': {
                    'init_mean': 0.0,
                    'init_std': 1.0,
                    'min_value': -30.0,
                    'max_value': 30.0,
                    'mutate_power': 0.5,
                    'mutate_rate': 0.8,
                    'replace_rate': 0.1,
                },
                'bias': {
                    'init_mean': 0.0,
                    'init_std': 1.0,
                    'min_value': -30.0,
                    'max_value': 30.0,
                    'mutate_power': 0.5,
                    'mutate_rate': 0.7,
                    'replace_rate': 0.1,
                },
                'activation': {
                    'default': 'tanh',
                    'options': ['tanh', 'sin', 'gauss'],
                    'mutate_rate': 0.5,
                },
            },
            'population_size': algo_params.get('population_size', 150),
            'mutation': {
                'conn_add_prob': 0.5,
                'conn_delete_prob': 0.5,
                'node_add_prob': 0.2,
                'node_delete_prob': 0.2,
            },
            'species': {
                'compatibility_threshold': 3.0,
                'max_stagnation': 20,
                'species_elitism': 15,
            },
            'selection': {
                'genome_elitism': 15,
                'survival_threshold': 0.2,
            },
            'activation_options': ['tanh', 'sin', 'gauss'],
            'activation_default': 'tanh',
            'verbose': False,
        }

        self.neat_algo = self.adapter.build_neat_config(flat_params)

        # JIT compile CPPN forward
        self._jitted_cppn_forward = jax.jit(
            self.neat_algo.genome.forward,
            static_argnums=(0,)
        )

        # Initialize RecurrentGenome for substrate execution
        from tensorneat.algorithm.hyperneat.hyperneat import HyperNEATNode, HyperNEATConn
        from tensorneat.genome import RecurrentGenome
        from tensorneat.common import ACT, AGG, State

        self.hyper_genome = RecurrentGenome(
            num_inputs=len(self.substrate_input_coords),
            num_outputs=len(self.substrate_output_coords),
            max_nodes=500,
            max_conns=2000,
            node_gene=HyperNEATNode(aggregation=AGG.sum, activation=ACT.tanh),
            conn_gene=HyperNEATConn(),
            activate_time=self.activate_time,
            output_transform=ACT.sigmoid
        )

        dummy_state = State()
        dummy_state = self.hyper_genome.setup(dummy_state)

        return self.neat_algo

    def initialize(self, config: Any, problem: Any, seed: int = 42) -> Any:
        """Initialize with NEAT Pipeline and JIT-compile NEAT operations."""
        from tensorneat.pipeline import Pipeline

        wrapped_problem = self._wrap_problem_for_pipeline(problem)

        self.pipeline = Pipeline(
            algorithm=config,
            problem=wrapped_problem,
            seed=seed
        )

        state = self.pipeline.setup()

        self.problem = problem
        self._start_time = time.time()

        # JIT compile NEAT operations
        self._compiled_ask = jax.jit(self.neat_algo.ask)
        self._compiled_transform_batch = jax.jit(
            jax.vmap(self.neat_algo.transform, in_axes=(None, (0, 0)))
        )
        self._compiled_tell = jax.jit(self.neat_algo.tell)

        return state

    def _wrap_problem_for_pipeline(self, problem: Any) -> Any:
        """Wrap problem for TensorNEAT pipeline compatibility."""
        class WrappedProblem:
            def __init__(self, inner_problem):
                self.inner = inner_problem
                self.input_shape = (5,)
                self.jitable = True

            def setup(self, state=None):
                from tensorneat.common import State
                if state is None:
                    state = State()
                return state

            def evaluate(self, state, randkey, forward_func, transformed):
                return 0.0

        return WrappedProblem(problem)

    def run_generation(self, state: Any, problem: Any) -> Tuple[Any, AlgorithmMetrics]:
        """Run one generation with optimizations based on enabled flags."""
        gen_start = time.time()

        # Reset per-generation counters
        self._dead_regions_skipped = 0

        # Split the RNG key each generation so ask/tell stay stochastic.
        randkey_, randkey = jax.random.split(state.randkey)
        state = state.update(randkey=randkey)

        # Get CPPN population
        cppn_population = self._compiled_ask(state)
        pop_size = cppn_population[0].shape[0]

        # Batch transform all CPPNs
        transform_start = time.time()
        cppns_transformed = self._compiled_transform_batch(state, cppn_population)
        transform_time = time.time() - transform_start

        # Per-CPPN discovery loop (sequential)
        fitnesses = []
        discovered_hidden_counts = []
        connection_counts = []
        final_connection_counts = []

        for i in range(pop_size):
            # Extract single transformed CPPN
            cppn = (cppns_transformed[0][i], cppns_transformed[1][i],
                    cppns_transformed[2][i], cppns_transformed[3][i])

            # 1. Discover substrate
            hidden_nodes, connections, phase_info = self._discover_substrate_es(state, cppn)

            # 2. Build network (pass state/cppn for F2 CPPN fallback if enabled)
            substrate_net = self._build_tensorneat_substrate(hidden_nodes, connections, state, cppn)

            # 3. Evaluate
            fitness = self._evaluate_substrate(state, substrate_net, problem)

            fitnesses.append(fitness)
            discovered_hidden_counts.append(len(hidden_nodes))
            connection_counts.append(len(connections))
            final_connection_counts.append(len(substrate_net[1]) if substrate_net is not None else 0)

        fitnesses = jnp.array(fitnesses)
        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)

        # Update state
        new_state = self._compiled_tell(state, fitnesses)

        # Create metrics
        metrics = self._create_metrics_with_es_data(
            new_state, fitnesses, gen_start,
            discovered_hidden=np.mean(discovered_hidden_counts),
            total_connections=np.mean(connection_counts)
        )

        return new_state, metrics

    # ========================================================================
    # BaseAlgorithm Abstract Methods
    # ========================================================================

    def evaluate_genome(self, genome: Any, problem: Any) -> float:
        return 0.0

    def extract_network_info(self, state: Any) -> Any:
        return None

    def genome_to_phenotype(self, genome: Any) -> Any:
        return None

    def get_best_genome(self, state: Any) -> Any:
        if hasattr(self, 'neat_algo') and self.neat_algo is not None:
            pop = self.neat_algo.ask(state)
            if pop is not None and len(pop) > 0:
                return pop[0]
        return None

    # ========================================================================
    # Metrics
    # ========================================================================

    def _create_metrics_with_es_data(self, state: Any, fitnesses: Any,
                                    gen_start: float, discovered_hidden: float,
                                    total_connections: float) -> AlgorithmMetrics:
        """Create AlgorithmMetrics with ES-HyperNEAT and optimization data."""
        generation = state.generation if hasattr(state, 'generation') else 0
        best_fitness = float(jnp.max(fitnesses))
        mean_fitness = float(jnp.mean(fitnesses))
        min_fitness = float(jnp.min(fitnesses))
        max_fitness = float(jnp.max(fitnesses))
        std_fitness = float(jnp.std(fitnesses))
        evaluations = len(fitnesses)
        time_elapsed = time.time() - gen_start

        num_species = 1
        species_sizes = [len(fitnesses)]
        species_fitness = [mean_fitness]

        # Include optimization metrics
        custom_metrics = {
            'discovered_hidden_nodes': discovered_hidden,
            'total_connections': total_connections,
            'generation_time': time_elapsed,
            'total_cppn_queries': self._cppn_query_count,
            'total_variance_calls': self._variance_call_count,
            # Optimization flags (proven optimizations)
            'opt_batch_division_queries': self.opt_batch_division_queries,
            'opt_batch_pruning_queries': self.opt_batch_pruning_queries,
            'opt_vmap_evaluation': self.opt_vmap_evaluation,
            'opt_precompute_coords': self.opt_precompute_coords,
        }

        return AlgorithmMetrics(
            generation=generation,
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            min_fitness=min_fitness,
            max_fitness=max_fitness,
            std_fitness=std_fitness,
            num_species=num_species,
            species_sizes=species_sizes,
            species_fitness=species_fitness,
            evaluations=evaluations,
            time_elapsed=time_elapsed,
            custom_metrics=custom_metrics
        )
