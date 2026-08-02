"""ES-HyperNEAT with HSHG substrate discovery.

Replaces quadtree-based division_initialization and pruning_extraction with
HSHG (Hierarchical Spatial Hash Grid) equivalents. Everything else (NEAT
evolution, substrate construction, fitness evaluation) is identical.

This enables direct comparison of quadtree vs HSHG substrate discovery
through the full evolutionary pipeline.
"""

import copy
import numpy as np
import jax
import jax.numpy as jnp
from typing import Set, Tuple

from .eshyperneat import ESHyperNEAT

try:
    from .substrate import Connection
except ImportError:
    from substrate import Connection

# HSHG substrate-discovery primitives (local hshg_core package).
from .hshg_core import (
    HSHGConfig,
    division_initialization_hshg,
    pruning_extraction_hshg,
    Connection as HSHGConnection,
)


class ESHyperNEATHSHG(ESHyperNEAT):
    """ES-HyperNEAT using HSHG instead of quadtree for substrate discovery.

    Overrides es_hyperneat() to use HSHG-based division and pruning.
    All other behavior (NEAT, substrate construction, evaluation) is inherited.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hshg_config = None
        # JIT-compile CPPN forward for HSHG batch queries (critical for performance)
        self._jit_cppn_forward = jax.jit(self.neat.forward)

    def _get_hshg_config(self):
        """Lazily create HSHG config matching current ES-HyperNEAT params."""
        if self._hshg_config is None:
            self._hshg_config = HSHGConfig.for_es_hyperneat(
                max_depth=self.max_depth
            )
        return self._hshg_config

    def _make_cppn_forward_batch(self, state, cppn_transformed):
        """Create a batched CPPN forward function for HSHG division.

        HSHG's batch_query_cppn_all_pairs calls:
            cppn_forward_fn(cppn_state, inputs_batch) -> outputs_batch
        where inputs_batch is shape [N, 4] (x1, y1, x2, y2).

        We add bias=1.0 (5th input) and use self.neat.forward.
        """
        def cppn_fn(cppn_state_unused, inputs_batch):
            # inputs_batch: [N, 4] -> need to add bias column -> [N, 5]
            n = inputs_batch.shape[0]
            bias = jnp.ones((n, 1), dtype=jnp.float32)
            inputs_with_bias = jnp.concatenate([inputs_batch, bias], axis=1)

            # Query CPPN for each input (cannot easily vmap due to variable topology)
            results = []
            for i in range(n):
                w = self._jit_cppn_forward(state, cppn_transformed, inputs_with_bias[i])
                if hasattr(w, 'ndim') and w.ndim > 0:
                    results.append(float(w.flatten()[0]))
                else:
                    results.append(float(w))
            return jnp.array(results, dtype=jnp.float32)
        return cppn_fn

    def _hshg_division_and_pruning(self, state, cppn_transformed, coord, outgoing):
        """Run HSHG division + pruning for a single source coordinate.

        Returns a set of Connection objects (base class type).
        """
        config = self._get_hshg_config()
        cppn_fn = self._make_cppn_forward_batch(state, cppn_transformed)

        # HSHG division: discover nodes
        div_result = division_initialization_hshg(
            cppn_forward_fn=cppn_fn,
            cppn_state=None,  # Not used; cppn_fn captures state
            source_coord=coord,
            outgoing=outgoing,
            max_depth=self.max_depth,
            initial_depth=self.initial_depth,
            division_threshold=self.division_threshold,
            variance_threshold=self.variance_threshold,
            max_weight=self.max_weight,
            config=config,
        )

        if not div_result.discovered_nodes:
            return set()

        # HSHG pruning: extract connections
        pruning_result = pruning_extraction_hshg(
            nodes=div_result.discovered_nodes,
            hshg_state=div_result.hshg_state,
            config=config,
            source_coord=coord,
            outgoing=outgoing,
            band_threshold=self.band_threshold,
            variance_threshold=self.variance_threshold,
            allow_same_layer_connections=False,  # Match PUREPLES: y1 < y2
        )

        # Convert HSHG Connection (NamedTuple) to base class Connection
        # CRITICAL: Convert JAX arrays to Python floats for hashability
        connections = set()
        for hc in pruning_result.connections:
            x1, y1 = float(hc.x1), float(hc.y1)
            x2, y2 = float(hc.x2), float(hc.y2)
            w = float(hc.weight)
            conn = Connection(x1, y1, x2, y2, w)
            # Apply same validation as quadtree pruning
            if conn.y1 < conn.y2 and not (conn.x1 == conn.x2 and conn.y1 == conn.y2):
                connections.add(conn)

        return connections

    def es_hyperneat(self, state, cppn_transformed):
        """Explores hidden nodes using HSHG instead of quadtree.

        Same algorithm structure as base class but uses HSHG for
        division and pruning at each step.
        """
        inputs = self.substrate.input_coordinates
        outputs = self.substrate.output_coordinates
        hidden_nodes = set()
        unexplored_hidden_nodes = set()
        connections1, connections2, connections3 = set(), set(), set()

        # Explore from inputs
        for coord in inputs:
            conns = self._hshg_division_and_pruning(
                state, cppn_transformed, tuple(coord), True
            )
            connections1 = connections1.union(conns)
            for conn in connections1:
                hidden_nodes.add((conn.x2, conn.y2))

        unexplored_hidden_nodes = copy.deepcopy(hidden_nodes)

        # Explore from hidden nodes
        for iteration in range(self.iteration_level):
            new_hidden_nodes = set()

            for coord in unexplored_hidden_nodes:
                conns = self._hshg_division_and_pruning(
                    state, cppn_transformed, coord, True
                )
                connections2 = connections2.union(conns)

                for conn in conns:
                    if (conn.x2, conn.y2) not in hidden_nodes:
                        new_hidden_nodes.add((conn.x2, conn.y2))

            hidden_nodes.update(new_hidden_nodes)
            unexplored_hidden_nodes = hidden_nodes - unexplored_hidden_nodes

        # Explore to outputs
        for coord in outputs:
            conns = self._hshg_division_and_pruning(
                state, cppn_transformed, tuple(coord), False
            )
            connections3 = connections3.union(conns)

        # Combine all connections
        connections = connections1.union(connections2).union(connections3)

        return self.clean_net(connections)
