"""ES-HyperNEAT algorithm implementation.

This is a faithful implementation of the ES-HyperNEAT algorithm following
the PUREPLES reference implementation. This vanilla version maintains
algorithmic correctness without performance optimizations.

For an optimized version with caching, batching, and other performance
enhancements, see eshyperneat_optimized.py.

Key algorithmic features matching PUREPLES:
- Band threshold default of 0.2
- Upward-only connections (y1 < y2)
- Set subtraction for unexplored nodes iteration
- No configurable connection directionality
"""

import time
import copy
import math
import numpy as np
from functools import partial
from typing import Tuple, List, Set
import jax
import jax.numpy as jnp

from tensorneat.algorithm import BaseAlgorithm, NEAT
from tensorneat.genome import RecurrentGenome
from tensorneat.genome.gene import BaseNode, BaseConn
from tensorneat.genome.utils import extract_gene_attrs, unflatten_conns
from tensorneat.common import State, ACT, AGG, I_INF

try:
    from .substrate import QuadPoint, Connection
except ImportError:
    from substrate import QuadPoint, Connection


class ESHyperNEATNode(BaseNode):
    """Custom node gene for ES-HyperNEAT substrate networks."""
    
    custom_attrs = ["bias", "response"]

    def __init__(self, aggregation=AGG.sum, activation=ACT.sigmoid):
        super().__init__()
        self.aggregation = aggregation
        self.activation = activation

    def new_identity_attrs(self, state):
        """Identity attributes for pass-through behavior."""
        return jnp.array([0.0, 1.0])  # bias=0, response=1

    def new_random_attrs(self, state, randkey):
        """Random initialization (not typically used in ES-HyperNEAT)."""
        k1, k2 = jax.random.split(randkey)
        bias = jax.random.normal(k1) * 0.1
        response = jax.random.normal(k2) * 0.1 + 1.0
        return jnp.array([bias, response])

    def mutate(self, state, randkey, attrs):
        """Mutation (not typically used in ES-HyperNEAT)."""
        return attrs

    def distance(self, state, attrs1, attrs2):
        """Distance metric between node attributes."""
        return jnp.abs(attrs1 - attrs2).sum()

    def forward(self, state, attrs, inputs, is_output_node=False):
        """Forward pass through node."""
        # Sum inputs
        z = self.aggregation(inputs)
        
        # Apply bias and response
        bias, response = attrs
        z = bias + response * z
        
        # Apply activation unless output node (using JAX-compatible conditional)
        z = jnp.where(
            is_output_node,
            z,  # No activation for output nodes
            self.activation(z)  # Apply activation for hidden nodes
        )
            
        return z


class ESHyperNEATConn(BaseConn):
    """Custom connection gene for ES-HyperNEAT substrate networks."""
    
    custom_attrs = ['weight']

    def __init__(self, max_weight=5.0):
        super().__init__()
        self.max_weight = max_weight

    def new_zero_attrs(self, state):
        """Zero weight attributes."""
        return jnp.array([0.0])

    def new_identity_attrs(self, state):
        """Identity weight attributes."""
        return jnp.array([1.0])

    def new_random_attrs(self, state, randkey):
        """Random weight initialization."""
        weight = jax.random.uniform(randkey, minval=-self.max_weight, maxval=self.max_weight)
        return jnp.array([weight])

    def mutate(self, state, randkey, attrs):
        """Mutation (not typically used in ES-HyperNEAT)."""
        return attrs

    def distance(self, state, attrs1, attrs2):
        """Distance metric between connection attributes."""
        return jnp.abs(attrs1[0] - attrs2[0])

    def forward(self, state, attrs, input_value):
        """Forward pass through connection."""
        weight = attrs[0]
        return input_value * weight


class BaseESHyperNEAT(BaseAlgorithm):
    """Base class for ES-HyperNEAT with common evolution methods."""
    
    def auto_run(self, state):
        """Run evolution with proper shape handling."""
        print("\nStarting evolution process...")
        tic = time.time()
        generation = 0
        best_fitness = float('-inf')
        best_genome = None

        while True:
            # Get current population
            pop = self.ask(state)
            
            # Transform population
            transformed_pop = jax.vmap(
                lambda x: self.transform(state, (x[0], x[1])),
                in_axes=0
            )(pop)
            
            # Evaluate population
            k1, k2 = jax.random.split(state.randkey)
            eval_keys = jax.random.split(k1, self.pop_size)
            state = state.update(randkey=k2)
            
            # Compute fitness
            fitnesses = jax.vmap(self.neat.forward, in_axes=(None, 0, None))(
                state, transformed_pop, jnp.zeros(self.substrate.num_inputs)
            )
            
            # Update state
            state = self.tell(state, fitnesses)
            
            # Track best solution
            max_fitness = jnp.max(fitnesses)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_idx = jnp.argmax(fitnesses)
                best_genome = jax.tree_map(lambda x: x[best_idx], pop)
            
            # Print progress
            generation += 1
            if generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {best_fitness:.6f}")
            
            # Check termination criteria
            if generation >= 1000 or best_fitness >= 0.95:
                break
        
        print(f"\nEvolution completed after {generation} generations")
        print(f"Total time: {time.time() - tic:.2f} seconds")
        print(f"Best fitness achieved: {best_fitness:.6f}")
        
        return state, best_genome


class ESHyperNEAT(BaseESHyperNEAT):
    """ES-HyperNEAT algorithm implementation with quadtree-based topology discovery.
    
    This implementation follows the PUREPLES reference implementation faithfully,
    maintaining algorithmic correctness without performance optimizations.
    """
    
    def __init__(self, substrate, neat, max_weight=5.0, band_threshold=0.2,
                 initial_depth=0, max_depth=1, variance_threshold=0.03,
                 division_threshold=0.5, iteration_level=1,
                 aggregation=AGG.sum, activation=ACT.sigmoid, activate_time=1,
                 output_transform=ACT.identity):
        super().__init__()
        
        self.substrate = substrate
        self.neat = neat
        self.max_weight = max_weight
        self.band_threshold = band_threshold
        
        # ES-HyperNEAT specific parameters
        self.initial_depth = initial_depth
        self.max_depth = max_depth
        self.variance_threshold = variance_threshold
        self.division_threshold = division_threshold
        self.iteration_level = iteration_level
        self.activation_func = activation
        
        # Pre-compute static shapes
        self.max_nodes = self.substrate.nodes_cnt
        self.max_conns = self.substrate.conns_cnt
        
        print(f"Initialized with max_nodes: {self.max_nodes}, max_conns: {self.max_conns}")
        
        # Initialize genome
        self.hyper_genome = RecurrentGenome(
            num_inputs=self.substrate.num_inputs,
            num_outputs=self.substrate.num_outputs,
            max_nodes=self.max_nodes,
            max_conns=self.max_conns,
            node_gene=ESHyperNEATNode(aggregation, activation),
            conn_gene=ESHyperNEATConn(max_weight=self.max_weight),
            output_transform=output_transform,
            activate_time=activate_time
        )
        
        self.pop_size = neat.pop_size

    def setup(self, state=State()):
        """Setup initial state."""
        state = self.neat.setup(state)
        state = self.hyper_genome.setup(state)
        return state
    
    def ask(self, state):
        """Get current population."""
        return self.neat.ask(state)
    
    def tell(self, state, fitness):
        """Update state with fitness results."""
        return self.neat.tell(state, fitness)

    def transform(self, state, individual):
        """Transform CPPN genome into substrate network using ES-HyperNEAT discovery."""
        cppn_nodes, cppn_conns = individual
        
        # Transform CPPN genome first
        cppn_transformed = self.neat.transform(state, (cppn_nodes, cppn_conns))
        
        # Run ES-HyperNEAT discovery
        hidden_nodes, connections = self.es_hyperneat(state, cppn_transformed)
        
        # Store connections for test access
        self.connections = connections
        
        # Store discovery metrics for later access
        self.last_discovery_metrics = {
            'discovered_nodes': len(hidden_nodes),
            'total_connections': len(connections),
            'substrate_nodes': len(self.substrate.input_coordinates) + len(hidden_nodes) + len(self.substrate.output_coordinates)
        }
        
        # Convert discovered topology to TensorNEAT format
        return self._convert_to_substrate(state, hidden_nodes, connections)
    
    def _convert_to_substrate(self, state, hidden_nodes, connections):
        """Convert discovered topology to TensorNEAT substrate format."""
        # Create coordinate to index mapping
        coord_to_idx = {}
        
        # Add input nodes
        for i, coord in enumerate(self.substrate.input_coordinates):
            coord_to_idx[tuple(float(c) for c in coord)] = i
        
        # Add hidden nodes
        hidden_idx = len(self.substrate.input_coordinates)
        for coord in sorted(hidden_nodes):
            coord_to_idx[coord] = hidden_idx
            hidden_idx += 1
        
        # Add output nodes
        for i, coord in enumerate(self.substrate.output_coordinates):
            coord_to_idx[tuple(float(c) for c in coord)] = hidden_idx + i
        
        # Create node array
        num_nodes = len(coord_to_idx)
        nodes = np.zeros((num_nodes, 3))
        for coord, idx in coord_to_idx.items():
            nodes[idx] = [idx, coord[0], coord[1]]
        
        # Create connection array
        conn_list = []
        for conn in connections:
            if (conn.x1, conn.y1) in coord_to_idx and (conn.x2, conn.y2) in coord_to_idx:
                from_idx = coord_to_idx[(conn.x1, conn.y1)]
                to_idx = coord_to_idx[(conn.x2, conn.y2)]
                conn_list.append([from_idx, to_idx, conn.weight])
        
        if len(conn_list) == 0:
            # No connections discovered - create minimal network
            conn_list = [[0, len(coord_to_idx)-1, 0.0]]
        
        conns = np.array(conn_list)
        
        # Convert to JAX arrays and transform
        h_nodes = jnp.array(nodes)
        h_conns = jnp.array(conns)
        
        # Update genome with discovered topology
        self.hyper_genome.max_nodes = num_nodes
        self.hyper_genome.max_conns = len(conns)
        
        return self.hyper_genome.transform(state, h_nodes, h_conns)


    def forward(self, state, transformed, inputs):
        """Forward pass through the network."""
        return self.hyper_genome.forward(state, transformed, inputs)

    @property
    def num_inputs(self):
        return self.substrate.num_inputs

    @property
    def num_outputs(self):
        return self.substrate.num_outputs

    def show_details(self, state, fitness):
        """Show algorithm details."""
        self.neat.show_details(state, fitness)
    
    
    
    def query_cppn(self, state, cppn_transformed, coord1, coord2, outgoing):
        """
        Get the weight from one point to another using the CPPN.
        Takes into consideration which point is source/target.
        """
        if outgoing:
            inputs = np.array([coord1[0], coord1[1], coord2[0], coord2[1], 1.0], dtype=np.float32)
        else:
            inputs = np.array([coord2[0], coord2[1], coord1[0], coord1[1], 1.0], dtype=np.float32)
        
        # Query CPPN
        inputs_jax = jnp.array(inputs)
        weight_jax = self.neat.forward(state, cppn_transformed, inputs_jax)
        
        # Convert to scalar for Python control flow
        # Handle both scalar and array outputs
        if hasattr(weight_jax, 'ndim'):
            # JAX array
            if weight_jax.ndim > 0:
                # Flatten the array and take first element
                weight = float(weight_jax.flatten()[0])
            else:
                weight = float(weight_jax)
        else:
            # Already a Python float/scalar
            weight = float(weight_jax)
        
        # Handle NaN and Inf
        if math.isnan(weight) or math.isinf(weight):
            return 0.0
        
        # Apply band threshold
        if abs(weight) > self.band_threshold:
            if weight > 0:
                weight = (weight - self.band_threshold) / (1.0 - self.band_threshold)
            else:
                weight = (weight + self.band_threshold) / (1.0 - self.band_threshold)
            # Clamp to max weight
            return max(-self.max_weight, min(weight * self.max_weight, self.max_weight))
        else:
            return 0.0
    
    @staticmethod
    def get_weights(quad_point):
        """Recursively collect all weights for a given QuadPoint."""
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
    
    def variance(self, quad_point):
        """Find the variance of a given QuadPoint."""
        if not quad_point:
            return 0.0
        weights = self.get_weights(quad_point)
        if len(weights) == 0:
            return 0.0
        return np.var(weights)
    
    def division_initialization(self, state, cppn_transformed, coord, outgoing):
        """Initialize the quadtree by dividing it in appropriate quads."""
        root = QuadPoint(0.0, 0.0, 1.0, 1)
        queue = [root]
        
        while queue:
            point = queue.pop(0)
            
            # Create child quadpoints
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
            
            # Query CPPN for each child
            for child in point.children:
                child.weight = self.query_cppn(
                    state, cppn_transformed, coord, (child.x, child.y), outgoing
                )
            
            # Decide whether to subdivide further
            if (point.level < self.initial_depth) or (
                point.level < self.max_depth and 
                self.variance(point) > self.division_threshold
            ):
                for child in point.children:
                    queue.append(child)
        
        return root
    
    def pruning_extraction(self, state, cppn_transformed, coord, point, outgoing):
        """Determines which connections to express - high variance = more connections."""
        connections = set()
        
        def extract(p):
            for child in p.children:
                if child is None:
                    continue
                    
                if self.variance(child) > self.variance_threshold:
                    # High variance - recurse deeper
                    extract(child)
                else:
                    # Low variance - check for band discontinuity
                    d_left = abs(child.weight - self.query_cppn(
                        state, cppn_transformed, coord, 
                        (child.x - p.width, child.y), outgoing
                    ))
                    d_right = abs(child.weight - self.query_cppn(
                        state, cppn_transformed, coord, 
                        (child.x + p.width, child.y), outgoing
                    ))
                    d_top = abs(child.weight - self.query_cppn(
                        state, cppn_transformed, coord, 
                        (child.x, child.y - p.width), outgoing
                    ))
                    d_bottom = abs(child.weight - self.query_cppn(
                        state, cppn_transformed, coord, 
                        (child.x, child.y + p.width), outgoing
                    ))
                    
                    # Check for band discontinuity
                    if max(min(d_top, d_bottom), min(d_left, d_right)) > self.band_threshold:
                        if outgoing:
                            conn = Connection(coord[0], coord[1], child.x, child.y, child.weight)
                        else:
                            conn = Connection(child.x, child.y, coord[0], coord[1], child.weight)
                        
                        # CRITICAL: ES-HyperNEAT validates the CHILD's weight, not the connection weight!
                        # This is because the child represents a potential node location, and we only
                        # create connections to locations with non-zero activation.
                        # Also: Connections must be upward (y1 < y2) - no lateral or downward connections.
                        if child.weight != 0.0 and conn.y1 < conn.y2 and not (conn.x1 == conn.x2 and conn.y1 == conn.y2):
                            connections.add(conn)
        
        extract(point)
        return connections
    
    def clean_net(self, connections):
        """
        Clean a net for dangling connections:
        Intersects paths from input nodes with paths to output.
        """
        connected_to_inputs = set(tuple(float(c) for c in coord) for coord in self.substrate.input_coordinates)
        connected_to_outputs = set(tuple(float(c) for c in coord) for coord in self.substrate.output_coordinates)
        true_connections = set()
        
        initial_input_connections = copy.deepcopy(connections)
        initial_output_connections = copy.deepcopy(connections)
        
        # Find all nodes reachable from inputs
        add_happened = True
        while add_happened:
            add_happened = False
            temp_connections = copy.deepcopy(initial_input_connections)
            for conn in temp_connections:
                if (conn.x1, conn.y1) in connected_to_inputs:
                    connected_to_inputs.add((conn.x2, conn.y2))
                    initial_input_connections.remove(conn)
                    add_happened = True
        
        # Find all nodes that can reach outputs
        add_happened = True
        while add_happened:
            add_happened = False
            temp_connections = copy.deepcopy(initial_output_connections)
            for conn in temp_connections:
                if (conn.x2, conn.y2) in connected_to_outputs:
                    connected_to_outputs.add((conn.x1, conn.y1))
                    initial_output_connections.remove(conn)
                    add_happened = True
        
        # Find true nodes (connected to both inputs and outputs)
        true_nodes = connected_to_inputs.intersection(connected_to_outputs)
        
        # Keep only connections between true nodes
        for conn in connections:
            if ((conn.x1, conn.y1) in true_nodes and 
                (conn.x2, conn.y2) in true_nodes):
                true_connections.add(conn)
        
        # Remove input and output coordinates from hidden nodes
        input_coords = set(tuple(float(c) for c in coord) for coord in self.substrate.input_coordinates)
        output_coords = set(tuple(float(c) for c in coord) for coord in self.substrate.output_coordinates)
        true_nodes = true_nodes - input_coords - output_coords
        
        return true_nodes, true_connections
    
    def es_hyperneat(self, state, cppn_transformed):
        """Explores the hidden nodes and their connections."""
        inputs = self.substrate.input_coordinates
        outputs = self.substrate.output_coordinates
        hidden_nodes = set()
        unexplored_hidden_nodes = set()
        connections1, connections2, connections3 = set(), set(), set()
        
        # Explore from inputs
        for coord in inputs:
            root = self.division_initialization(state, cppn_transformed, tuple(coord), True)
            conns = self.pruning_extraction(state, cppn_transformed, tuple(coord), root, True)
            connections1 = connections1.union(conns)
            for conn in connections1:
                hidden_nodes.add((conn.x2, conn.y2))
        
        unexplored_hidden_nodes = copy.deepcopy(hidden_nodes)
        
        # Explore from hidden nodes
        for iteration in range(self.iteration_level):
            new_hidden_nodes = set()
            
            for coord in unexplored_hidden_nodes:
                root = self.division_initialization(state, cppn_transformed, coord, True)
                conns = self.pruning_extraction(state, cppn_transformed, coord, root, True)
                connections2 = connections2.union(conns)
                
                # Track new nodes discovered this iteration
                for conn in conns:
                    if (conn.x2, conn.y2) not in hidden_nodes:
                        new_hidden_nodes.add((conn.x2, conn.y2))
            
            hidden_nodes.update(new_hidden_nodes)
            unexplored_hidden_nodes = hidden_nodes - unexplored_hidden_nodes
        
        # Explore to outputs
        for coord in outputs:
            root = self.division_initialization(state, cppn_transformed, tuple(coord), False)
            conns = self.pruning_extraction(state, cppn_transformed, tuple(coord), root, False)
            connections3 = connections3.union(conns)
        
        # Combine all connections
        connections = connections1.union(connections2).union(connections3)
        
        return self.clean_net(connections)