"""Custom substrate implementations for ES-HyperNEAT."""

from dataclasses import dataclass
from typing import List, Tuple
import jax.numpy as jnp
from jax import vmap
from tensorneat.algorithm.hyperneat.substrate import BaseSubstrate


class Substrate:
    """Simple substrate for ES-HyperNEAT testing."""
    
    def __init__(self, input_coords: List[tuple], output_coords: List[tuple]):
        """
        Initialize substrate.
        
        Args:
            input_coords: List of (x, y) coordinates for inputs
            output_coords: List of (x, y) coordinates for outputs
        """
        self.input_coords = input_coords
        self.output_coords = output_coords
        self.input_coordinates = input_coords  # Alias for compatibility
        self.output_coordinates = output_coords  # Alias for compatibility
        self.num_inputs = len(input_coords)
        self.num_outputs = len(output_coords)


class ESHyperSubstrate(BaseSubstrate):
    """Basic ES-HyperNEAT substrate with bias support."""
    
    def __init__(self, input_coordinates, hidden_coordinates, output_coordinates, 
                 resolution, include_bias=False):
        # Ensure that all coordinate arrays are 2D, even if empty
        self.input_coordinates = jnp.array(input_coordinates).reshape(-1, 2)
        self.hidden_coordinates = jnp.array(hidden_coordinates).reshape(-1, 2)
        self.output_coordinates = jnp.array(output_coordinates).reshape(-1, 2)
        self.resolution = resolution
        self.include_bias = include_bias

        # Add bias coordinate if include_bias is True
        if self.include_bias:
            self.add_bias_coordinate()

        # Initialize the nodes and connections for the substrate
        self.nodes, self.conns = self.initialize_substrate()

    def add_bias_coordinate(self):
        """Add bias input node with automatic positioning."""
        # Determine alignment of input nodes
        x_coords = self.input_coordinates[:, 0]
        y_coords = self.input_coordinates[:, 1]
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()

        if x_range < y_range:
            # Nodes are vertically aligned
            alignment = 'vertical'
            varying_axis = y_coords
            fixed_value = x_coords.mean()
            axis_index = 1
        else:
            # Nodes are horizontally aligned
            alignment = 'horizontal'
            varying_axis = x_coords
            fixed_value = y_coords.mean()
            axis_index = 0

        # Sort the varying axis values
        sorted_indices = jnp.argsort(varying_axis)
        sorted_coords = varying_axis[sorted_indices]

        # Calculate average spacing between nodes
        if sorted_coords.shape[0] > 1:
            spacings = jnp.diff(sorted_coords)
            avg_spacing = spacings.mean()
        else:
            avg_spacing = 1.0

        # Determine the position for the bias coordinate
        new_coord_value = sorted_coords[-1] + avg_spacing

        # Construct the bias coordinate
        if alignment == 'vertical':
            bias_coordinate = jnp.array([[fixed_value, new_coord_value]])
        else:
            bias_coordinate = jnp.array([[new_coord_value, fixed_value]])

        # Check for overlaps
        existing_coords = self.input_coordinates
        if not jnp.any(jnp.all(existing_coords == bias_coordinate, axis=1)):
            self.input_coordinates = jnp.concatenate([self.input_coordinates, bias_coordinate], axis=0)
        else:
            # If overlap, adjust the position slightly
            bias_coordinate[0, axis_index] += avg_spacing
            self.input_coordinates = jnp.concatenate([self.input_coordinates, bias_coordinate], axis=0)

    @property
    def num_inputs(self):
        return self.input_coordinates.shape[0]

    @property
    def num_outputs(self):
        return self.output_coordinates.shape[0]

    @property
    def nodes_cnt(self):
        return self.nodes.shape[0]

    @property
    def conns_cnt(self):
        return self.conns.shape[0]

    def initialize_substrate(self):
        """Initialize substrate nodes and connections."""
        # Assign indices to nodes
        input_idx = jnp.arange(self.num_inputs)
        hidden_idx = jnp.arange(self.num_inputs, self.num_inputs + self.hidden_coordinates.shape[0])
        output_idx = jnp.arange(self.num_inputs + self.hidden_coordinates.shape[0], 
                                self.num_inputs + self.hidden_coordinates.shape[0] + self.num_outputs)

        # Create nodes with indices and coordinates
        node_indices = jnp.concatenate([input_idx, hidden_idx, output_idx])
        node_coords = jnp.concatenate([self.input_coordinates, self.hidden_coordinates, self.output_coordinates], axis=0)
        nodes = jnp.column_stack([node_indices, node_coords])

        # Create connections
        connections = []
        if self.hidden_coordinates.shape[0] > 0:
            # Input to hidden
            input_hidden_conns = self.create_connections(input_idx, hidden_idx)
            connections.append(input_hidden_conns)
            # Hidden to hidden
            hidden_hidden_conns = self.create_connections(hidden_idx, hidden_idx)
            connections.append(hidden_hidden_conns)
            # Hidden to output
            hidden_output_conns = self.create_connections(hidden_idx, output_idx)
            connections.append(hidden_output_conns)
        else:
            # Input to output directly
            input_output_conns = self.create_connections(input_idx, output_idx)
            connections.append(input_output_conns)

        conns = jnp.concatenate(connections, axis=0)
        return nodes, conns

    def create_connections(self, from_indices, to_indices):
        """Generate all combinations of from_indices and to_indices."""
        from_indices_repeated = jnp.repeat(from_indices, to_indices.shape[0])
        to_indices_tiled = jnp.tile(to_indices, from_indices.shape[0])
        conns = jnp.column_stack([from_indices_repeated, to_indices_tiled])
        return conns

    @property
    def query_coors(self):
        """Prepare CPPN input coordinates for each connection."""
        from_coords = self.nodes[self.conns[:, 0].astype(int), 1:]
        to_coords = self.nodes[self.conns[:, 1].astype(int), 1:]

        # Always include bias for CPPN queries
        bias_input = jnp.ones((from_coords.shape[0], 1))
        query_coordinates = jnp.concatenate([from_coords, to_coords, bias_input], axis=1)

        return query_coordinates

    def make_nodes(self, nodes_indices):
        """Return nodes at the given indices."""
        return self.nodes[nodes_indices]

    def make_conns(self, weights):
        """Assign weights to connections."""
        conns_with_weights = jnp.column_stack([self.conns, weights])
        return conns_with_weights


class QuadPoint:
    """
    Class representing an area in the quadtree.
    Defined by a center coordinate and the distance to the edges of the area.
    """
    
    def __init__(self, x, y, width, level):
        self.x = x
        self.y = y
        self.width = width
        self.level = level
        self.weight = 0.0
        self.children = [None] * 4  # 4 quadrants


class Connection:
    """
    Class representing a connection from one point to another with a certain weight.
    """
    
    def __init__(self, x1, y1, x2, y2, weight):
        self.x1 = float(x1) if hasattr(x1, '__float__') else x1
        self.y1 = float(y1) if hasattr(y1, '__float__') else y1
        self.x2 = float(x2) if hasattr(x2, '__float__') else x2
        self.y2 = float(y2) if hasattr(y2, '__float__') else y2
        self.weight = float(weight) if hasattr(weight, '__float__') else weight
    
    def __eq__(self, other):
        if not isinstance(other, Connection):
            return NotImplemented
        return (self.x1, self.y1, self.x2, self.y2) == (other.x1, other.y1, other.x2, other.y2)
    
    def __hash__(self):
        # Only hash coordinates, not weight (to match __eq__)
        return hash((self.x1, self.y1, self.x2, self.y2))


class ESHyperNEATUtils:
    """Utility functions for ES-HyperNEAT quadtree decomposition."""
    
    @staticmethod
    def divide_pattern(p1: QuadPoint, p2: QuadPoint, p3: QuadPoint, p4: QuadPoint, 
                      level: int, threshold: float, min_level: int) -> List[QuadPoint]:
        """Recursively divide a pattern into smaller regions to find high-variance points."""
        points = []
        
        if level == min_level:
            return [p1, p2, p3, p4]
            
        # Calculate center point weights using JAX operations
        center_x = (p1.x + p2.x + p3.x + p4.x) / 4
        center_y = (p1.y + p2.y + p3.y + p4.y) / 4
        center_weight = (p1.weight + p2.weight + p3.weight + p4.weight) / 4
        
        # Create center point
        center = QuadPoint(center_x, center_y, center_weight)
        
        # Calculate variance
        variance = jnp.sqrt(
            (p1.weight - center.weight)**2 +
            (p2.weight - center.weight)**2 +
            (p3.weight - center.weight)**2 +
            (p4.weight - center.weight)**2
        ) / 4
        
        if variance > threshold or level < min_level:
            # Divide into four quadrants
            points.extend(ESHyperNEATUtils.divide_pattern(
                p1, p2, center, p4, level + 1, threshold, min_level
            ))
            points.extend(ESHyperNEATUtils.divide_pattern(
                p2, p3, center, p1, level + 1, threshold, min_level
            ))
            points.extend(ESHyperNEATUtils.divide_pattern(
                p3, p4, center, p2, level + 1, threshold, min_level
            ))
            points.extend(ESHyperNEATUtils.divide_pattern(
                p4, p1, center, p3, level + 1, threshold, min_level
            ))
        else:
            points.append(center)
            
        return points


class ESHyperNEATSubstrate(BaseSubstrate):
    """Enhanced substrate with quadtree decomposition support."""
    
    def __init__(self, input_coordinates, hidden_coordinates, output_coordinates, 
                 resolution=10, include_bias=True, variance_threshold=0.3,
                 band_threshold=0.3, min_level=2, max_level=5):
        super().__init__()
        
        # Store configuration
        self.resolution = resolution
        self.include_bias = include_bias
        self.variance_threshold = variance_threshold
        self.band_threshold = band_threshold
        self.min_level = min_level
        self.max_level = max_level
        
        # Convert coordinates to JAX arrays
        self.input_coordinates = jnp.array(input_coordinates, dtype=jnp.float32)
        self.hidden_coordinates = jnp.array(hidden_coordinates, dtype=jnp.float32)
        self.output_coordinates = jnp.array(output_coordinates, dtype=jnp.float32)
        
        # Ensure 2D shapes
        if self.input_coordinates.ndim == 1:
            self.input_coordinates = self.input_coordinates.reshape(-1, 2)
        if self.hidden_coordinates.ndim == 1:
            self.hidden_coordinates = self.hidden_coordinates.reshape(-1, 2)
        if self.output_coordinates.ndim == 1:
            self.output_coordinates = self.output_coordinates.reshape(-1, 2)
        
        # Add bias if needed
        if self.include_bias:
            self._add_bias_coordinate()
        
        # Initialize structure
        self._initialize_structure()
        
    def _add_bias_coordinate(self):
        """Add bias input node with proper positioning."""
        if self.input_coordinates.shape[0] > 0:
            x_coords = self.input_coordinates[:, 0]
            y_coords = self.input_coordinates[:, 1]
            
            # Determine if inputs are arranged horizontally or vertically
            x_range = jnp.ptp(x_coords)
            y_range = jnp.ptp(y_coords)
            
            if x_range > y_range:
                # Horizontal arrangement
                bias_x = jnp.max(x_coords) + 0.2
                bias_y = jnp.mean(y_coords)
            else:
                # Vertical arrangement
                bias_x = jnp.mean(x_coords)
                bias_y = jnp.max(y_coords) + 0.2
                
            bias_coord = jnp.array([[bias_x, bias_y]], dtype=jnp.float32)
        else:
            # Default position if no other inputs
            bias_coord = jnp.array([[0.0, -1.0]], dtype=jnp.float32)
            
        self.input_coordinates = jnp.concatenate([self.input_coordinates, bias_coord], axis=0)
        
    def _initialize_structure(self):
        """Initialize substrate structure with proper shapes."""
        # Calculate total number of nodes
        self.num_nodes = (self.input_coordinates.shape[0] + 
                         self.hidden_coordinates.shape[0] + 
                         self.output_coordinates.shape[0])
        
        # Combine all coordinates
        all_coords = jnp.concatenate([
            self.input_coordinates,
            self.hidden_coordinates,
            self.output_coordinates
        ], axis=0)
        
        # Create node indices
        node_indices = jnp.arange(self.num_nodes, dtype=jnp.int32)
        
        # Create nodes array [index, x, y]
        self.nodes = jnp.concatenate([
            node_indices.reshape(-1, 1),
            all_coords
        ], axis=1)
        
        # Initialize connections
        self._initialize_connections()
        
    def _initialize_connections(self):
        """Initialize connection structure."""
        num_inputs = self.input_coordinates.shape[0]
        num_hidden = self.hidden_coordinates.shape[0]
        num_outputs = self.output_coordinates.shape[0]
        
        connections = []
        
        # Input to hidden connections
        if num_hidden > 0:
            for i in range(num_inputs):
                for h in range(num_hidden):
                    connections.append([i, num_inputs + h])
            
            # Hidden to output connections
            for h in range(num_hidden):
                for o in range(num_outputs):
                    connections.append([
                        num_inputs + h,
                        num_inputs + num_hidden + o
                    ])
        else:
            # Direct input to output connections
            for i in range(num_inputs):
                for o in range(num_outputs):
                    connections.append([i, num_inputs + o])
        
        self.conns = jnp.array(connections, dtype=jnp.int32)
        
    @property
    def query_coords(self):
        """Generate query coordinates for CPPN evaluation."""
        from_coords = self.nodes[self.conns[:, 0], 1:]
        to_coords = self.nodes[self.conns[:, 1], 1:]
        
        # Always include bias for CPPN queries
        bias = jnp.ones((from_coords.shape[0], 1), dtype=jnp.float32)
        return jnp.concatenate([from_coords, to_coords, bias], axis=1)
    
    @property
    def num_inputs(self):
        return self.input_coordinates.shape[0]
    
    @property
    def num_outputs(self):
        return self.output_coordinates.shape[0]
    
    @property
    def nodes_cnt(self):
        return self.nodes.shape[0]
    
    @property
    def conns_cnt(self):
        return self.conns.shape[0]
    
    def query_coors(self):
        """Alias for query_coords to maintain compatibility."""
        return self.query_coords
    
    def make_nodes(self, indices):
        """Create nodes from indices with proper shape handling."""
        indices = jnp.asarray(indices, dtype=jnp.int32)
        valid_mask = indices < self.nodes_cnt
        return jnp.where(
            valid_mask[:, None],
            self.nodes[indices],
            jnp.nan
        )
    
    def make_conns(self, weights):
        """Create connections with weights."""
        weights = jnp.asarray(weights)
        valid_mask = ~jnp.isnan(weights)
        return jnp.where(
            valid_mask[:, None],
            jnp.concatenate([self.conns, weights[:, None]], axis=1),
            jnp.nan
        )