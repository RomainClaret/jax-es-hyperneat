"""Unit tests for the HSHG spatial-hash engine (jax/numpy only, no evolution).

The failed-HSHG ablation's data structure -- insert/query/directional-neighbor over a
hierarchical spatial hash grid -- was loaded by the suite but never asserted on. These tests pin
its behavior: hashing, allocation, insertion (incl. node overflow), radius queries, the
4-directional neighbor finder used for band detection, and clearing. The jitted query helpers
compile once (seconds); they stay in the fast gate alongside the existing JIT smoke test.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from jax_es_hyperneat.hshg.hshg_core import operations as ops
from jax_es_hyperneat.hshg.hshg_core.config import HSHGConfig
from jax_es_hyperneat.hshg.hshg_core.state import (
    HSHGState, QueryResult, create_empty_query_result,
)


def _f32(*vals):
    return jnp.array(vals, dtype=jnp.float32)


# --- spatial hashing -------------------------------------------------------------------------- #

def test_spatial_hash_is_deterministic_and_in_range():
    c = HSHGConfig()
    args = (_f32(0.3)[0], _f32(0.4)[0], c.cell_size, c.num_cells, c.hash_prime_x, c.hash_prime_y)
    h1, h2 = int(ops.spatial_hash(*args)), int(ops.spatial_hash(*args))
    assert h1 == h2 and 0 <= h1 < c.num_cells


def test_spatial_hash_same_cell_same_bucket():
    # Two points both inside the first cell [0, cell_size) hash identically (spatial coherence).
    c = HSHGConfig()  # cell_size 0.25
    a = int(ops.spatial_hash(_f32(0.01)[0], _f32(0.02)[0], c.cell_size, c.num_cells,
                             c.hash_prime_x, c.hash_prime_y))
    b = int(ops.spatial_hash(_f32(0.2)[0], _f32(0.24)[0], c.cell_size, c.num_cells,
                             c.hash_prime_x, c.hash_prime_y))
    assert a == b


def test_spatial_hash_2d_preserves_batch_shape_and_range():
    c = HSHGConfig()
    pos = jnp.array([[0.1, 0.1], [0.5, 0.5], [-0.3, 0.9]], dtype=jnp.float32)
    h = ops.spatial_hash_2d(pos, c.cell_size, c.num_cells, c.hash_prime_x, c.hash_prime_y)
    assert h.shape == (3,)
    assert bool(jnp.all(h >= 0)) and bool(jnp.all(h < c.num_cells))


# --- allocation ------------------------------------------------------------------------------- #

def test_allocate_shapes_and_empty_defaults():
    c = HSHGConfig()
    st = ops.allocate(c)
    assert st.positions.shape == (c.max_nodes, 2)
    assert st.weights.shape == (c.max_nodes,)
    assert st.levels.shape == (c.max_nodes,)
    assert st.valid_mask.shape == (c.max_nodes,)
    assert st.cells.shape == (c.num_levels, c.num_cells, c.cell_capacity)
    assert st.cell_counts.shape == (c.num_levels, c.num_cells)
    assert int(st.num_nodes) == 0
    assert not bool(jnp.any(st.valid_mask))
    assert bool(jnp.all(st.cells == -1))
    assert bool(jnp.all(st.cell_counts == 0))
    assert not bool(st.did_buffer_overflow)


# --- insertion -------------------------------------------------------------------------------- #

def test_insert_single_updates_node_and_cell():
    c = HSHGConfig()
    pos = _f32(0.3, 0.4)
    st = ops.insert_single(ops.allocate(c), pos, 1.5, 2, c.cell_size, c.num_cells,
                           c.hash_prime_x, c.hash_prime_y, c.cell_capacity, hierarchy_level=0)
    assert int(st.num_nodes) == 1
    assert bool(st.valid_mask[0])
    assert float(st.weights[0]) == pytest.approx(1.5)
    assert int(st.levels[0]) == 2
    assert np.allclose(np.array(st.positions[0]), [0.3, 0.4], atol=1e-6)
    cell = int(ops.spatial_hash(pos[0], pos[1], c.cell_size, c.num_cells,
                                c.hash_prime_x, c.hash_prime_y))
    assert int(st.cell_counts[0, cell]) == 1
    assert int(st.cells[0, cell, 0]) == 0  # node index 0 written to the cell's first slot
    assert not bool(st.did_buffer_overflow)


def test_insert_batch_counts_all_nodes():
    c = HSHGConfig()
    positions = jnp.array([[0.1, 0.1], [0.5, 0.5], [-0.3, 0.2]], dtype=jnp.float32)
    st = ops.insert_batch(ops.allocate(c), positions, _f32(1.0, 2.0, 3.0),
                          jnp.array([1, 2, 3], dtype=jnp.int32), c)
    assert int(st.num_nodes) == 3
    assert int(jnp.sum(st.valid_mask)) == 3
    assert not bool(st.did_buffer_overflow)
    assert np.allclose(np.array(st.positions[:3]), np.array(positions), atol=1e-6)


def test_insert_batch_flags_node_overflow():
    # max_nodes=2 -> the third insertion overflows and is dropped, flag is set.
    c = HSHGConfig(max_nodes=2, num_cells=7, cell_capacity=4, cell_size=0.5,
                   hierarchy_levels=(0.5,))
    positions = jnp.array([[0.1, 0.1], [0.6, 0.6], [1.1, 1.1]], dtype=jnp.float32)
    st = ops.insert_batch(ops.allocate(c), positions, _f32(1.0, 2.0, 3.0),
                          jnp.array([1, 1, 1], dtype=jnp.int32), c)
    assert int(st.num_nodes) == 2
    assert bool(st.did_buffer_overflow)


# --- radius queries --------------------------------------------------------------------------- #

def test_query_radius_finds_near_and_excludes_far():
    c = HSHGConfig()  # cell_size 0.25
    positions = jnp.array([[0.0, 0.0], [0.1, 0.0], [0.9, 0.0]], dtype=jnp.float32)
    st = ops.insert_batch(ops.allocate(c), positions, _f32(1.0, 1.0, 1.0),
                          jnp.array([1, 1, 1], dtype=jnp.int32), c)
    res = ops.query_radius_batch(st, jnp.array([[0.0, 0.0]], dtype=jnp.float32), 0.2, c,
                                 max_neighbors=10)
    assert int(res.neighbor_count[0]) == 2  # (0,0) and (0.1,0); (0.9,0) is out of radius
    dists = np.sort(np.array(res.neighbor_distances[0]))
    assert dists[0] == pytest.approx(0.0, abs=1e-5)
    assert dists[1] == pytest.approx(0.1, abs=1e-5)


# --- directional neighbors (band detection) --------------------------------------------------- #

def test_find_directional_neighbors_resolves_all_four():
    c = HSHGConfig()  # cell_size 0.25
    cx, cy, s = 0.5, 0.5, 0.25
    positions = jnp.array([
        [cx - s, cy],   # left   -> weight 1
        [cx + s, cy],   # right  -> weight 2
        [cx, cy - s],   # top    -> weight 3
        [cx, cy + s],   # bottom -> weight 4
    ], dtype=jnp.float32)
    st = ops.insert_batch(ops.allocate(c), positions, _f32(1.0, 2.0, 3.0, 4.0),
                          jnp.array([1, 1, 1, 1], dtype=jnp.int32), c)
    idx, w = ops.find_directional_neighbors(
        st, _f32(cx, cy), s, c.cell_size, c.num_cells, c.hash_prime_x, c.hash_prime_y,
        c.cell_capacity)
    assert np.all(np.array(idx) >= 0)
    assert np.allclose(np.array(w), [1.0, 2.0, 3.0, 4.0], atol=1e-5)


def test_find_directional_neighbors_marks_missing_direction():
    c = HSHGConfig()
    # only a left neighbor present at (0.25, 0.5); the other three directions are empty
    st = ops.insert_batch(ops.allocate(c), jnp.array([[0.25, 0.5]], dtype=jnp.float32),
                          _f32(7.0), jnp.array([1], dtype=jnp.int32), c)
    idx, w = ops.find_directional_neighbors(
        st, _f32(0.5, 0.5), 0.25, c.cell_size, c.num_cells, c.hash_prime_x, c.hash_prime_y,
        c.cell_capacity)
    idx, w = np.array(idx), np.array(w)
    assert idx[0] >= 0 and w[0] == pytest.approx(7.0)   # left found
    assert idx[1] == -1 and w[1] == pytest.approx(0.0)  # right missing -> -1 / 0.0


# --- clear ------------------------------------------------------------------------------------ #

def test_clear_resets_counts_but_keeps_shapes():
    c = HSHGConfig()
    st = ops.insert_batch(ops.allocate(c), jnp.array([[0.1, 0.1], [0.2, 0.2]], dtype=jnp.float32),
                          _f32(1.0, 2.0), jnp.array([1, 1], dtype=jnp.int32), c)
    assert int(st.num_nodes) == 2
    cl = ops.clear(st)
    assert int(cl.num_nodes) == 0
    assert not bool(jnp.any(cl.valid_mask))
    assert bool(jnp.all(cl.cells == -1))
    assert bool(jnp.all(cl.cell_counts == 0))
    assert not bool(cl.did_buffer_overflow)
    assert cl.positions.shape == st.positions.shape


# --- value types ------------------------------------------------------------------------------ #

def test_state_and_query_result_fields():
    assert HSHGState._fields == (
        "positions", "weights", "levels", "valid_mask", "num_nodes",
        "cells", "cell_counts", "did_buffer_overflow",
    )
    assert QueryResult._fields == ("neighbor_indices", "neighbor_distances", "neighbor_count")


def test_create_empty_query_result_padding():
    qr = create_empty_query_result(5)
    assert qr.neighbor_indices.shape == (5,)
    assert bool(jnp.all(qr.neighbor_indices == -1))
    assert bool(jnp.all(jnp.isinf(qr.neighbor_distances)))
    assert int(qr.neighbor_count) == 0
