"""Unit tests for the main impl's pure substrate-discovery helpers (no evolution, no CPPN).

These exercise the lazy-quadtree building blocks that do the actual work of ES-HyperNEAT
substrate discovery in ``eshyperneat.py`` -- weight expression, leaf-weight collection, region
variance, quadtree subdivision, and network reachability cleaning. Only the two value classes
(``Connection``/``QuadPoint``) were covered before; the helpers below are the real algorithm.

The class ``__init__`` constructs a TensorNEATAdapter, so each test builds a surrogate instance
via ``object.__new__`` and sets only the one or two attributes the helper under test reads.
"""
import math

import numpy as np
import pytest

from jax_es_hyperneat.eshyperneat import Connection, QuadPoint, TensorNEATESHyperNEATOptimized


def _algo(**attrs):
    """A bare instance with only the attributes a helper needs (skips __init__/adapter)."""
    obj = object.__new__(TensorNEATESHyperNEATOptimized)
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def _qp(x=0.0, y=0.0, width=1.0, level=1, weight=0.0):
    q = QuadPoint(x, y, width, level)
    q.weight = weight
    return q


# --- _process_weights: CPPN weight sparsification + scaling (main impl's own copy) --------- #

@pytest.mark.parametrize("raw,expected", [
    (0.1, 0.0),     # below threshold -> dead
    (0.2, 0.0),     # exactly at threshold (strict >) -> dead
    (-0.2, 0.0),
    (0.6, 4.0),     # (0.6-0.2)/0.8 * 8 = 4.0
    (1.0, 8.0),     # saturates at +max_weight
    (-1.0, -8.0),
    (-0.6, -4.0),
])
def test_process_weights_table(raw, expected):
    out = _algo(max_weight=8.0)._process_weights(np.array([raw], dtype=np.float64))
    assert out[0] == pytest.approx(expected)


def test_process_weights_nan_and_inf_become_zero():
    out = _algo(max_weight=8.0)._process_weights(
        np.array([float("nan"), float("inf"), float("-inf")], dtype=np.float64))
    assert list(out) == [0.0, 0.0, 0.0]
    assert not any(math.isnan(w) or math.isinf(w) for w in out)


def test_process_weights_respects_custom_max_weight():
    out = _algo(max_weight=5.0)._process_weights(np.array([0.6], dtype=np.float64))
    assert out[0] == pytest.approx(2.5)


def test_process_weights_preserves_length_and_order():
    out = _algo(max_weight=8.0)._process_weights(np.array([1.0, 0.1, -1.0], dtype=np.float64))
    assert list(out) == [pytest.approx(8.0), 0.0, pytest.approx(-8.0)]


# --- _get_weights: recursive leaf-weight collection (static) -------------------------------- #

def test_get_weights_single_leaf():
    assert TensorNEATESHyperNEATOptimized._get_weights(_qp(weight=0.7)) == [0.7]


def test_get_weights_collects_leaf_children():
    root = _qp(weight=99.0)  # root weight ignored: all four children present -> internal node
    root.children = [_qp(weight=1.0), _qp(weight=2.0), _qp(weight=3.0), _qp(weight=4.0)]
    assert sorted(TensorNEATESHyperNEATOptimized._get_weights(root)) == [1.0, 2.0, 3.0, 4.0]


def test_get_weights_partial_children_is_treated_as_leaf():
    # A node with any None child (here 3 set + 1 None) is a leaf -> its OWN weight, no recursion.
    root = _qp(weight=5.0)
    root.children = [_qp(weight=1.0), _qp(weight=2.0), _qp(weight=3.0), None]
    assert TensorNEATESHyperNEATOptimized._get_weights(root) == [5.0]


def test_get_weights_recurses_into_nested_subtree():
    inner = _qp(weight=88.0)
    inner.children = [_qp(weight=10.0), _qp(weight=20.0), _qp(weight=30.0), _qp(weight=40.0)]
    root = _qp(weight=99.0)
    root.children = [_qp(weight=1.0), _qp(weight=2.0), inner, _qp(weight=4.0)]
    # inner expands to its 4 grandchildren; the other three children stay leaves
    assert sorted(TensorNEATESHyperNEATOptimized._get_weights(root)) == \
        [1.0, 2.0, 4.0, 10.0, 20.0, 30.0, 40.0]


# --- _variance: population variance over a region's leaf weights ---------------------------- #

def test_variance_is_population_variance_over_leaves():
    algo = _algo(_variance_call_count=0)
    root = _qp()
    root.children = [_qp(weight=w) for w in (1.0, 2.0, 3.0, 4.0)]
    assert algo._variance(root) == pytest.approx(float(np.var([1.0, 2.0, 3.0, 4.0])))
    assert algo._variance_call_count == 1


def test_variance_single_point_is_zero():
    algo = _algo(_variance_call_count=0)
    assert algo._variance(_qp(weight=3.0)) == 0.0
    assert algo._variance_call_count == 1


def test_variance_none_is_zero_but_still_counts_the_call():
    algo = _algo(_variance_call_count=0)
    assert algo._variance(None) == 0.0
    assert algo._variance_call_count == 1  # counter increments before the None guard


# --- _create_children_fast: quadtree subdivision geometry ----------------------------------- #

def test_create_children_geometry_at_origin():
    kids = _algo(_child_offsets=None)._create_children_fast(QuadPoint(0.0, 0.0, 1.0, 1))
    coords = [(round(float(k.x), 6), round(float(k.y), 6)) for k in kids]
    assert coords == [(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)]  # BL, TL, TR, BR
    assert all(k.width == pytest.approx(0.5) for k in kids)
    assert all(k.level == 2 for k in kids)


def test_create_children_geometry_off_origin():
    kids = _algo(_child_offsets=None)._create_children_fast(QuadPoint(0.5, 0.5, 0.5, 2))
    coords = [(round(float(k.x), 6), round(float(k.y), 6)) for k in kids]
    assert coords == [(0.25, 0.25), (0.25, 0.75), (0.75, 0.75), (0.75, 0.25)]
    assert all(k.width == pytest.approx(0.25) for k in kids)
    assert all(k.level == 3 for k in kids)


def test_create_children_initializes_offsets_lazily():
    algo = _algo(_child_offsets=None)
    algo._create_children_fast(QuadPoint(0.0, 0.0, 1.0, 1))
    assert algo._child_offsets is not None and algo._child_offsets.shape == (4, 2)


# --- _clean_net: forward-reachable-from-inputs intersect backward-reachable-to-outputs ------ #

def test_clean_net_keeps_only_the_connecting_path():
    algo = _algo(substrate_input_coords=[(0.0, 0.0)], substrate_output_coords=[(1.0, 1.0)])
    path = {Connection(0.0, 0.0, 0.5, 0.5, 0.1), Connection(0.5, 0.5, 1.0, 1.0, 0.2)}
    nodes, true_conns = algo._clean_net(set(path))
    assert nodes == {(0.5, 0.5)}        # input/output coords stripped from the node set
    assert true_conns == path


def test_clean_net_prunes_dead_end_and_island():
    algo = _algo(substrate_input_coords=[(0.0, 0.0)], substrate_output_coords=[(1.0, 1.0)])
    path = {Connection(0.0, 0.0, 0.5, 0.5, 0.1), Connection(0.5, 0.5, 1.0, 1.0, 0.2)}
    dead_end = Connection(0.0, 0.0, 0.9, 0.9, 0.3)   # reachable from input, never reaches output
    island = Connection(0.3, 0.3, 0.4, 0.4, 0.4)     # neither input-reachable nor output-reaching
    nodes, true_conns = algo._clean_net(path | {dead_end, island})
    assert nodes == {(0.5, 0.5)}
    assert true_conns == path
    assert dead_end not in true_conns and island not in true_conns
