"""Unit tests for the pure HSHG-core building blocks (no evolution; jax/numpy only).

These exercise the actual substrate-discovery primitives: the band-detection formula, the CPPN
weight sparsification/scaling, the HSHG configuration, and grid-position generation.
"""
import pytest

from jax_es_hyperneat.hshg.hshg_core.config import HSHGConfig
from jax_es_hyperneat.hshg.hshg_core.division import (
    DiscoveredNode, GridPosition, generate_all_grid_positions, sparsify_and_scale_weight,
)
from jax_es_hyperneat.hshg.hshg_core.pruning import (
    Connection as PruningConnection, PruningResult, compute_band_metric,
)


# --- weight sparsification + scaling ---------------------------------------- #

@pytest.mark.parametrize("raw,expected", [
    (0.1, 0.0),     # below threshold -> dead
    (0.2, 0.0),     # exactly at threshold (strict >) -> dead
    (-0.2, 0.0),
    (0.6, 4.0),     # (0.6-0.2)/0.8 * 8 = 4.0
    (1.0, 8.0),     # saturates at +max_weight
    (-1.0, -8.0),
    (-0.6, -4.0),
])
def test_sparsify_and_scale_weight_defaults(raw, expected):
    assert sparsify_and_scale_weight(raw) == pytest.approx(expected)


def test_sparsify_respects_custom_max_weight():
    assert sparsify_and_scale_weight(0.6, max_weight=5.0) == pytest.approx(2.5)


# --- band-detection metric (the ES-HyperNEAT discovery signal) -------------- #

@pytest.mark.parametrize("c,l,r,t,b,expected", [
    (0, 0, 0, 0, 0, 0.0),   # flat field -> no band
    (0, 0, 0, 1, 1, 1.0),   # vertical band (both top & bottom differ)
    (0, 1, 1, 0, 0, 1.0),   # horizontal band (both left & right differ)
    (0, 1, 0, 1, 0, 0.0),   # one-sided gradient on each axis -> inner min = 0 -> no band
    (0, 1, 1, 1, 1, 1.0),   # surrounded -> band
])
def test_compute_band_metric(c, l, r, t, b, expected):
    # signature: compute_band_metric(center, left, right, top, bottom)
    assert compute_band_metric(c, l, r, t, b) == pytest.approx(expected)


def test_band_metric_requires_change_in_both_directions_of_an_axis():
    # A gradient present on only one side of both axes must NOT register as a band.
    one_sided = compute_band_metric(0.0, 1.0, 0.0, 1.0, 0.0)
    both_sided = compute_band_metric(0.0, 1.0, 1.0, 0.0, 0.0)
    assert one_sided == 0.0 and both_sided > 0.0


# --- HSHG configuration ----------------------------------------------------- #

def test_hshg_config_defaults():
    c = HSHGConfig()
    assert c.cell_size == 0.25 and c.max_nodes == 1000 and c.cell_capacity == 32
    assert c.num_cells == 1009 and c.hierarchy_levels == (0.125, 0.25, 0.5)
    assert c.num_levels == 3


@pytest.mark.parametrize("kwargs", [
    {"cell_size": 0.0}, {"max_nodes": 0}, {"cell_capacity": 0}, {"hierarchy_levels": ()},
])
def test_hshg_config_validation_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        HSHGConfig(**kwargs)


def test_hshg_config_for_es_hyperneat_node_budget():
    # max_nodes = sum(4^i for i in 1..max_depth) * 20; d=3 -> (4+16+64)*20 = 1680
    assert HSHGConfig.for_es_hyperneat(max_depth=3).max_nodes == 1680


# --- grid generation + value tuples ----------------------------------------- #

def test_generate_all_grid_positions_count_and_shape():
    positions, levels, widths = generate_all_grid_positions(max_depth=2, initial_depth=1)
    assert positions.shape == (20, 2)          # 4 (level 1) + 16 (level 2)
    assert len(levels) == 20 and len(widths) == 20
    assert {int(l) for l in levels} == {1, 2}


def test_named_tuples_fields():
    assert GridPosition(0.1, 0.2, 1, 0.5)._fields == ("x", "y", "level", "width")
    assert DiscoveredNode(0.1, 0.2, 0.5, 1).weight == 0.5
    assert PruningConnection(0, 0, 1, 1, 0.5).weight == 0.5
    pr = PruningResult(connections=set(), num_leaves_processed=3, num_connections_created=2)
    assert pr.num_leaves_processed == 3 and pr.num_connections_created == 2
