"""Unit tests for the impl's value classes: Connection and QuadPoint."""
import math

from jax_es_hyperneat.eshyperneat import Connection, QuadPoint


def test_connection_equality_keyed_on_coords_not_weight():
    a = Connection(0.0, 0.0, 1.0, 1.0, weight=0.5)
    b = Connection(0.0, 0.0, 1.0, 1.0, weight=-3.0)  # same coords, different weight
    assert a == b and hash(a) == hash(b)


def test_connection_distinct_coords_differ():
    a = Connection(0.0, 0.0, 1.0, 1.0, weight=0.5)
    c = Connection(0.0, 0.0, 1.0, 0.5, weight=0.5)
    assert a != c and hash(a) != hash(c)


def test_connection_set_dedupes_by_coords():
    conns = {Connection(0, 0, 1, 1, 0.5), Connection(0, 0, 1, 1, 9.0), Connection(0, 0, 1, 0, 0.5)}
    assert len(conns) == 2  # first two collapse to one


def test_connection_nan_weight_becomes_zero():
    c = Connection(0.0, 0.0, 1.0, 1.0, weight=float("nan"))
    assert c.weight == 0.0 and not math.isnan(c.weight)


def test_connection_compare_with_non_connection():
    assert Connection(0, 0, 1, 1, 0.5).__eq__(42) is NotImplemented


def test_quadpoint_init_defaults():
    q = QuadPoint(0.0, 0.0, width=1.0, level=1)
    assert (q.x, q.y, q.width, q.level) == (0.0, 0.0, 1.0, 1)
    assert q.weight == 0.0
    assert q.children == [None, None, None, None]
    assert q.cached_variance is None
