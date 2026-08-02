"""Unit tests for the optional GPU monitor (must be a clean no-op without an NVIDIA device)."""
import pytest

from jax_es_hyperneat.gpu_metrics import GPUMetrics, GPUMonitor, get_gpu_info


def test_gpu_monitor_is_a_noop_context_manager():
    with GPUMonitor(sample_interval_ms=10) as mon:
        pass  # no NVIDIA device on CI/dev -> background sampler never starts
    m = mon.get_metrics()
    assert isinstance(m, GPUMetrics)
    assert m.avg_utilization == 0.0 and m.max_utilization == 0


def test_get_metrics_aggregates_collected_samples():
    # The no-op test only covers the empty-samples path; this covers the avg/max/peak aggregation.
    mon = GPUMonitor(sample_interval_ms=10)
    mon._samples = [10, 20, 30]
    mon._peak_vram = 100
    m = mon.get_metrics()
    assert m.avg_utilization == pytest.approx(20.0)
    assert m.max_utilization == 30
    assert m.peak_vram_mb == 100


def test_gpu_metrics_defaults():
    m = GPUMetrics()
    assert (m.avg_utilization, m.max_utilization, m.peak_vram_mb) == (0.0, 0, 0)


def test_get_gpu_info_returns_expected_keys():
    info = get_gpu_info()
    assert {"name", "available", "memory_mb"} <= set(info)
    assert isinstance(info["available"], bool)
