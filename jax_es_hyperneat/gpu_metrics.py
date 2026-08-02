"""Lightweight GPU utilization / VRAM monitor.

Optional: if ``pynvml`` and an NVIDIA device are present it samples utilization in a
background thread; otherwise it is a no-op that reports zeros. The GPU-utilization
fields are recorded for provenance and are not load-bearing for solve-rate or timing
reproduction.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import jax


@dataclass
class GPUMetrics:
    avg_utilization: float = 0.0
    max_utilization: int = 0
    peak_vram_mb: int = 0


class GPUMonitor:
    """Context manager sampling NVIDIA GPU utilization when available."""

    def __init__(self, sample_interval_ms: int = 100):
        self._interval = sample_interval_ms / 1000.0
        self._samples = []
        self._peak_vram = 0
        self._stop = threading.Event()
        self._thread = None
        self._handle = None
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() > 0:
                self._pynvml = pynvml
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._pynvml = None

    def _sample_loop(self):
        while not self._stop.is_set():
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self._samples.append(util.gpu)
                self._peak_vram = max(self._peak_vram, int(mem.used / (1024 * 1024)))
            except Exception:
                pass
            time.sleep(self._interval)

    def __enter__(self):
        if self._handle is not None:
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def get_metrics(self) -> GPUMetrics:
        if not self._samples:
            return GPUMetrics(0.0, 0, self._peak_vram)
        return GPUMetrics(
            avg_utilization=float(sum(self._samples) / len(self._samples)),
            max_utilization=int(max(self._samples)),
            peak_vram_mb=self._peak_vram,
        )


def get_gpu_info() -> dict:
    """Best-effort device description for result metadata."""
    try:
        dev = jax.devices()[0]
        is_gpu = 'gpu' in str(dev).lower() or 'cuda' in str(dev).lower()
        info = {'name': str(dev), 'available': is_gpu, 'memory_mb': 0}
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() > 0:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                info['name'] = pynvml.nvmlDeviceGetName(h)
                info['memory_mb'] = int(pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024 * 1024))
                info['available'] = True
        except Exception:
            pass
        return info
    except Exception:
        return {'name': 'CPU', 'available': False, 'memory_mb': 0}
