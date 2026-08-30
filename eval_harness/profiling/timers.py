"""Timing primitives for the KV-compression performance benchmark.

Two rules encoded here, both of which are the difference between a real
measurement and a plausible-looking wrong one:

1. **Events are pre-allocated, recorded inside the loop, and read only after a
   single trailing ``synchronize()``.**  ``Event.elapsed_time()`` blocks until
   both events complete, so calling it inside the loop serializes the pipeline;
   allocating an event inside the loop puts a driver call (10-50 us under
   contention) *inside* the measured region.
2. **Wall and device time are both recorded.**  Their difference is the
   diagnostic: it isolates host work that happens off the CUDA stream.  In this
   codebase that is not hypothetical -- ``research_pipeline.py:477`` calls
   ``new_id.item()`` every decode step (a blocking D2H copy) and
   ``snapkv_sketch.py:190`` calls ``scores.max().item()`` once per layer during
   prefill.

``Recorder`` is constructed through an injected factory so the CUDA path can be
exercised on a CPU-only box (see ``tests/test_profiling_timers.py``).
"""

from __future__ import annotations

import time
from typing import List, Optional

import torch


class WallRecorder:
    """``time.perf_counter`` spans. Works everywhere; the CPU/test fallback."""

    kind = "wall"

    def __init__(self, capacity: int = 0) -> None:
        del capacity
        self._spans: List[tuple] = []
        self._t0: Optional[float] = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        if self._t0 is None:
            raise RuntimeError("stop() without start()")
        self._spans.append((self._t0, time.perf_counter()))
        self._t0 = None

    def read_ms(self) -> List[float]:
        return [(b - a) * 1000.0 for a, b in self._spans]

    def reset(self) -> None:
        self._spans.clear()
        self._t0 = None


class CudaEventRecorder:
    """Pre-allocated ``cuda.Event(enable_timing=True)`` pairs.

    ``capacity`` pairs are allocated up front; ``start``/``stop`` only call
    ``record()`` (~1-2 us host each).  ``read_ms`` performs ONE
    ``torch.cuda.synchronize()`` and then reads every span.
    """

    kind = "cuda_event"

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("CudaEventRecorder needs a positive capacity")
        # enable_timing defaults to False, and elapsed_time() on such an event
        # raises at runtime -- a bug unreachable on CPU, so it is asserted in tests.
        self._starts = [torch.cuda.Event(enable_timing=True) for _ in range(capacity)]
        self._ends = [torch.cuda.Event(enable_timing=True) for _ in range(capacity)]
        self._n = 0
        self._open = False

    def start(self) -> None:
        if self._n >= len(self._starts):
            raise RuntimeError(
                f"CudaEventRecorder capacity {len(self._starts)} exhausted; "
                "allocating events mid-loop would land a driver call inside the measurement"
            )
        self._starts[self._n].record()
        self._open = True

    def stop(self) -> None:
        if not self._open:
            raise RuntimeError("stop() without start()")
        self._ends[self._n].record()
        self._n += 1
        self._open = False

    def read_ms(self) -> List[float]:
        if self._n == 0:
            return []
        torch.cuda.synchronize()          # exactly one sync, after the loop
        return [
            float(self._starts[i].elapsed_time(self._ends[i]))   # already ms
            for i in range(self._n)
        ]

    def reset(self) -> None:
        self._n = 0
        self._open = False


class DualRecorder:
    """Records wall and device spans simultaneously.

    ``read_ms()`` returns the device spans when CUDA is in play (they exclude
    host-side gaps); ``read_wall_ms()`` returns the wall spans.  The gap between
    the two sums is reported as ``host_gap_ms``.
    """

    def __init__(self, capacity: int, device_recorder=None) -> None:
        self.wall = WallRecorder()
        self.device = device_recorder
        self.kind = device_recorder.kind if device_recorder is not None else self.wall.kind

    def start(self) -> None:
        if self.device is not None:
            self.device.start()
        self.wall.start()

    def stop(self) -> None:
        self.wall.stop()
        if self.device is not None:
            self.device.stop()

    def read_wall_ms(self) -> List[float]:
        return self.wall.read_ms()

    def read_device_ms(self) -> List[float]:
        return self.device.read_ms() if self.device is not None else []

    def read_ms(self) -> List[float]:
        dev = self.read_device_ms()
        return dev if dev else self.read_wall_ms()

    def reset(self) -> None:
        self.wall.reset()
        if self.device is not None:
            self.device.reset()


def make_recorder(capacity: int, device: str = "cpu") -> DualRecorder:
    """Factory: a CUDA-event recorder on GPU, wall-only elsewhere."""
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    dev_rec = CudaEventRecorder(capacity) if use_cuda else None
    return DualRecorder(capacity, dev_rec)


def synchronize(device: str = "cpu") -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
