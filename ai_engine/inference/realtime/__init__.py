"""Real-time inference and audio streaming package (P4)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_engine.inference.realtime.latency import LatencyResult, measure_latency
    from ai_engine.inference.realtime.stream import (
        PassthroughStreamingEnhancer,
        StreamStatistics,
        list_audio_devices,
        print_audio_devices,
        run_stream,
    )

__all__ = [
    "LatencyResult",
    "PassthroughStreamingEnhancer",
    "StreamStatistics",
    "list_audio_devices",
    "measure_latency",
    "print_audio_devices",
    "run_stream",
]


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes to prevent runpy circular execution warnings."""
    if name in {"LatencyResult", "measure_latency"}:
        from ai_engine.inference.realtime import latency

        return getattr(latency, name)
    if name in {
        "PassthroughStreamingEnhancer",
        "StreamStatistics",
        "list_audio_devices",
        "print_audio_devices",
        "run_stream",
    }:
        import ai_engine.inference.realtime.stream as s

        return getattr(s, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
