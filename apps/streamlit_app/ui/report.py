"""RunReport data structure and utilities for tracking analysis runs.

This module provides:
- RunReport TypedDict for structured analysis results
- Timing tracking utilities
- Report serialization for export
"""

import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypedDict
from uuid import uuid4


class AudioStats(TypedDict):
    """Audio statistics computed from the waveform."""
    duration_sec: float
    sample_rate: int
    rms: float
    peak: float
    silence_ratio: float


class TimingInfo(TypedDict, total=False):
    """Step timings in milliseconds."""
    load_ms: float
    validate_ms: float
    preprocess_ms: float
    windowing_ms: float
    infer_ms: float
    smooth_ms: float
    merge_ms: float
    total_ms: float


class ConfigSnapshot(TypedDict, total=False):
    """Snapshot of configuration used for the run."""
    model_id: str
    device: str
    mode: str  # "single" or "timeline"
    window_sec: float
    hop_sec: float
    pad_mode: str
    smoothing_method: str
    hysteresis_min_run: int
    majority_window: int
    ema_alpha: float
    include_scores: bool
    include_windows: bool


class RunReport(TypedDict, total=False):
    """Structured report of an analysis run."""
    run_id: str
    timestamp_start: str
    timestamp_end: str
    config: ConfigSnapshot
    audio_stats: AudioStats
    timings: TimingInfo
    logs: list[str]
    results: dict[str, Any]
    error: str | None
    error_traceback: str | None


@dataclass
class TimingTracker:
    """Track timings for individual pipeline steps."""
    
    _timings: dict[str, float] = field(default_factory=dict)
    _start_times: dict[str, float] = field(default_factory=dict)
    _total_start: float = field(default_factory=time.perf_counter)
    
    def start(self, step: str) -> None:
        """Start timing a step."""
        self._start_times[step] = time.perf_counter()
    
    def stop(self, step: str) -> float:
        """Stop timing a step and return duration in ms."""
        if step not in self._start_times:
            return 0.0
        elapsed_ms = (time.perf_counter() - self._start_times[step]) * 1000
        self._timings[f"{step}_ms"] = elapsed_ms
        return elapsed_ms
    
    def get_timings(self) -> TimingInfo:
        """Get all recorded timings."""
        result = dict(self._timings)
        result["total_ms"] = (time.perf_counter() - self._total_start) * 1000
        return result  # type: ignore


def create_run_report(
    config: ConfigSnapshot,
    audio_stats: AudioStats | None = None,
    timings: TimingInfo | None = None,
    logs: list[str] | None = None,
    results: dict[str, Any] | None = None,
    error: Exception | None = None,
) -> RunReport:
    """Create a new RunReport with the given data.
    
    Args:
        config: Configuration snapshot.
        audio_stats: Audio statistics (optional).
        timings: Step timings (optional).
        logs: Log messages (optional).
        results: Analysis results (optional).
        error: Exception if analysis failed (optional).
        
    Returns:
        Complete RunReport dict.
    """
    now = datetime.now()
    
    report: RunReport = {
        "run_id": str(uuid4()),
        "timestamp_start": now.isoformat(),
        "timestamp_end": now.isoformat(),
        "config": config,
        "logs": logs or [],
        "error": None,
        "error_traceback": None,
    }
    
    if audio_stats:
        report["audio_stats"] = audio_stats
    
    if timings:
        report["timings"] = timings
    
    if results:
        report["results"] = results
    
    if error:
        report["error"] = str(error)
        report["error_traceback"] = traceback.format_exc()
    
    return report


def report_to_json(report: RunReport) -> str:
    """Serialize RunReport to JSON string.
    
    Args:
        report: The report to serialize.
        
    Returns:
        JSON string representation.
    """
    def default_serializer(obj: Any) -> Any:
        """Handle non-serializable types."""
        if hasattr(obj, "item"):  # numpy/torch scalar
            return obj.item()
        if hasattr(obj, "tolist"):  # numpy array
            return obj.tolist()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return str(obj)
    
    return json.dumps(report, indent=2, default=default_serializer)


def segments_to_csv(segments: list[dict[str, Any]]) -> str:
    """Convert timeline segments to CSV string.
    
    Args:
        segments: List of segment dictionaries.
        
    Returns:
        CSV string with headers.
    """
    if not segments:
        return "start_sec,end_sec,emotion,confidence\n"
    
    lines = ["start_sec,end_sec,emotion,confidence"]
    for seg in segments:
        start = seg.get("start_sec", 0)
        end = seg.get("end_sec", 0)
        emotion = seg.get("emotion", "")
        confidence = seg.get("confidence", 0)
        lines.append(f"{start:.4f},{end:.4f},{emotion},{confidence:.4f}")
    
    return "\n".join(lines)
