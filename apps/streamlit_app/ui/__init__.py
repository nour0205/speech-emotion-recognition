"""UI components package for Streamlit app."""

from .components import (
    display_emotion_result,
    display_error,
    display_api_status,
    display_single_prediction_result,
    render_audio_uploader,
    render_audio_recorder,
    render_config_sidebar,
    render_footer,
    EMOTION_EMOJIS,
    DEFAULT_CONFIG,
)

from .logging_handler import (
    SessionStateLogHandler,
    LogCaptureContext,
)

from .report import (
    RunReport,
    AudioStats,
    TimingInfo,
    ConfigSnapshot,
    TimingTracker,
    create_run_report,
    report_to_json,
    segments_to_csv,
)

from .visualization import (
    compute_audio_stats,
    plot_waveform,
    plot_spectrogram,
    plot_emotion_bar_chart,
    plot_timeline,
    plot_segments,
    display_audio_metrics,
    display_timings_table,
    display_segments_table,
    display_windows_table,
)

__all__ = [
    # Components
    "display_emotion_result",
    "display_error",
    "display_api_status",
    "display_single_prediction_result",
    "render_audio_uploader",
    "render_audio_recorder",
    "render_config_sidebar",
    "render_footer",
    "EMOTION_EMOJIS",
    "DEFAULT_CONFIG",
    # Logging
    "SessionStateLogHandler",
    "LogCaptureContext",
    # Report
    "RunReport",
    "AudioStats",
    "TimingInfo",
    "ConfigSnapshot",
    "TimingTracker",
    "create_run_report",
    "report_to_json",
    "segments_to_csv",
    # Visualization
    "compute_audio_stats",
    "plot_waveform",
    "plot_spectrogram",
    "plot_emotion_bar_chart",
    "plot_timeline",
    "plot_segments",
    "display_audio_metrics",
    "display_timings_table",
    "display_segments_table",
    "display_windows_table",
]

