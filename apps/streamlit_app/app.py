"""Streamlit frontend for Speech Emotion Recognition.

This is the main entry point for the Streamlit UI. It provides:
- Audio upload and recording
- Single prediction and timeline analysis modes
- Waveform and spectrogram visualization
- Detailed run reports with timings and logs
- Export capabilities (JSON report, CSV segments)

Run with:
    streamlit run apps/streamlit_app/app.py
    
Or via Docker:
    docker compose up frontend
"""

import io
import logging
import os
import sys
import traceback
from datetime import datetime

# Add apps directory to path for imports when running as script
_apps_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _apps_dir not in sys.path:
    sys.path.insert(0, _apps_dir)

# Add src directory to path for local inference
_src_dir = os.path.dirname(_apps_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import streamlit as st
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit

from streamlit_app.ui import (
    # Components
    render_config_sidebar,
    render_footer,
    display_single_prediction_result,
    EMOTION_EMOJIS,
    DEFAULT_CONFIG,
    # Logging
    LogCaptureContext,
    # Report
    TimingTracker,
    create_run_report,
    report_to_json,
    segments_to_csv,
    # Visualization
    compute_audio_stats,
    plot_waveform,
    plot_spectrogram,
    plot_timeline,
    plot_segments,
    display_audio_metrics,
    display_timings_table,
    display_segments_table,
    display_windows_table,
)

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)

# Setup logging for the app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_session_state() -> None:
    """Initialize session state variables."""
    defaults = {
        "audio_bytes": None,
        "audio_filename": None,
        "run_report": None,
        "logs": [],
        "config": dict(DEFAULT_CONFIG),
        "analysis_complete": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def run_single_prediction(
    audio_bytes: bytes,
    config: dict,
    tracker: TimingTracker,
) -> dict:
    """Run single-clip emotion prediction.
    
    Args:
        audio_bytes: Raw WAV audio bytes.
        config: Configuration dictionary.
        tracker: Timing tracker instance.
        
    Returns:
        Dict with prediction results.
    """
    from src.audioio import load_validate_preprocess, AudioConfig
    from src.model.infer import predict_clip
    
    # Load audio
    tracker.start("load")
    waveform, sample_rate = load_validate_preprocess(audio_bytes, AudioConfig())
    tracker.stop("load")
    
    # Run prediction
    tracker.start("infer")
    result = predict_clip(
        path_or_bytes=audio_bytes,
        model_id=config["model_id"],
        device=config["device"],
    )
    tracker.stop("infer")
    
    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
        "emotion": result.emotion,
        "confidence": float(result.confidence),
        "scores": dict(result.scores) if result.scores else None,
        "model_name": result.model_name,
        "duration_sec": float(result.duration_sec),
    }


def run_timeline_analysis(
    audio_bytes: bytes,
    config: dict,
    tracker: TimingTracker,
) -> dict:
    """Run timeline emotion analysis.
    
    Args:
        audio_bytes: Raw WAV audio bytes.
        config: Configuration dictionary.
        tracker: Timing tracker instance.
        
    Returns:
        Dict with timeline results.
    """
    from src.audioio import load_validate_preprocess, AudioConfig
    from src.timeline import (
        generate_timeline_from_waveform,
        WindowingConfig,
        SmoothingConfig,
    )
    
    # Load audio
    tracker.start("load")
    waveform, sample_rate = load_validate_preprocess(audio_bytes, AudioConfig())
    tracker.stop("load")
    
    # Build windowing config
    windowing_config = WindowingConfig(
        window_sec=config["window_sec"],
        hop_sec=config["hop_sec"],
        pad_mode=config["pad_mode"],
    )
    
    # Build smoothing config
    smoothing_config = SmoothingConfig(
        method=config["smoothing_method"],
        hysteresis_min_run=config.get("hysteresis_min_run", 3),
        majority_window=config.get("majority_window", 5),
        ema_alpha=config.get("ema_alpha", 0.6),
    )
    
    # Generate timeline
    tracker.start("infer")
    result = generate_timeline_from_waveform(
        waveform=waveform,
        sample_rate=sample_rate,
        windowing_config=windowing_config,
        model_id=config["model_id"],
        device=config["device"],
        smoothing_config=smoothing_config,
        include_windows=config.get("include_windows", False),
        include_scores=config.get("include_scores", True),
    )
    tracker.stop("infer")
    
    # Convert to dict
    result_dict = result.to_dict(
        include_windows=config.get("include_windows", False),
        include_scores=config.get("include_scores", True),
    )
    result_dict["waveform"] = waveform
    result_dict["sample_rate"] = sample_rate
    
    return result_dict


def run_analysis(audio_bytes: bytes, config: dict) -> dict:
    """Run the full analysis pipeline and create a RunReport.
    
    Args:
        audio_bytes: Raw WAV audio bytes.
        config: Configuration dictionary.
        
    Returns:
        RunReport dictionary.
    """
    tracker = TimingTracker()
    logs = []
    error = None
    results = None
    audio_stats = None
    
    try:
        # Capture logs during analysis
        with LogCaptureContext() as log_ctx:
            if config["mode"] == "single":
                results = run_single_prediction(audio_bytes, config, tracker)
            else:
                results = run_timeline_analysis(audio_bytes, config, tracker)
            
            logs = list(log_ctx.logs)
        
        # Compute audio stats
        if results and "waveform" in results:
            audio_stats = compute_audio_stats(
                results["waveform"],
                results.get("sample_rate", 16000),
            )
            # Remove waveform from results (keep separately)
            waveform = results.pop("waveform", None)
            sample_rate = results.get("sample_rate", 16000)
            
            # Store for visualization
            st.session_state["_waveform"] = waveform
            st.session_state["_sample_rate"] = sample_rate
    
    except Exception as e:
        error = e
        logs.append(f"ERROR: {str(e)}")
        logger.exception("Analysis failed")
    
    # Create report
    report = create_run_report(
        config=config,
        audio_stats=audio_stats,
        timings=tracker.get_timings(),
        logs=logs,
        results=results,
        error=error,
    )
    
    report["timestamp_end"] = datetime.now().isoformat()
    
    return report


def render_input_section() -> bytes | None:
    """Render audio input section and return audio bytes if available."""
    st.markdown("### 📥 Audio Input")
    
    tab1, tab2 = st.tabs(["📁 Upload File", "🎙️ Record Audio"])
    
    audio_bytes = None
    filename = "audio.wav"
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose a WAV file",
            type=["wav"],
            key="file_uploader",
            help="Upload a WAV audio file for emotion analysis",
        )
        
        if uploaded_file is not None:
            audio_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name
            st.audio(uploaded_file, format="audio/wav")
    
    with tab2:
        audio_value = st.audio_input(
            "Click to record",
            key="audio_recorder",
            help="Record audio from your microphone",
        )
        
        if audio_value is not None:
            audio_bytes = audio_value.getvalue()
            filename = "recording.wav"
    
    if audio_bytes:
        st.session_state["audio_bytes"] = audio_bytes
        st.session_state["audio_filename"] = filename
    
    return audio_bytes


def render_results_section(report: dict) -> None:
    """Render the analysis results section."""
    if report.get("error"):
        st.error(f"Analysis failed: {report['error']}")
        with st.expander("Error Details", expanded=False):
            if report.get("error_traceback"):
                st.code(report["error_traceback"])
        return
    
    results = report.get("results", {})
    config = report.get("config", {})
    mode = config.get("mode", "single")
    
    # Audio stats first
    audio_stats = report.get("audio_stats", {})
    if audio_stats:
        st.markdown("### 📊 Audio Metrics")
        display_audio_metrics(audio_stats)
        st.markdown("")
    
    # Waveform and Spectrogram - stacked vertically with full width
    waveform = st.session_state.get("_waveform")
    sample_rate = st.session_state.get("_sample_rate", 16000)
    
    if waveform is not None:
        st.markdown("### 🎵 Waveform")
        fig = plot_waveform(waveform, sample_rate)
        st.pyplot(fig, use_container_width=True)
        st.markdown("")
        
        st.markdown("### 📈 Spectrogram")
        fig = plot_spectrogram(waveform, sample_rate)
        st.pyplot(fig, use_container_width=True)
        st.markdown("")
    
    st.markdown("---")
    st.markdown("")
    
    # Results based on mode
    if mode == "single":
        render_single_results(results, config)
    else:
        render_timeline_results(results, config)


def render_single_results(results: dict, config: dict) -> None:
    """Render single prediction results."""
    emotion = results.get("emotion", "Unknown")
    confidence = results.get("confidence", 0)
    scores = results.get("scores")
    
    display_single_prediction_result(
        emotion=emotion,
        confidence=confidence,
        scores=scores if config.get("include_scores") else None,
    )


def render_timeline_results(results: dict, config: dict) -> None:
    """Render timeline analysis results."""
    st.markdown("### 🎯 Timeline Results")
    st.markdown("")
    
    segments = results.get("segments", [])
    windows = results.get("windows", [])
    duration_sec = results.get("duration_sec", 0)
    
    # Segments visualization
    if segments:
        st.markdown("#### Emotion Segments")
        
        # Segment bar chart
        fig = plot_segments(segments, duration_sec)
        st.pyplot(fig, use_container_width=True)
        st.markdown("")
        
        # Segments table
        with st.expander("📋 Segments Table", expanded=True):
            display_segments_table(segments)
        st.markdown("")
    
    # Timeline plot (if we have windows)
    if windows:
        st.markdown("#### Timeline Plot")
        fig = plot_timeline(windows, duration_sec)
        st.pyplot(fig, use_container_width=True)
        st.markdown("")
    elif segments:
        # Create windows from segments for timeline plot
        pseudo_windows = []
        for i, seg in enumerate(segments):
            pseudo_windows.append({
                "index": i,
                "start_sec": seg.get("start_sec", 0),
                "end_sec": seg.get("end_sec", 0),
                "emotion": seg.get("emotion", ""),
            })
        if pseudo_windows:
            st.markdown("#### Timeline Plot")
            fig = plot_timeline(pseudo_windows, duration_sec)
            st.pyplot(fig, use_container_width=True)
            st.markdown("")
    
    # Window predictions table
    if config.get("include_windows") and windows:
        with st.expander("🔍 Window Predictions", expanded=False):
            display_windows_table(windows)


def render_performance_section(report: dict) -> None:
    """Render performance metrics section."""
    with st.expander("⚡ Performance", expanded=False):
        timings = report.get("timings", {})
        if timings:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                display_timings_table(timings)
            
            with col2:
                total_ms = timings.get("total_ms", 0)
                st.metric("Total Time", f"{total_ms:.0f} ms")


def render_logs_section(report: dict) -> None:
    """Render captured logs section."""
    logs = report.get("logs", [])
    
    with st.expander(f"📋 Logs ({len(logs)} entries)", expanded=False):
        if logs:
            for log in logs:
                st.text(log)
        else:
            st.info("No logs captured during this run.")


def render_export_section(report: dict) -> None:
    """Render export buttons section."""
    st.markdown("### 📤 Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export RunReport as JSON
        json_str = report_to_json(report)
        st.download_button(
            label="📥 Download Report (JSON)",
            data=json_str,
            file_name=f"run_report_{report.get('run_id', 'unknown')[:8]}.json",
            mime="application/json",
        )
    
    with col2:
        # Export segments as CSV (if timeline mode)
        config = report.get("config", {})
        results = report.get("results", {})
        
        if config.get("mode") == "timeline" and results.get("segments"):
            csv_str = segments_to_csv(results["segments"])
            st.download_button(
                label="📥 Download Segments (CSV)",
                data=csv_str,
                file_name=f"segments_{report.get('run_id', 'unknown')[:8]}.csv",
                mime="text/csv",
            )
        else:
            st.button("📥 Download Segments (CSV)", disabled=True, help="Only available in timeline mode")
    
    with col3:
        # Copy run ID
        st.code(f"Run ID: {report.get('run_id', 'N/A')[:8]}")


def main():
    """Main application entry point."""
    init_session_state()
    
    # Title
    st.title("🎤 Speech Emotion Recognition")
    st.markdown("""
    Analyze emotions in speech using a **Wav2Vec2** model trained on the IEMOCAP dataset.
    Upload or record audio, configure analysis parameters, and view detailed results.
    """)
    
    # Sidebar configuration
    config = render_config_sidebar()
    st.session_state["config"] = config
    
    # Full-width input section at top
    st.markdown("")
    audio_bytes = render_input_section()
    
    # Analysis button
    if audio_bytes:
        st.markdown("")
        if st.button("🔍 Analyze", type="primary", use_container_width=False):
            with st.spinner("Analyzing audio..."):
                report = run_analysis(audio_bytes, config)
                st.session_state["run_report"] = report
                st.session_state["analysis_complete"] = True
    
    # Results section
    report = st.session_state.get("run_report")
    
    if report:
        st.markdown("")
        st.markdown("---")
        st.markdown("")
        
        render_results_section(report)
        
        # Performance and logs in columns
        st.markdown("")
        st.markdown("---")
        st.markdown("### ⚡ Performance & Logs")
        st.markdown("")
        
        col_perf, col_logs = st.columns(2)
        
        with col_perf:
            render_performance_section(report)
        
        with col_logs:
            render_logs_section(report)
        
        st.markdown("")
        render_export_section(report)
    elif audio_bytes:
        st.info("👆 Click **Analyze** to process the audio.")
    
    # Footer
    st.markdown("")
    render_footer()


if __name__ == "__main__":
    main()
