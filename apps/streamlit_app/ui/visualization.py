"""Visualization components for audio analysis.

This module provides:
- Waveform plotting
- Spectrogram visualization (STFT magnitude)
- Results visualization (bar charts, timeline plots)
- Audio summary metrics display
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch


# Emotion color mapping for visualizations
EMOTION_COLORS = {
    "angry": "#FF4444",
    "happy": "#44DD44",
    "sad": "#4444FF",
    "neutral": "#888888",
    "Angry": "#FF4444",
    "Happy": "#44DD44",
    "Sad": "#4444FF",
    "Neutral": "#888888",
}

EMOTION_INDEX = {
    "angry": 0,
    "happy": 1,
    "sad": 2,
    "neutral": 3,
}


def compute_audio_stats(
    waveform: torch.Tensor | np.ndarray,
    sample_rate: int,
    silence_threshold: float = 0.01,
) -> dict[str, float]:
    """Compute audio statistics from waveform.
    
    Args:
        waveform: Audio tensor [1, T] or [T] or numpy array.
        sample_rate: Sample rate in Hz.
        silence_threshold: RMS threshold for silence detection.
        
    Returns:
        Dict with: duration_sec, sample_rate, rms, peak, silence_ratio
    """
    # Convert to numpy if needed
    if isinstance(waveform, torch.Tensor):
        data = waveform.numpy()
    else:
        data = np.asarray(waveform)
    
    # Flatten if needed
    data = data.flatten().astype(np.float32)
    
    # Duration
    duration_sec = len(data) / sample_rate if sample_rate > 0 else 0.0
    
    # RMS
    rms = float(np.sqrt(np.mean(data ** 2)))
    
    # Peak
    peak = float(np.max(np.abs(data))) if len(data) > 0 else 0.0
    
    # Silence ratio using windowed RMS
    if len(data) > 0:
        # Use 25ms frames
        frame_size = int(0.025 * sample_rate)
        if frame_size < 1:
            frame_size = 1
        
        num_frames = len(data) // frame_size
        if num_frames > 0:
            frames = data[:num_frames * frame_size].reshape(num_frames, frame_size)
            frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
            silence_ratio = float(np.mean(frame_rms < silence_threshold))
        else:
            silence_ratio = 1.0 if rms < silence_threshold else 0.0
    else:
        silence_ratio = 1.0
    
    return {
        "duration_sec": round(duration_sec, 3),
        "sample_rate": sample_rate,
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "silence_ratio": round(silence_ratio, 3),
    }


def plot_waveform(
    waveform: torch.Tensor | np.ndarray,
    sample_rate: int,
    title: str = "Waveform",
    figsize: tuple[int, int] = (14, 4),
) -> plt.Figure:
    """Create waveform plot.
    
    Args:
        waveform: Audio tensor [1, T] or [T] or numpy array.
        sample_rate: Sample rate in Hz.
        title: Plot title.
        figsize: Figure size (width, height).
        
    Returns:
        Matplotlib figure.
    """
    # Convert to numpy
    if isinstance(waveform, torch.Tensor):
        data = waveform.numpy()
    else:
        data = np.asarray(waveform)
    
    data = data.flatten()
    
    # Time axis
    time_axis = np.arange(len(data)) / sample_rate
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_axis, data, linewidth=0.5, color="#1f77b4")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim(0, time_axis[-1] if len(time_axis) > 0 else 1)
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_spectrogram(
    waveform: torch.Tensor | np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    title: str = "Spectrogram",
    figsize: tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Create log-magnitude spectrogram plot using STFT.
    
    Args:
        waveform: Audio tensor [1, T] or [T] or numpy array.
        sample_rate: Sample rate in Hz.
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        title: Plot title.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    # Convert to numpy
    if isinstance(waveform, torch.Tensor):
        data = waveform.numpy()
    else:
        data = np.asarray(waveform)
    
    data = data.flatten().astype(np.float32)
    
    # Compute STFT using numpy
    # Apply window, compute FFT, take magnitude
    from scipy.signal import stft as scipy_stft
    
    freqs, times, Zxx = scipy_stft(
        data,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        window="hann",
    )
    
    # Log magnitude
    magnitude = np.abs(Zxx)
    log_mag = 20 * np.log10(magnitude + 1e-10)
    
    # Clip to reasonable range
    log_mag = np.clip(log_mag, -80, 0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    mesh = ax.pcolormesh(
        times,
        freqs,
        log_mag,
        shading="gouraud",
        cmap="magma",
    )
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.set_ylim(0, sample_rate // 2)  # Nyquist
    
    cbar = fig.colorbar(mesh, ax=ax, label="dB")
    
    plt.tight_layout()
    return fig


def plot_emotion_bar_chart(
    scores: dict[str, float],
    predicted_emotion: str,
    title: str = "Emotion Scores",
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Create bar chart of emotion scores.
    
    Args:
        scores: Dict mapping emotion labels to confidence scores.
        predicted_emotion: The predicted emotion (highlighted).
        title: Plot title.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    emotions = list(scores.keys())
    values = [scores[e] for e in emotions]
    
    # Colors - highlight predicted emotion
    colors = []
    for e in emotions:
        if e.lower() == predicted_emotion.lower():
            colors.append(EMOTION_COLORS.get(e, "#4CAF50"))
        else:
            colors.append("#CCCCCC")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.barh(emotions, values, color=colors, edgecolor="black", linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}",
            va="center",
            fontsize=10,
        )
    
    ax.set_xlabel("Confidence")
    ax.set_title(title)
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis="x", alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_timeline(
    windows: list[dict[str, Any]],
    duration_sec: float,
    title: str = "Emotion Timeline",
    figsize: tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Create timeline plot showing emotion over time.
    
    Args:
        windows: List of window prediction dicts with start_sec, emotion.
        duration_sec: Total audio duration.
        title: Plot title.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    if not windows:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.text(0.5, 0.5, "No windows", ha="center", va="center", transform=ax.transAxes)
        return fig
    
    # Get emotion labels and assign indices
    all_emotions = sorted(set(w.get("emotion", "unknown").lower() for w in windows))
    emotion_to_idx = {e: i for i, e in enumerate(all_emotions)}
    
    # Prepare data
    times = []
    indices = []
    colors = []
    
    for w in windows:
        start = w.get("start_sec", 0)
        end = w.get("end_sec", start)
        mid = (start + end) / 2
        emotion = w.get("emotion", "unknown").lower()
        
        times.append(mid)
        indices.append(emotion_to_idx.get(emotion, 0))
        colors.append(EMOTION_COLORS.get(emotion, "#888888"))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot as step/scatter
    ax.scatter(times, indices, c=colors, s=80, edgecolors="black", linewidths=0.5)
    
    # Connect with lines
    ax.step(times, indices, where="mid", color="#666666", alpha=0.5, linestyle="--")
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Emotion")
    ax.set_title(title)
    ax.set_xlim(0, duration_sec)
    ax.set_ylim(-0.5, len(all_emotions) - 0.5)
    ax.set_yticks(range(len(all_emotions)))
    ax.set_yticklabels([e.capitalize() for e in all_emotions])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_segments(
    segments: list[dict[str, Any]],
    duration_sec: float,
    title: str = "Emotion Segments",
    figsize: tuple[int, int] = (14, 3),
) -> plt.Figure:
    """Create horizontal bar chart showing emotion segments over time.
    
    Args:
        segments: List of segment dicts with start_sec, end_sec, emotion.
        duration_sec: Total audio duration.
        title: Plot title.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for seg in segments:
        start = seg.get("start_sec", 0)
        end = seg.get("end_sec", start)
        emotion = seg.get("emotion", "unknown")
        color = EMOTION_COLORS.get(emotion, EMOTION_COLORS.get(emotion.lower(), "#888888"))
        
        ax.barh(
            y=0,
            width=end - start,
            left=start,
            height=0.5,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
        
        # Label if segment is wide enough
        if end - start > duration_sec * 0.05:
            ax.text(
                (start + end) / 2,
                0,
                emotion.capitalize(),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
    
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.set_xlim(0, duration_sec)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.3)
    
    plt.tight_layout()
    return fig


def display_audio_metrics(stats: dict[str, float]) -> None:
    """Display audio metrics in Streamlit columns.
    
    Args:
        stats: Dict with audio statistics.
    """
    # Use 4 columns for better spacing
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        duration = stats.get('duration_sec', 0)
        st.metric("Duration", f"{duration:.2f} sec")
    
    with col2:
        sr = stats.get('sample_rate', 0)
        st.metric("Sample Rate", f"{sr:,} Hz")
    
    with col3:
        rms = stats.get('rms', 0)
        peak = stats.get('peak', 0)
        st.metric("RMS / Peak", f"{rms:.3f} / {peak:.3f}")
    
    with col4:
        silence = stats.get("silence_ratio", 0) * 100
        st.metric("Silence Ratio", f"{silence:.1f}%")


def display_timings_table(timings: dict[str, float]) -> None:
    """Display timings as a table.
    
    Args:
        timings: Dict of step names to milliseconds.
    """
    import pandas as pd
    
    rows = []
    for key, value in timings.items():
        if key == "total_ms":
            continue
        step_name = key.replace("_ms", "").replace("_", " ").title()
        rows.append({"Step": step_name, "Time (ms)": f"{value:.2f}"})
    
    # Add total at the end
    if "total_ms" in timings:
        rows.append({"Step": "Total", "Time (ms)": f"{timings['total_ms']:.2f}"})
    
    if rows:
        df = pd.DataFrame(rows)
        st.table(df)
    else:
        st.info("No timing data available.")


def display_segments_table(segments: list[dict[str, Any]]) -> None:
    """Display segments as a table.
    
    Args:
        segments: List of segment dicts.
    """
    import pandas as pd
    
    if not segments:
        st.info("No segments to display.")
        return
    
    rows = []
    for i, seg in enumerate(segments):
        rows.append({
            "#": i + 1,
            "Start (s)": f"{seg.get('start_sec', 0):.3f}",
            "End (s)": f"{seg.get('end_sec', 0):.3f}",
            "Emotion": seg.get("emotion", "").capitalize(),
            "Confidence": f"{seg.get('confidence', 0):.1%}",
        })
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def display_windows_table(windows: list[dict[str, Any]]) -> None:
    """Display window predictions as a dataframe.
    
    Args:
        windows: List of window prediction dicts.
    """
    import pandas as pd
    
    if not windows:
        st.info("No window predictions to display.")
        return
    
    rows = []
    for w in windows:
        row = {
            "Index": w.get("index", 0),
            "Start (s)": f"{w.get('start_sec', 0):.3f}",
            "End (s)": f"{w.get('end_sec', 0):.3f}",
            "Emotion": w.get("emotion", "").capitalize(),
            "Confidence": f"{w.get('confidence', 0):.1%}",
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
