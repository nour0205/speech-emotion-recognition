"""Reusable UI components for the Streamlit Speech Emotion Recognition app.

This module provides modular, reusable UI widgets that can be composed
to build the main application interface.
"""

import streamlit as st
from typing import Any, Optional

# Emotion emoji mappings
EMOTION_EMOJIS = {
    "Angry": "😠",
    "Happy": "😊",
    "Sad": "😢",
    "Neutral": "😐",
    "angry": "😠",
    "happy": "😊",
    "sad": "😢",
    "neutral": "😐",
}

# Default configuration values
DEFAULT_CONFIG = {
    "model_id": "baseline",
    "device": "cpu",
    "mode": "single",  # "single" or "timeline"
    "window_sec": 2.0,
    "hop_sec": 0.5,
    "pad_mode": "zero",
    "smoothing_method": "none",
    "hysteresis_min_run": 3,
    "majority_window": 5,
    "ema_alpha": 0.6,
    "include_scores": True,
    "include_windows": False,
}


def display_emotion_result(
    emotion: str,
    confidence: float,
    inference_time: Optional[float] = None,
) -> None:
    """Display emotion prediction results with formatting.
    
    Args:
        emotion: The predicted emotion label.
        confidence: Confidence score (0-1).
        inference_time: Optional inference time in seconds.
    """
    emoji = EMOTION_EMOJIS.get(emotion, EMOTION_EMOJIS.get(emotion.capitalize(), "🎭"))
    
    st.markdown("---")
    st.markdown("### 🎯 Results")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(
            f"<h1 style='text-align: center; font-size: 80px;'>{emoji}</h1>", 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(f"**Detected Emotion:** {emotion.capitalize()}")
        st.progress(confidence, text=f"Confidence: {confidence:.1%}")
        if inference_time is not None:
            st.caption(f"⏱️ Inference time: {inference_time:.3f}s")


def display_error(message: str, suggestion: Optional[str] = None) -> None:
    """Display an error message with optional suggestion.
    
    Args:
        message: The error message.
        suggestion: Optional suggestion for fixing the error.
    """
    st.error(f"❌ {message}")
    if suggestion:
        st.info(f"💡 {suggestion}")


def display_api_status(is_healthy: bool, api_url: str) -> None:
    """Display API connection status in the sidebar.
    
    Args:
        is_healthy: Whether the API is healthy and connected.
        api_url: The API URL being used.
    """
    if is_healthy:
        st.success("✅ Backend connected")
        st.caption(f"API: {api_url}")
    else:
        st.error("❌ Backend not available")
        st.markdown("""
        **Start the backend:**
        ```bash
        # Using Docker
        docker compose up api
        
        # Or locally
        make api
        ```
        """)


def render_audio_uploader(
    key: str = "file_uploader",
    label: str = "Choose a WAV file",
) -> Optional[bytes]:
    """Render an audio file uploader widget.
    
    Args:
        key: Unique key for the widget.
        label: Label text for the uploader.
        
    Returns:
        Audio bytes if a file was uploaded, None otherwise.
    """
    st.markdown("Upload a `.wav` audio file to analyze.")
    
    uploaded_file = st.file_uploader(
        label,
        type=["wav"],
        key=key,
        help="Upload a WAV audio file for emotion analysis"
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        return uploaded_file.getvalue()
    
    return None


def render_audio_recorder(key: str = "audio_recorder") -> Optional[bytes]:
    """Render an audio recorder widget.
    
    Args:
        key: Unique key for the widget.
        
    Returns:
        Audio bytes if recording was made, None otherwise.
    """
    st.markdown("Record audio directly from your microphone.")
    
    audio_value = st.audio_input(
        "Click to record",
        key=key,
        help="Record audio from your microphone"
    )
    
    if audio_value is not None:
        return audio_value.getvalue()
    
    return None


def render_footer() -> None:
    """Render the application footer with model information."""
    st.markdown("---")
    st.caption("""
    **Model:** [speechbrain/emotion-recognition-wav2vec2-IEMOCAP](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP)  
    **Emotions detected:** Angry, Happy, Sad, Neutral
    """)


def render_config_sidebar() -> dict[str, Any]:
    """Render configuration sidebar and return config dict.
    
    Returns:
        Dictionary with all configuration values.
    """
    config = dict(DEFAULT_CONFIG)
    
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Model settings
    st.sidebar.markdown("### Model")
    config["model_id"] = st.sidebar.selectbox(
        "Model ID",
        options=["baseline"],
        index=0,
        help="Model to use for inference",
    )
    
    config["device"] = st.sidebar.selectbox(
        "Device",
        options=["cpu", "cuda"],
        index=0,
        help="Device for inference (cuda requires GPU)",
    )
    
    # Mode toggle
    st.sidebar.markdown("### Analysis Mode")
    mode = st.sidebar.radio(
        "Mode",
        options=["Single Prediction", "Timeline"],
        index=0,
        horizontal=True,
        help="Single prediction analyzes entire clip; Timeline segments the audio",
    )
    config["mode"] = "single" if mode == "Single Prediction" else "timeline"
    
    # Timeline parameters (only show if timeline mode)
    if config["mode"] == "timeline":
        st.sidebar.markdown("### Timeline Parameters")
        
        config["window_sec"] = st.sidebar.slider(
            "Window (sec)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.25,
            help="Window duration for segmentation",
        )
        
        config["hop_sec"] = st.sidebar.slider(
            "Hop (sec)",
            min_value=0.1,
            max_value=config["window_sec"],
            value=min(0.5, config["window_sec"]),
            step=0.1,
            help="Hop/stride between windows",
        )
        
        config["pad_mode"] = st.sidebar.selectbox(
            "Padding Mode",
            options=["zero", "reflect", "none"],
            index=0,
            help="How to pad partial windows at end",
        )
        
        # Smoothing parameters
        st.sidebar.markdown("### Smoothing")
        
        config["smoothing_method"] = st.sidebar.selectbox(
            "Smoothing Method",
            options=["none", "majority", "hysteresis", "ema"],
            index=0,
            help="Method for smoothing window predictions",
        )
        
        if config["smoothing_method"] == "hysteresis":
            config["hysteresis_min_run"] = st.sidebar.slider(
                "Hysteresis Min Run",
                min_value=1,
                max_value=10,
                value=3,
                help="Minimum consecutive windows for emotion switch",
            )
        elif config["smoothing_method"] == "majority":
            config["majority_window"] = st.sidebar.slider(
                "Majority Window",
                min_value=3,
                max_value=11,
                value=5,
                step=2,
                help="Window size for majority voting (must be odd)",
            )
        elif config["smoothing_method"] == "ema":
            config["ema_alpha"] = st.sidebar.slider(
                "EMA Alpha",
                min_value=0.1,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="EMA coefficient (higher = more weight to recent)",
            )
    
    # Output options
    st.sidebar.markdown("### Output Options")
    
    config["include_scores"] = st.sidebar.checkbox(
        "Include Scores",
        value=True,
        help="Include per-emotion probability scores",
    )
    
    if config["mode"] == "timeline":
        config["include_windows"] = st.sidebar.checkbox(
            "Include Windows",
            value=False,
            help="Include per-window predictions in output",
        )
    
    return config


def display_single_prediction_result(
    emotion: str,
    confidence: float,
    scores: Optional[dict[str, float]] = None,
    inference_time: Optional[float] = None,
) -> None:
    """Display single prediction results with formatting.
    
    Args:
        emotion: The predicted emotion label.
        confidence: Confidence score (0-1).
        scores: Optional per-emotion scores.
        inference_time: Optional inference time in seconds.
    """
    emoji = EMOTION_EMOJIS.get(emotion, EMOTION_EMOJIS.get(emotion.capitalize(), "🎭"))
    
    st.markdown("### 🎯 Prediction Result")
    st.markdown("")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown(
            f"<h1 style='text-align: center; font-size: 80px;'>{emoji}</h1>", 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(f"**Detected Emotion:** {emotion.capitalize()}")
        st.progress(confidence, text=f"Confidence: {confidence:.1%}")
        if inference_time is not None:
            st.caption(f"⏱️ Inference time: {inference_time:.3f}s")
    
    st.markdown("")
    
    # Show bar chart if scores available
    if scores:
        st.markdown("#### Emotion Scores")
        from .visualization import plot_emotion_bar_chart
        fig = plot_emotion_bar_chart(scores, emotion)
        st.pyplot(fig, use_container_width=True)

