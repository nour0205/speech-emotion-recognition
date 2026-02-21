"""Reusable UI components for the Streamlit Speech Emotion Recognition app.

This module provides modular, reusable UI widgets that can be composed
to build the main application interface.
"""

import streamlit as st
from typing import Optional

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
