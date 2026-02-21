"""Streamlit frontend for Speech Emotion Recognition.

This is the main entry point for the Streamlit UI. It uses the API client
to communicate with the backend and displays results using the UI components.

Run with:
    streamlit run apps/streamlit_app/app.py
    
Or via Docker:
    docker compose up frontend
"""

import os
import sys

# Add apps directory to path for imports when running as script
_apps_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _apps_dir not in sys.path:
    sys.path.insert(0, _apps_dir)

import streamlit as st
import httpx

from streamlit_app.api_client import get_client, EmotionAPIClient
from streamlit_app.ui.components import (
    display_emotion_result,
    display_api_status,
    render_footer,
    EMOTION_EMOJIS,
)

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="centered"
)

# Constants - use environment variable for Docker, fallback to localhost
API_URL = os.environ.get("API_URL", "http://localhost:8000")


def check_backend_health(client: EmotionAPIClient) -> bool:
    """Check if backend is running."""
    try:
        health = client.health_check()
        return health.get("status") == "healthy" and health.get("model_loaded", False)
    except httpx.ConnectError:
        return False
    except Exception:
        return False


def display_results(result: dict) -> None:
    """Display prediction results using UI components."""
    emotion = result.get("emotion", result.get("label", "Unknown"))
    confidence = result["confidence"]
    inference_time = result.get("inference_time_sec")
    
    display_emotion_result(
        emotion=emotion,
        confidence=confidence,
        inference_time=inference_time,
    )


def main():
    st.title("🎤 Speech Emotion Recognition")
    st.markdown("""
    Upload or record audio to detect emotions using a **Wav2Vec2** model 
    trained on the IEMOCAP dataset.
    """)
    
    # Initialize API client
    client = get_client(API_URL)
    
    # Check backend health
    with st.sidebar:
        st.markdown("### ⚙️ API Status")
        
        if st.button("🔄 Check Connection", use_container_width=True):
            st.session_state.pop("backend_healthy", None)
        
        if "backend_healthy" not in st.session_state:
            with st.spinner("Checking API..."):
                st.session_state.backend_healthy = check_backend_health(client)
        
        if st.session_state.backend_healthy:
            st.success("✅ Backend connected")
        else:
            st.error("❌ Backend not available")
            st.markdown("""
            **Start the backend:**
            ```bash
            make backend
            ```
            Or:
            ```bash
            uvicorn backend.main:app --reload
            ```
            """)
            return
    
    # Initialize session state for results
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_audio_hash" not in st.session_state:
        st.session_state.last_audio_hash = None
    
    # Tabs for input methods
    tab1, tab2 = st.tabs(["📁 Upload Audio", "🎙️ Record Audio"])
    
    with tab1:
        st.markdown("Upload a `.wav` audio file to analyze.")
        
        uploaded_file = st.file_uploader(
            "Choose a WAV file",
            type=["wav"],
            key="file_uploader",
            help="Upload a WAV audio file for emotion analysis"
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
            
            # Get audio bytes and compute hash for change detection
            audio_bytes = uploaded_file.getvalue()
            audio_hash = hash(audio_bytes)
            
            if st.button("🔍 Analyze Emotion", key="analyze_upload", type="primary"):
                with st.spinner("Analyzing audio..."):
                    try:
                        result = client.analyze_emotion(audio_bytes, uploaded_file.name)
                        st.session_state.last_result = result
                        st.session_state.last_audio_hash = audio_hash
                    except httpx.HTTPStatusError as e:
                        st.error(f"API Error: {e.response.text}")
                    except httpx.ConnectError:
                        st.error("Cannot connect to backend. Is it running?")
                        st.session_state.backend_healthy = False
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            # Display results if we have them for current audio
            if (st.session_state.last_result is not None and 
                st.session_state.last_audio_hash == audio_hash):
                display_results(st.session_state.last_result)
    
    with tab2:
        st.markdown("Record audio directly from your microphone.")
        
        audio_value = st.audio_input(
            "Click to record",
            key="audio_recorder",
            help="Record audio from your microphone"
        )
        
        if audio_value is not None:
            audio_bytes = audio_value.getvalue()
            audio_hash = hash(audio_bytes)
            
            if st.button("🔍 Analyze Emotion", key="analyze_record", type="primary"):
                with st.spinner("Analyzing audio..."):
                    try:
                        result = client.analyze_emotion(audio_bytes, "recording.wav")
                        st.session_state.last_result = result
                        st.session_state.last_audio_hash = audio_hash
                    except httpx.HTTPStatusError as e:
                        st.error(f"API Error: {e.response.text}")
                    except httpx.ConnectError:
                        st.error("Cannot connect to backend. Is it running?")
                        st.session_state.backend_healthy = False
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            # Display results if we have them for current audio
            if (st.session_state.last_result is not None and 
                st.session_state.last_audio_hash == audio_hash):
                display_results(st.session_state.last_result)
    
    # Footer with model information
    render_footer()


if __name__ == "__main__":
    main()
