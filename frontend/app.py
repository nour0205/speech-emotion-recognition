"""Streamlit frontend for Speech Emotion Recognition."""

import os
import streamlit as st
import httpx

from api_client import get_client, EmotionAPIClient

# Page config - must be first
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="centered"
)

# Constants - use environment variable for Docker, fallback to localhost
API_URL = os.environ.get("API_URL", "http://localhost:8000")

EMOTION_EMOJIS = {
    "Angry": "😠",
    "Happy": "😊",
    "Sad": "😢",
    "Neutral": "😐",
}


def check_backend_health(client: EmotionAPIClient) -> bool:
    """Check if backend is running."""
    try:
        health = client.health_check()
        return health.get("status") == "healthy" and health.get("model_loaded", False)
    except httpx.ConnectError:
        return False
    except Exception:
        return False


def display_results(result: dict):
    """Display prediction results with nice formatting."""
    emotion = result.get("emotion", result.get("label", "Unknown"))
    confidence = result["confidence"]
    inference_time = result["inference_time_sec"]
    
    emoji = EMOTION_EMOJIS.get(emotion, "🎭")
    
    st.markdown("---")
    st.markdown("### 🎯 Results")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(
            f"<h1 style='text-align: center; font-size: 80px;'>{emoji}</h1>", 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(f"**Detected Emotion:** {emotion}")
        st.progress(confidence, text=f"Confidence: {confidence:.1%}")
        st.caption(f"⏱️ Inference time: {inference_time:.3f}s")


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
    
    # Footer
    st.markdown("---")
    st.caption("""
    **Model:** [speechbrain/emotion-recognition-wav2vec2-IEMOCAP](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP)  
    **Emotions detected:** Angry, Happy, Sad, Neutral
    """)


if __name__ == "__main__":
    main()
