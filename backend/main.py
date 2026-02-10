"""FastAPI backend for Speech Emotion Recognition."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.core.model import EmotionClassifier
from backend.schemas.emotion import EmotionResponse, get_emotion_name


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global classifier instance
classifier: EmotionClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global classifier
    
    logger.info("Loading emotion recognition model...")
    classifier = EmotionClassifier()
    logger.info("Model loaded successfully!")
    
    yield
    
    logger.info("Shutting down...")
    classifier = None


app = FastAPI(
    title="Speech Emotion Recognition API",
    description="Detect emotions from audio using Wav2Vec2 trained on IEMOCAP",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware - allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None
    }


@app.post("/api/v1/emotion/analyze", response_model=EmotionResponse)
async def analyze_emotion(file: UploadFile = File(..., description="WAV audio file")):
    """Analyze emotion from uploaded audio file.
    
    Accepts a WAV audio file and returns the predicted emotion with confidence score.
    Audio is automatically converted to mono 16kHz if needed.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400, 
            detail="Only WAV files are supported. Please upload a .wav file."
        )
    
    try:
        # Read file bytes
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Run prediction
        result = classifier.predict_from_bytes(audio_bytes)
        
        # Add human-readable emotion name
        result["emotion"] = get_emotion_name(result["label"])
        
        return EmotionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing audio")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@app.get("/api/v1/emotions")
async def list_emotions():
    """List all supported emotions."""
    return {
        "emotions": [
            {"code": "ang", "name": "Angry", "emoji": "😠"},
            {"code": "hap", "name": "Happy", "emoji": "😊"},
            {"code": "sad", "name": "Sad", "emoji": "😢"},
            {"code": "neu", "name": "Neutral", "emoji": "😐"},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
