"""Emotion classification model wrapper."""

import time
import tempfile
import os
from pathlib import Path

import torch
import torchaudio
from speechbrain.inference.interfaces import foreign_class


MODEL_SOURCE = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
SAMPLE_RATE = 16000


class EmotionClassifier:
    """Stateless emotion classifier using wav2vec2."""
    
    def __init__(self):
        """Load the pretrained model."""
        self.classifier = foreign_class(
            source=MODEL_SOURCE,
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
        )
    
    def _ensure_mono_16k(self, waveform: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        """Convert audio to mono 16kHz."""
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
            sr = SAMPLE_RATE
        
        return waveform, sr
    
    def predict_from_file(self, wav_path: str | Path) -> dict:
        """Run inference on an audio file.
        
        Args:
            wav_path: Path to WAV audio file.
            
        Returns:
            Dictionary with label, confidence, and inference time.
        """
        waveform, sr = torchaudio.load(str(wav_path))
        waveform, sr = self._ensure_mono_16k(waveform, sr)
        
        # SpeechBrain expects [batch, time] - clone to ensure fresh tensor
        wavs = waveform.squeeze(0).unsqueeze(0).clone()
        
        t0 = time.time()
        out_prob, score, index, text_lab = self.classifier.classify_batch(wavs)
        t1 = time.time()
        
        return {
            "label": str(text_lab[0]),
            "confidence": float(score[0]),
            "inference_time_sec": round(t1 - t0, 4),
        }
    
    def predict_from_bytes(self, audio_bytes: bytes) -> dict:
        """Run inference on raw audio bytes.
        
        Args:
            audio_bytes: Raw WAV file bytes.
            
        Returns:
            Dictionary with label, confidence, and inference time.
        """
        # Write to temp file for torchaudio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        try:
            return self.predict_from_file(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


# Singleton instance - loaded once at module import
_classifier: EmotionClassifier | None = None


def get_classifier() -> EmotionClassifier:
    """Get or create the singleton classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = EmotionClassifier()
    return _classifier
