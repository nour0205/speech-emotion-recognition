# Speech Emotion Recognition

Detect emotions from speech audio using a Wav2Vec2 model trained on IEMOCAP.

## 🎯 Supported Emotions

| Emotion | Code | Emoji |
|---------|------|-------|
| Angry   | ang  | 😠    |
| Happy   | hap  | 😊    |
| Sad     | sad  | 😢    |
| Neutral | neu  | 😐    |

## 📁 Project Structure

```
speech-emotion-recognition/
├── backend/                 # FastAPI backend
│   ├── Dockerfile
│   ├── main.py             # API entry point
│   ├── core/
│   │   └── model.py        # Emotion classifier
│   └── schemas/
│       └── emotion.py      # Pydantic models
├── frontend/               # Streamlit UI
│   ├── Dockerfile
│   ├── app.py             # Web interface
│   └── api_client.py      # Backend HTTP client
├── requirements/
│   ├── base.txt           # Core ML dependencies
│   ├── backend.txt        # FastAPI dependencies
│   ├── frontend.txt       # Streamlit dependencies
│   └── dev.txt            # Development tools
├── docker-compose.yml     # Container orchestration
└── Makefile               # Convenience commands
```

## 🚀 Quick Start

### Using Docker

```bash
# Start both services

# Or build fresh and start
docker compose up --build
```

Services will be available at:
- **Frontend:** http://localhost:8501
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

Stop with `Ctrl+C` or:
```bash
docker compose down
```

> **Note:** First run downloads the model (~360MB) which takes a few minutes. Subsequent runs use the cached model.

## 📡 API Endpoints

### `POST /api/v1/emotion/analyze`

Analyze emotion from audio file.

**Request:**
- Content-Type: `multipart/form-data`
- Body: WAV audio file

**Response:**
```json
{
  "label": "hap",
  "emotion": "Happy",
  "confidence": 0.92,
  "inference_time_sec": 0.156
}
```

### `GET /api/v1/emotions`

List supported emotions.

### `GET /health`

Health check endpoint.

## 📚 Model

Uses [speechbrain/emotion-recognition-wav2vec2-IEMOCAP](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP) from SpeechBrain.

## 🎤 Audio I/O Module

The `audioio` module provides a production-grade audio pipeline for loading, validating, and preprocessing audio files.

### Quick Start

```python
from audioio import load_validate_preprocess, AudioConfig

# Load and preprocess with default settings
waveform, sr = load_validate_preprocess("speech.wav")
# waveform shape: [1, num_samples] (mono, float32)
# sr: 16000 (default target sample rate)

# Load from bytes (e.g., uploaded file)
with open("speech.wav", "rb") as f:
    audio_bytes = f.read()
waveform, sr = load_validate_preprocess(audio_bytes)

# Custom configuration
config = AudioConfig(
    min_duration_sec=0.5,      # Minimum 0.5 seconds
    max_duration_sec=30.0,     # Maximum 30 seconds
    target_sample_rate=16000,  # Resample to 16kHz
    reject_silence=True,       # Reject silent audio
    silence_rms_threshold=1e-4,
    normalize=True,            # Peak normalize
    peak_target=0.95,          # Target peak amplitude
)
waveform, sr = load_validate_preprocess("speech.wav", config)
```

### Output Format

The pipeline always outputs:
- **Shape**: `[1, T]` — mono channel, T samples
- **Dtype**: `torch.float32`
- **Sample rate**: Configurable (default 16000 Hz)
- **Normalized**: Peak amplitude at 0.95 (configurable)

### Error Handling

The module raises structured exceptions with error codes:

```python
from audioio import load_validate_preprocess, AudioConfig
from audioio.errors import AudioDecodeError, AudioValidationError, AudioPreprocessError

try:
    waveform, sr = load_validate_preprocess("audio.wav")
except AudioDecodeError as e:
    print(f"[{e.code}] {e.message}")
    # e.details contains additional info
except AudioValidationError as e:
    print(f"[{e.code}] {e.message}")
except AudioPreprocessError as e:
    print(f"[{e.code}] {e.message}")
```

### Error Codes

| Code | Exception | Description |
|------|-----------|-------------|
| `FILE_NOT_FOUND` | AudioDecodeError | Audio file does not exist |
| `EMPTY_FILE` | AudioDecodeError | File has zero bytes |
| `INVALID_WAV` | AudioDecodeError | Cannot decode as WAV |
| `EMPTY_AUDIO` | AudioValidationError | Waveform has no samples |
| `TOO_SHORT` | AudioValidationError | Duration below minimum |
| `TOO_LONG` | AudioValidationError | Duration exceeds maximum |
| `INVALID_SAMPLE_RATE` | AudioValidationError | Sample rate outside 8kHz-192kHz |
| `TOO_MANY_CHANNELS` | AudioValidationError | More channels than allowed |
| `SILENCE` | AudioValidationError | Audio is near-silent |
| `NON_FINITE` | AudioValidationError | Contains NaN or Inf values |
| `INVALID_DTYPE` | AudioValidationError | Not a float tensor |
| `UNSUPPORTED_CHANNELS` | AudioPreprocessError | Cannot process >2 channels |

### Low-Level API

```python
from audioio import load_wav, load_wav_bytes, validate_wav, preprocess_audio

# Load only
waveform, sr = load_wav("audio.wav")  # or load_wav_bytes(bytes)

# Validate only
validate_wav(waveform, sr, min_duration_sec=0.1, reject_silence=True)

# Preprocess only
processed, target_sr = preprocess_audio(waveform, sr, target_sample_rate=16000)
```

## 📄 License

MIT
