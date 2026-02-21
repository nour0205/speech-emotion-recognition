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

- **Frontend:** <http://localhost:8501>
- **Backend API:** <http://localhost:8000>
- **API Docs:** <http://localhost:8000/docs>

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

## ⏱️ Timeline Windowing Module

The `timeline` module provides deterministic audio windowing/segmentation for time-series emotion inference. It takes preprocessed audio from `audioio` and segments it into overlapping windows with precise timestamps.

### Quick Start

```python
import torch
from timeline import WindowingConfig, segment_audio

# Example waveform from audioio (shape [1, T], 16kHz)
waveform = torch.randn(1, 48000)  # 3 seconds

# Configure windowing
config = WindowingConfig(
    window_sec=2.0,        # 2-second windows
    hop_sec=0.5,           # 0.5-second hop (75% overlap)
    pad_mode="zero",       # Pad last window with zeros
)

# Segment audio
windows = segment_audio(waveform, sample_rate=16000, config=config)

for w in windows:
    print(f"Window {w['index']}: {w['start_sec']:.2f}s - {w['end_sec']:.2f}s")
# Output:
# Window 0: 0.00s - 2.00s
# Window 1: 0.50s - 2.50s
# Window 2: 1.00s - 3.00s
```

### WindowingConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_sec` | float | 2.0 | Window duration in seconds |
| `hop_sec` | float | 0.5 | Hop/stride duration in seconds |
| `pad_mode` | str | "zero" | Padding mode: "none", "zero", or "reflect" |
| `include_partial_last_window` | bool | True | Include partial window at end |
| `min_window_sec` | float | 0.25 | Minimum allowed window size |
| `max_window_sec` | float | 10.0 | Maximum allowed window size |

### Window Output Format

Each window in the returned list is a dictionary containing:

```python
{
    "index": 0,                    # Window index (0-based)
    "start_sec": 0.0,              # Start time in seconds
    "end_sec": 2.0,                # End time in seconds (virtual if padded)
    "start_sample": 0,             # Start sample index
    "end_sample": 32000,           # End sample index (virtual if padded)
    "waveform": torch.Tensor,      # Shape [1, window_samples]
    "is_padded": False,            # Whether window was padded
    "actual_end_sample": 32000,    # Only present when padded
}
```

### Padding Modes

| Mode | Behavior |
|------|----------|
| `"zero"` | Pad partial windows with zeros to reach full `window_sec` length |
| `"reflect"` | Pad by reflecting the audio tail (deterministic) |
| `"none"` | No padding - last window may be shorter than `window_sec` |

**Note on timestamps:**

- When `pad_mode="zero"` or `"reflect"`, `end_sec` and `end_sample` represent the *virtual* end (as if audio continued)
- When `pad_mode="none"`, `end_sec` and `end_sample` represent the *actual* audio end

### Error Handling

```python
from timeline import WindowingConfig, segment_audio
from timeline.errors import WindowingConfigError, WindowingRuntimeError

try:
    config = WindowingConfig(hop_sec=3.0, window_sec=2.0)  # Invalid!
except WindowingConfigError as e:
    print(f"[{e.code}] {e.message}")
    # [INVALID_CONFIG] hop_sec (3.0) must be <= window_sec (2.0)

try:
    windows = segment_audio(waveform, 16000, config)
except WindowingRuntimeError as e:
    print(f"[{e.code}] {e.message}")
```

### Error Codes

| Code | Exception | Description |
|------|-----------|-------------|
| `INVALID_CONFIG` | WindowingConfigError | Invalid configuration parameters |
| `INVALID_SHAPE` | WindowingRuntimeError | Waveform not shape [1, T] |
| `EMPTY_INPUT` | WindowingRuntimeError | Waveform has zero samples |

### Integration with audioio

```python
from audioio import load_validate_preprocess
from timeline import WindowingConfig, segment_audio

# Phase 1: Load and preprocess audio
waveform, sr = load_validate_preprocess("speech.wav")

# Phase 2: Segment into windows for emotion timeline
config = WindowingConfig(window_sec=2.0, hop_sec=0.5)
windows = segment_audio(waveform, sr, config)

# Ready for Phase 3: emotion inference on each window
for w in windows:
    # emotion = model.predict(w["waveform"])
    print(f"{w['start_sec']:.1f}s - {w['end_sec']:.1f}s")
```

## 📄 License

MIT
