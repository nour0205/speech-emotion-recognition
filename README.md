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
├── audioio/                # Audio I/O module (Phase 1)
│   ├── loader.py          # WAV loading
│   ├── validate.py        # Audio validation
│   ├── preprocess.py      # Preprocessing pipeline
│   └── errors.py          # Custom exceptions
├── timeline/              # Timeline windowing (Phase 2)
│   ├── windowing.py       # Audio segmentation
│   └── errors.py          # Custom exceptions
├── model/                 # Emotion inference (Phase 3)
│   ├── infer.py          # Main inference functions
│   ├── labels.py         # Canonical labels & mapping
│   ├── registry.py       # Model loading & caching
│   ├── types.py          # Type definitions
│   └── errors.py         # Custom exceptions
├── backend/              # FastAPI backend
│   ├── Dockerfile
│   ├── main.py           # API entry point
│   ├── core/
│   │   └── model.py      # Emotion classifier
│   └── schemas/
│       └── emotion.py    # Pydantic models
├── frontend/             # Streamlit UI
│   ├── Dockerfile
│   ├── app.py           # Web interface
│   └── api_client.py    # Backend HTTP client
├── scripts/             # CLI tools
│   ├── predict_file.py  # Single file prediction
│   ├── eval_dataset.py  # Dataset evaluation
│   └── generate_fixtures.py  # Generate test audio
├── tests/               # Test suite
│   ├── test_label_mapping.py
│   ├── test_model_infer_smoke.py
│   └── fixtures/
├── requirements/
│   ├── base.txt         # Core ML dependencies
│   ├── backend.txt      # FastAPI dependencies
│   ├── frontend.txt     # Streamlit dependencies
│   └── dev.txt          # Development tools
├── docker-compose.yml   # Container orchestration
├── Dockerfile.dev       # Development container
└── Makefile             # Convenience commands
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

## 🧠 Model Inference Module (Phase 3)

The `model` module provides speech emotion recognition inference using a pretrained SpeechBrain wav2vec2 model trained on IEMOCAP.

### Quick Start

```python
from model import predict_clip, predict_waveform, CANONICAL_LABELS

# Predict from audio file
result = predict_clip("speech.wav")
print(f"Emotion: {result.emotion}")
print(f"Confidence: {result.confidence:.2%}")
print(f"All scores: {result.scores}")

# Predict from bytes (e.g., uploaded file)
with open("speech.wav", "rb") as f:
    result = predict_clip(f.read())

# Predict from preprocessed waveform
import torch
waveform = torch.randn(1, 32000).float()  # [1, T] mono float32
result = predict_waveform(waveform, sample_rate=16000)
```

### Canonical Labels

The project uses these canonical emotion labels:

| Canonical | Description | Supported by Baseline |
|-----------|-------------|----------------------|
| `neutral` | No strong emotion | ✅ |
| `happy` | Joy, positive | ✅ |
| `sad` | Sorrow, negative | ✅ |
| `angry` | Anger, frustration | ✅ |
| `fear` | Fear, anxiety | ❌ |
| `disgust` | Revulsion | ❌ |
| `surprise` | Unexpected | ❌ |

**Note:** The baseline model (SpeechBrain IEMOCAP) only supports 4 emotions. Labels marked ❌ will always have 0.0 probability in the output.

### Label Mapping

The baseline model outputs IEMOCAP labels which are mapped to canonical labels:

| Model Output | Canonical Label |
|--------------|-----------------|
| `neu` | `neutral` |
| `hap` | `happy` |
| `sad` | `sad` |
| `ang` | `angry` |

```python
from model.labels import map_raw_to_canonical

raw_scores = {"neu": 0.1, "hap": 0.6, "sad": 0.2, "ang": 0.1}
canonical = map_raw_to_canonical(raw_scores)
# {'neutral': 0.1, 'happy': 0.6, 'sad': 0.2, 'angry': 0.1, 'fear': 0.0, ...}
```

### PredictionResult

The inference functions return a `PredictionResult` object:

```python
@dataclass
class PredictionResult:
    emotion: str           # Canonical label (e.g., "happy")
    confidence: float      # Probability of predicted emotion (0.0-1.0)
    scores: dict[str, float]  # All canonical labels -> probabilities
    model_name: str        # Model identifier
    raw_label: str | None  # Original model label
    raw_scores: dict | None  # Original model scores
    duration_sec: float    # Audio duration in seconds

# Convert to dict for JSON serialization
result.to_dict()
```

### CLI: Single File Prediction

```bash
# Basic usage
python scripts/predict_file.py --input speech.wav

# Pretty-print output
python scripts/predict_file.py --input speech.wav --pretty

# Using Docker
docker compose run --rm dev python scripts/predict_file.py --input tests/fixtures/example.wav --pretty
```

Example output:
```json
{
  "emotion": "happy",
  "confidence": 0.85,
  "scores": {
    "neutral": 0.05,
    "happy": 0.85,
    "sad": 0.05,
    "angry": 0.05,
    "fear": 0.0,
    "disgust": 0.0,
    "surprise": 0.0
  },
  "model_name": "speechbrain-iemocap",
  "raw_label": "hap",
  "duration_sec": 2.5
}
```

### CLI: Dataset Evaluation

Evaluate on a folder-organized dataset:

```
dataset/
  angry/*.wav
  happy/*.wav
  sad/*.wav
  neutral/*.wav
```

```bash
# Basic evaluation
python scripts/eval_dataset.py --data_root ./dataset

# Limit samples per class
python scripts/eval_dataset.py --data_root ./dataset --limit_per_class 100

# With confusion matrix
python scripts/eval_dataset.py --data_root ./dataset --confusion

# Save results to JSON
python scripts/eval_dataset.py --data_root ./dataset --json results.json
```

Example output:
```
Evaluation Results
==================
Total samples: 400
Overall accuracy: 78.5%
Macro F1: 0.760

Per-class results:
--------------------------------------------------
angry        Acc: 80.0%  F1: 0.780  (80/100)
happy        Acc: 75.0%  F1: 0.740  (75/100)
neutral      Acc: 77.0%  F1: 0.765  (77/100)
sad          Acc: 82.0%  F1: 0.815  (82/100)
```

### Model Registry

Models are cached for efficient reuse:

```python
from model.registry import get_model, list_available_models

# List available models
print(list_available_models())  # ['baseline', 'speechbrain-iemocap']

# Get a model
model = get_model("baseline", device="cpu")
raw_scores = model.predict(waveform, sample_rate=16000)
```

### Error Handling

```python
from model import predict_clip
from model.errors import ModelError, ModelLoadError, InferenceError

try:
    result = predict_clip("speech.wav")
except ModelLoadError as e:
    print(f"Model error: [{e.code}] {e.message}")
except InferenceError as e:
    print(f"Inference error: [{e.code}] {e.message}")
```

### Error Codes

| Code | Exception | Description |
|------|-----------|-------------|
| `MODEL_NOT_FOUND` | ModelLoadError | Unknown model ID |
| `DOWNLOAD_FAILED` | ModelLoadError | Failed to download model |
| `LOAD_FAILED` | ModelLoadError | Model loading failed |
| `INVALID_INPUT` | InferenceError | Invalid waveform input |
| `INFERENCE_FAILED` | InferenceError | Model forward pass failed |

### First Run / Model Download

On first run, the SpeechBrain model (~360MB) is downloaded from HuggingFace Hub. This happens automatically and is cached for subsequent runs.

To pre-download the model:
```bash
# In Docker
docker compose run --rm dev python -c "from model import get_model; get_model('baseline')"

# Locally
python -c "from model import get_model; get_model('baseline')"
```

### Integration Example

Complete pipeline from audio file to emotion prediction:

```python
from audioio import load_validate_preprocess, AudioConfig
from timeline import WindowingConfig, segment_audio
from model import predict_waveform

# Phase 1: Load audio
waveform, sr = load_validate_preprocess("long_speech.wav")

# Phase 2: Segment into windows
windows = segment_audio(waveform, sr, WindowingConfig(window_sec=2.0, hop_sec=1.0))

# Phase 3: Predict emotion for each window
for w in windows:
    result = predict_waveform(w["waveform"], sample_rate=sr)
    print(f"{w['start_sec']:.1f}s: {result.emotion} ({result.confidence:.0%})")
```

## 🐳 Docker Development

### Running in Docker

```bash
# Build and run the dev container
docker compose run --rm dev bash

# Run tests in Docker
docker compose run --rm dev pytest -v

# Run tests with integration tests (downloads model)
docker compose run --rm -e RUN_INTEGRATION_TESTS=1 dev pytest -v

# Run prediction
docker compose run --rm dev python scripts/predict_file.py --input tests/fixtures/example.wav --pretty

# Generate test fixtures
docker compose run --rm dev python scripts/generate_fixtures.py
```

### Makefile Commands

```bash
make docker-dev     # Interactive dev container
make docker-test    # Run tests in Docker
make fixtures       # Generate test audio files
make predict FILE=audio.wav  # Predict emotion
```

## 📄 License

MIT
