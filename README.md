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
├── timeline/              # Timeline windowing & generation (Phase 2 & 4)
│   ├── windowing.py       # Audio segmentation (Phase 2)
│   ├── generate.py        # Timeline orchestration (Phase 4)
│   ├── smooth.py          # Smoothing strategies (Phase 4)
│   ├── merge.py           # Segment merging (Phase 4)
│   ├── schema.py          # Output data structures (Phase 4)
│   └── errors.py          # Custom exceptions
├── model/                 # Emotion inference (Phase 3)
│   ├── infer.py          # Main inference functions
│   ├── labels.py         # Canonical labels & mapping
│   ├── registry.py       # Model loading & caching
│   ├── types.py          # Type definitions
│   └── errors.py         # Custom exceptions
├── src/api/              # REST API (Phase 5)
│   ├── main.py           # FastAPI application
│   ├── config.py         # Settings management
│   ├── schemas.py        # API request/response models
│   ├── errors.py         # Error handling & mapping
│   ├── logging.py        # Structured logging
│   └── deps.py           # Dependency injection
├── backend/              # Legacy FastAPI backend
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
├── docker/              # Docker configuration
│   └── Dockerfile.api   # Production API container
├── scripts/             # CLI tools
│   ├── predict_file.py  # Single file prediction
│   ├── predict_timeline.py  # Timeline generation
│   ├── eval_dataset.py  # Dataset evaluation
│   ├── run_api.sh       # API server runner
│   └── generate_fixtures.py  # Generate test audio
├── tests/               # Test suite
│   ├── test_label_mapping.py
│   ├── test_model_infer_smoke.py
│   ├── test_timeline_smoothing.py   # Phase 4 tests
│   ├── test_timeline_merge.py       # Phase 4 tests
│   ├── test_timeline_generate.py    # Phase 4 tests
│   ├── test_api_health.py           # Phase 5 tests
│   ├── test_api_predict.py          # Phase 5 tests
│   ├── test_api_timeline.py         # Phase 5 tests
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
# Build and start the API service
docker compose build api
docker compose up api

# Or start all services (API + frontend)
docker compose up --build
```

Services will be available at:

- **API:** <http://localhost:8000>
- **API Docs:** <http://localhost:8000/docs>
- **Frontend:** <http://localhost:8501>

### Quick Test

```bash
# Health check
curl http://localhost:8000/health

# Predict emotion
curl -X POST http://localhost:8000/predict -F "file=@sample.wav"

# Generate timeline
curl -X POST http://localhost:8000/timeline \
  -F "file=@sample.wav" \
  -F "window_sec=2.0" \
  -F "hop_sec=0.5"
```

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

## 🎭 Timeline Emotion Generation (Phase 4)

The Phase 4 timeline module provides complete emotion timeline generation, including windowing, inference, smoothing, and segment merging. It produces clean, stable emotion timelines from audio files.

### Quick Start

```python
from timeline import generate_timeline, SmoothingConfig, MergeConfig, WindowingConfig

# Generate timeline from audio file (simplest usage)
result = generate_timeline("speech.wav")

# Print emotion segments
for segment in result.segments:
    print(f"{segment.start_sec:.2f}s - {segment.end_sec:.2f}s: {segment.emotion} ({segment.confidence:.0%})")

# Output:
# 0.00s - 3.50s: neutral (82%)
# 3.50s - 7.00s: happy (91%)
# 7.00s - 10.50s: neutral (78%)
```

### Full Configuration

```python
from timeline import (
    generate_timeline,
    generate_timeline_from_waveform,
    WindowingConfig,
    SmoothingConfig,
    MergeConfig,
)
from audioio import AudioConfig

# Configure each stage
windowing_config = WindowingConfig(
    window_sec=2.0,      # 2-second analysis windows
    hop_sec=0.5,         # 0.5-second hop (75% overlap)
    pad_mode="zero",     # Zero-pad partial windows
)

smoothing_config = SmoothingConfig(
    method="hysteresis",     # Recommended smoothing method
    hysteresis_min_run=3,    # Require 3 consecutive windows to switch emotion
)

merge_config = MergeConfig(
    merge_adjacent=True,     # Merge same-emotion segments
    min_segment_sec=0.25,    # Minimum segment duration
    drop_short_segments=False,
)

# Generate timeline with full control
result = generate_timeline(
    path_or_bytes="speech.wav",
    audio_config=AudioConfig(),
    windowing_config=windowing_config,
    model_id="baseline",
    device="cpu",
    smoothing_config=smoothing_config,
    merge_config=merge_config,
    include_windows=True,    # Include per-window predictions
    include_scores=True,     # Include per-label probability scores
)

# Access results
print(f"Duration: {result.duration_sec:.2f}s")
print(f"Segments: {result.segment_count}")
print(f"Windows: {result.window_count}")
```

### Output Schema

The `TimelineResult` object contains:

```python
@dataclass
class TimelineResult:
    model_name: str           # Model used for inference
    sample_rate: int          # Audio sample rate (Hz)
    duration_sec: float       # Total audio duration
    window_sec: float         # Window duration used
    hop_sec: float            # Hop duration used
    pad_mode: str             # Padding mode used
    smoothing: dict           # Smoothing config applied
    segments: list[Segment]   # Emotion segments
    windows: list | None      # Per-window predictions (if requested)
    is_padded_timeline: bool  # Whether any window was padded
    merge_config: dict        # Merge config applied
```

Each `Segment` contains:

```python
@dataclass
class Segment:
    start_sec: float          # Segment start time
    end_sec: float            # Segment end time
    emotion: str              # Canonical emotion label
    confidence: float         # Average confidence (0.0-1.0)
    scores: dict | None       # Average per-label scores (if requested)
    window_count: int         # Number of windows merged
```

### JSON Output

Convert to JSON for serialization:

```python
# Convert to dict
data = result.to_dict(include_windows=True, include_scores=True)

# Serialize to JSON
import json
json_output = json.dumps(data, indent=2)
```

Example JSON output:

```json
{
  "model_name": "baseline",
  "sample_rate": 16000,
  "duration_sec": 10.5,
  "window_sec": 2.0,
  "hop_sec": 0.5,
  "pad_mode": "zero",
  "smoothing": {
    "method": "hysteresis",
    "hysteresis_min_run": 3
  },
  "is_padded_timeline": true,
  "segments": [
    {
      "start_sec": 0.0,
      "end_sec": 3.5,
      "emotion": "neutral",
      "confidence": 0.82
    },
    {
      "start_sec": 3.5,
      "end_sec": 7.0,
      "emotion": "happy",
      "confidence": 0.91
    },
    {
      "start_sec": 7.0,
      "end_sec": 10.5,
      "emotion": "neutral",
      "confidence": 0.78
    }
  ]
}
```

### Smoothing Methods

The smoothing stage reduces jitter in window-by-window predictions:

| Method | Description | Best For |
|--------|-------------|----------|
| `none` | No smoothing | Debugging, analysis |
| `majority` | Majority vote in sliding window | General stabilization |
| `hysteresis` | Require N consecutive windows to switch (recommended) | Clean transitions |
| `ema` | Exponential moving average on scores | Smooth probability curves |

#### Hysteresis Smoothing (Recommended)

Hysteresis requires a new emotion to persist for `hysteresis_min_run` consecutive windows before switching:

```python
# Example: [A, A, A, B, B] with min_run=3
# Output: [A, A, A, A, A]  (B only persists 2 windows, not enough to switch)

# Example: [A, A, A, B, B, B] with min_run=3
# Output: [A, A, A, B, B, B]  (B persists 3 windows, switch happens at first B)
```

#### Majority Smoothing

Replaces each window's emotion with the majority in a centered sliding window:

```python
SmoothingConfig(
    method="majority",
    majority_window=5,  # Must be odd; looks at 5 windows centered on each position
)
```

#### EMA Smoothing

Applies exponential moving average to per-label probability scores:

```python
SmoothingConfig(
    method="ema",
    ema_alpha=0.6,  # Higher = more weight to recent predictions
)
```

### CLI: Timeline Generation

```bash
# Basic usage
python scripts/predict_timeline.py --input speech.wav

# With custom windowing
python scripts/predict_timeline.py --input speech.wav --window_sec 3.0 --hop_sec 1.0

# With smoothing options
python scripts/predict_timeline.py --input speech.wav --smoothing hysteresis --hysteresis_min_run 4
python scripts/predict_timeline.py --input speech.wav --smoothing majority --majority_window 7
python scripts/predict_timeline.py --input speech.wav --smoothing ema --ema_alpha 0.7

# Include window predictions and scores
python scripts/predict_timeline.py --input speech.wav --include_windows --include_scores --pretty

# Save to file
python scripts/predict_timeline.py --input speech.wav --output timeline.json --pretty

# Using Docker
docker compose run --rm dev python scripts/predict_timeline.py --input tests/fixtures/example.wav --pretty
```

### Recommended Defaults

For most speech emotion recognition use cases:

```python
# Recommended configuration
windowing_config = WindowingConfig(
    window_sec=2.0,     # 2-second windows capture prosodic patterns
    hop_sec=0.5,        # 75% overlap for smooth timeline
    pad_mode="zero",
)

smoothing_config = SmoothingConfig(
    method="hysteresis",
    hysteresis_min_run=3,   # Requires ~1.5s persistence to switch
)

merge_config = MergeConfig(
    merge_adjacent=True,
    min_segment_sec=0.5,    # Minimum 0.5s segments
    drop_short_segments=False,
)
```

### Error Handling

```python
from timeline import generate_timeline
from timeline.errors import TimelineError, WindowingConfigError, WindowingRuntimeError
from audioio.errors import AudioIOError
from model.errors import ModelError

try:
    result = generate_timeline("speech.wav")
except AudioIOError as e:
    print(f"Audio error: [{e.code}] {e.message}")
except ModelError as e:
    print(f"Model error: [{e.code}] {e.message}")
except TimelineError as e:
    print(f"Timeline error: [{e.code}] {e.message}")
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

# Run timeline generation
docker compose run --rm dev python scripts/predict_timeline.py --input tests/fixtures/example.wav --pretty

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

## 🌐 REST API (Phase 5)

The REST API provides production-ready endpoints for emotion prediction and timeline generation.

### Building and Running the API

```bash
# Build the API container
docker compose build api

# Start the API service
docker compose up api

# Or start all services (API, frontend, etc.)
docker compose up
```

The API will be available at:

- **API:** <http://localhost:8000>
- **OpenAPI Docs:** <http://localhost:8000/docs>
- **ReDoc:** <http://localhost:8000/redoc>

### API Endpoints

#### `GET /health`

Check service health and model status.

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "ok",
  "model_id": "baseline",
  "device": "cpu"
}
```

#### `POST /predict`

Predict emotion from a single audio clip.

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@sample.wav"
```

With scores:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@sample.wav" \
  -F "include_scores=true"
```

**Response:**

```json
{
  "emotion": "happy",
  "confidence": 0.85,
  "scores": {"angry": 0.05, "happy": 0.85, "neutral": 0.05, "sad": 0.05},
  "model_name": "speechbrain-iemocap",
  "duration_sec": 2.5
}
```

#### `POST /timeline`

Generate an emotion timeline for longer audio.

```bash
curl -X POST http://localhost:8000/timeline \
  -F "file=@sample.wav" \
  -F "window_sec=2.0" \
  -F "hop_sec=0.5"
```

With all options:

```bash
curl -X POST http://localhost:8000/timeline \
  -F "file=@sample.wav" \
  -F "window_sec=2.0" \
  -F "hop_sec=0.5" \
  -F "pad_mode=zero" \
  -F "smoothing_method=hysteresis" \
  -F "hysteresis_min_run=3" \
  -F "include_windows=true" \
  -F "include_scores=true"
```

**Response:**

```json
{
  "model_name": "speechbrain-iemocap",
  "sample_rate": 16000,
  "duration_sec": 10.5,
  "window_sec": 2.0,
  "hop_sec": 0.5,
  "pad_mode": "zero",
  "smoothing": {"method": "hysteresis", "hysteresis_min_run": 3},
  "segments": [
    {"start_sec": 0.0, "end_sec": 3.5, "emotion": "neutral", "confidence": 0.82},
    {"start_sec": 3.5, "end_sec": 7.0, "emotion": "happy", "confidence": 0.91},
    {"start_sec": 7.0, "end_sec": 10.5, "emotion": "neutral", "confidence": 0.78}
  ]
}
```

### API Parameters

| Parameter | Endpoint | Type | Default | Description |
|-----------|----------|------|---------|-------------|
| `file` | Both | UploadFile | Required | WAV audio file |
| `include_scores` | Both | bool | false | Include per-label scores |
| `window_sec` | /timeline | float | 2.0 | Window duration (seconds) |
| `hop_sec` | /timeline | float | 0.5 | Hop stride (seconds) |
| `pad_mode` | /timeline | string | "zero" | Padding: none, zero, reflect |
| `smoothing_method` | /timeline | string | "hysteresis" | Smoothing: none, majority, hysteresis, ema |
| `hysteresis_min_run` | /timeline | int | 3 | Min windows to switch emotion |
| `majority_window` | /timeline | int | 5 | Majority voting window size |
| `ema_alpha` | /timeline | float | 0.6 | EMA smoothing coefficient |
| `include_windows` | /timeline | bool | false | Include per-window predictions |

### Error Handling

All errors return a consistent JSON structure:

```json
{
  "error": {
    "code": "INVALID_AUDIO",
    "message": "Failed to decode audio file",
    "details": {"filename": "test.txt"}
  }
}
```

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| `INVALID_AUDIO` | 400 | Audio file cannot be decoded |
| `INVALID_INPUT` | 422 | Request validation failed |
| `INVALID_WINDOWING` | 422 | Invalid windowing configuration |
| `MODEL_LOAD_FAILED` | 500 | Model loading error |
| `INFERENCE_FAILED` | 500 | Model inference error |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

### Environment Variables

Configure the API via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | SER Service | Application name |
| `LOG_LEVEL` | INFO | Logging level |
| `MODEL_ID` | baseline | Model identifier |
| `DEVICE` | cpu | Inference device (cpu/cuda) |
| `MAX_DURATION_SEC` | 600 | Max audio duration |
| `DEFAULT_WINDOW_SEC` | 2.0 | Default window size |
| `DEFAULT_HOP_SEC` | 0.5 | Default hop size |
| `DEFAULT_PAD_MODE` | zero | Default padding |
| `DEFAULT_SMOOTHING_METHOD` | hysteresis | Default smoothing |
| `DEFAULT_HYSTERESIS_MIN_RUN` | 3 | Default hysteresis param |

### Running Tests

```bash
# Run API tests (no model required)
docker compose run --rm dev pytest tests/test_api_health.py tests/test_api_predict.py tests/test_api_timeline.py -v

# Run with integration tests (downloads model)
docker compose run --rm -e RUN_INTEGRATION_TESTS=1 dev pytest tests/test_api_*.py -v
```

## 📄 License

MIT
