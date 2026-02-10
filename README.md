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
docker compose up

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

## 📄 License

MIT
