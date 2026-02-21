# Speech Emotion Recognition Pipeline Report

**Scope:** Audio emotion analysis pipeline from user input (upload/record) to single-clip prediction or emotion timeline output.  
**Interfaces:** Streamlit UI (debug/monitoring cockpit) and FastAPI service.  
**Core:** Deterministic audio pipeline + windowing + inference + smoothing + segment merging.

---

## 1. System Overview

The system processes speech audio and predicts emotion in two modes:

- **Single-clip mode:** One emotion label and confidence for the entire audio clip.
- **Timeline mode:** Emotion over time computed through sliding windows, smoothing, and segment merging.

Design goals:

- **Reproducibility:** Same input and settings produce the same output.
- **Robustness:** Validation rejects invalid or meaningless input early.
- **Observability:** Logs and step timings are captured and shown in Streamlit (RunReport concept).
- **Modularity:** Core logic is shared between UI and API.

---

## 2. Clean Modular Architecture (Layered)

```mermaid
flowchart TB
  subgraph L1[Presentation Layer]
    ST[Streamlit UI<br/>apps/streamlit_app<br/>Upload Record Visualize RunReport]
    API[FastAPI Service<br/>src/api<br/>REST validation error mapping]
  end

  subgraph L2[Application Layer]
    ORCH[Orchestrator<br/>run_analysis and timeline.generate<br/>collect timings logs config]
  end

  subgraph L3[Core Library]
    AIO[audioio<br/>src/audioio<br/>Decode Validate Preprocess Stats]
    WIN[Windowing<br/>src/timeline/windowing<br/>Window hop padding timestamps]
    SER[Model<br/>src/model<br/>Registry cache inference mapping]
    TLINE[Timeline<br/>src/timeline<br/>Smooth Merge Segments]
  end

  subgraph L4[Infrastructure]
    LOG[Logging<br/>request id structured logs UI buffer]
    CFG[Config<br/>AudioConfig WindowingConfig SmoothingConfig MergeConfig]
    DOCKER[Docker<br/>reproducible dev and api runtime]
  end

  ST --> ORCH
  API --> ORCH
  ORCH --> AIO
  ORCH --> WIN
  ORCH --> SER
  ORCH --> TLINE
  TLINE --> WIN
  TLINE --> SER
  ORCH --> LOG
  AIO --> CFG
  WIN --> CFG
  SER --> CFG
  TLINE --> CFG
  DOCKER --> API
  DOCKER --> ST
```

---

## 3. Activity Diagram (Detailed)

End-to-end activity from input to outputs, including error paths and monitoring.

```mermaid
flowchart TD
  A([Start user opens UI or calls API]) --> B{Input type}
  B -->|Upload| C[Read file bytes]
  B -->|Record| D[Capture audio to bytes]
  C --> E[Init run id and log buffer]
  D --> E

  E --> F[Decode wav]
  F -->|Decode error| FERR[Return error INVALID_WAV]
  F --> G[Validate audio]
  G -->|Validation error| GERR[Return error INVALID_INPUT]
  G --> H[Preprocess audio]
  H --> I[Compute audio stats]
  I --> J{Mode}

  J -->|Single clip| K[Infer full clip]
  K --> K1[Map labels to canonical]
  K1 --> K2[Build prediction result]
  K2 --> OUT1[Return prediction and logs timings]

  J -->|Timeline| L[Windowing segment audio]
  L --> L1[Infer per window]
  L1 --> L2[Window predictions]
  L2 --> M[Smoothing]
  M --> N[Merge windows to segments]
  N --> OUT2[Return timeline and logs timings]

  OUT1 --> Z([End])
  OUT2 --> Z
```

---

## 4. Activity Diagram (Simplified)

Minimal view for executive summary.

```mermaid
flowchart TD
  A([Start]) --> B[Input audio]
  B --> C[Decode validate preprocess]
  C --> D{Mode}
  D -->|Single| E[Infer once]
  D -->|Timeline| F[Window infer smooth merge]
  E --> G[Output prediction]
  F --> H[Output timeline segments]
  G --> Z([End])
  H --> Z
```

---

## 5. Runtime Sequence Architecture

Runtime interactions between modules.

```mermaid
sequenceDiagram
  autonumber
  actor U as User
  participant UI as Streamlit or Client
  participant OR as Orchestrator
  participant A as audioio
  participant W as windowing
  participant M as model
  participant S as smoothing
  participant G as merge

  U->>UI: Upload or Record
  UI->>OR: run_analysis(bytes, config)
  OR->>A: decode validate preprocess
  A-->>OR: waveform and stats

  alt Single clip
    OR->>M: predict_waveform(full clip)
    M-->>OR: scores emotion confidence
  else Timeline
    OR->>W: segment_audio(waveform)
    W-->>OR: list of windows
    loop each window
      OR->>M: predict_waveform(window)
      M-->>OR: scores emotion confidence
    end
    OR->>S: smooth windows
    S-->>OR: smoothed windows
    OR->>G: merge to segments
    G-->>OR: segments
  end

  OR-->>UI: results plus logs timings run id
  UI-->>U: plots tables export
```

---

## 6. Configuration Index

### 6.1 Audio validation and preprocessing

| Setting | Meaning | Why it matters |
|---|---|---|
| `target_sample_rate` | Resample all audio to a common sampling rate (e.g., 16000 Hz). | Ensures consistent time scale and model compatibility. |
| `to_mono` | Convert audio to single channel. | Standardizes input and avoids channel imbalance. |
| `normalize` + `peak_target` | Peak normalize amplitude so max abs value approaches target. | Reduces loudness variation and clipping risk. |
| `min_duration_sec` / `max_duration_sec` | Allowed duration range. | Prevents invalid input and runaway compute. |
| `reject_silence` + `silence_rms_threshold` | Reject near-silent audio using RMS energy. | Prevents meaningless predictions and improves reliability. |

### 6.2 Windowing and stride parameters (Timeline mode)

| Setting | Meaning | Practical effect |
|---|---|---|
| `window_sec` | Length of each analysis chunk. | Larger means more context, smoother but less reactive. |
| `hop_sec` (stride) | Step between window starts. | Smaller means higher time resolution but more compute. |
| overlap | Derived as `window_sec - hop_sec`. | More overlap stabilizes but increases compute. |
| `pad_mode` | How to handle end-of-audio windows. | `zero` adds silence, `reflect` avoids silence, `none` gives shorter last window. |
| `include_partial_last_window` | Keep or drop trailing partial window. | Controls coverage of the last seconds. |

### 6.3 Smoothing parameters

| Setting | Meaning | Practical effect |
|---|---|---|
| `method=none` | No stabilization. | Most jitter, fastest. |
| `method=majority`, `majority_window` | Majority vote across neighbor windows. | Strong jitter reduction, can lag transitions. |
| `method=hysteresis`, `hysteresis_min_run` | Switch emotion only after N consistent windows. | Stable and still responsive. |
| `method=ema`, `ema_alpha` | Smooth score probabilities over time. | Smooth confidence curves and transitions. |

### 6.4 Segment merging parameters

| Setting | Meaning | Practical effect |
|---|---|---|
| `merge_adjacent` | Merge consecutive same-emotion windows into one segment. | Produces clean human-readable timeline. |
| `min_segment_sec` | Minimum acceptable segment duration. | Defines what counts as noise. |
| `drop_short_segments` + strategy | Merge tiny segments into neighbors. | Reduces flicker segments in the final timeline. |

---

## 7. Why this pipeline is strong

1. **Deterministic processing:** decoding, resampling, windowing, smoothing are stable and testable.  
2. **Clean modular design:** UI and API reuse the same core library.  
3. **Observability via RunReport:** logs + timings + config snapshot per run.  
4. **Temporal stability controls:** smoothing and merging reduce jitter and improve usability.  
5. **Robust validation:** prevents garbage-in garbage-out.

---

## 8. Recommended defaults

| Category | Defaults |
|---|---|
| Audio | `target_sample_rate=16000`, `to_mono=True`, `normalize=True` |
| Windowing | `window_sec=2.0`, `hop_sec=0.5`, `pad_mode=zero` |
| Smoothing | `method=hysteresis`, `hysteresis_min_run=3` |
| Merge | `merge_adjacent=True`, `min_segment_sec=0.25`, `drop_short_segments=False` |
