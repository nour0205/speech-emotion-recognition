# Unreal Engine 5 Integration Guide

This document explains the full architecture of the `unreal/` subtree in this monorepo.
It is written for a student developer who is familiar with Python but new to Unreal Engine.

---

## Table of Contents

1. [Architecture overview](#architecture-overview)
2. [Why API + Unreal instead of ML inside UE](#why-api--unreal)
3. [Monorepo layout](#monorepo-layout)
4. [Prerequisites on macOS](#prerequisites-on-macos)
5. [Starting the backend](#starting-the-backend)
6. [Building the Unreal project](#building-the-unreal-project)
7. [Plugin discovery](#plugin-discovery)
8. [Opening the editor tab](#opening-the-editor-tab)
9. [End-to-end demo](#end-to-end-demo)
10. [Testing with a sample WAV](#testing-with-a-sample-wav)
11. [Code walkthrough](#code-walkthrough)
12. [Troubleshooting](#troubleshooting)
13. [Future work](#future-work)

---

## Architecture overview

```
┌─────────────────────────────────────┐      HTTP (localhost:8000)
│   Unreal Editor (macOS)             │ ──────────────────────────────►
│                                     │                                │
│  ┌───────────────────────────────┐  │  POST /timeline               │
│  │  Emotion Bridge tab (Slate)   │  │  multipart/form-data          │
│  │                               │  │  file=<wav bytes>             │
│  │  • WAV file picker            │  │  + optional params            │
│  │  • Timeline param controls    │  │                                │
│  │  • Analyze / Play / Stop      │  │ ◄──────────────────────────────
│  └───────────────────────────────┘  │  JSON: { segments: [...] }
│            │                        │
│            ▼                        │
│  ┌───────────────────────────────┐  │
│  │   AEmotionLampActor           │  │
│  │   (cube + point light)        │  │
│  │   ApplyEmotion("happy", 0.91) │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  Python backend (Docker or local)   │
│                                     │
│  FastAPI + SpeechBrain IEMOCAP      │
│  GET  /health                       │
│  POST /predict    → single emotion  │
│  POST /timeline   → emotion list    │
└─────────────────────────────────────┘
```

The Unreal side is entirely client-side: it never modifies the Python backend and does not
run any ML code. All intelligence lives in the backend.

---

## Why API + Unreal

| Approach | Pros | Cons |
|----------|------|------|
| ML inside Unreal (C++ ONNX) | No network required | 1 GB+ model weights, no PyTorch API, limited model support |
| Python plugin via subprocess | Tight integration | Packaging nightmare, OS-specific Python path |
| **REST API (our approach)** | Backend evolves independently; any language can call it; easy to swap models | Requires local network; first-run model download |

The REST API pattern is standard in production entertainment systems where ML runs on a
GPU server and the game engine consumes results. This integration mimics that architecture
at localhost scale.

---

## Monorepo layout

```
speech-emotion-recognition/         ← repo root
├── src/                            ← Python ML library
│   ├── audioio/
│   ├── model/
│   ├── timeline/
│   └── api/                        ← FastAPI app (unchanged)
├── apps/streamlit_app/             ← existing web frontend (unchanged)
├── tests/
├── docker-compose.yml
├── docs/
│   └── UNREAL_INTEGRATION.md       ← this file
└── unreal/                         ← NEW: Unreal Engine integration
    ├── EmotionDemo.uproject
    ├── README.md
    ├── Config/
    ├── Source/EmotionDemo/         ← thin game module (IMPLEMENT_PRIMARY_GAME_MODULE)
    └── Plugins/
        └── EmotionBridge/          ← the actual plugin
            ├── EmotionBridge.uplugin
            └── Source/
                ├── EmotionBridge/          ← runtime module
                └── EmotionBridgeEditor/    ← editor module
```

The Python backend is not touched. The `unreal/` tree is self-contained and could be
extracted into its own repo later without any changes.

---

## Prerequisites on macOS

### 1. Unreal Engine 5.4+

Install via the **Epic Games Launcher** (free account required).
The engine is installed to `~/Library/Epic/UE_5.4` by default.
The launcher also manages updates and multiple engine versions.

### 2. Xcode 15+

```bash
# Check
xcode-select -p
# Install if missing
xcode-select --install
```

Open Xcode once after installation and accept the license agreement.

### 3. Xcode Command Line Tools

Required for the UBT (Unreal Build Tool) build system.
`xcode-select --install` covers this.

### 4. Python backend dependencies

```bash
pip install -r requirements/backend.txt
# or use Docker
docker compose build api
```

---

## Starting the backend

### Docker (recommended — no Python install needed)

```bash
cd speech-emotion-recognition
docker compose up api
```

Check health:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok", "model": "speechbrain-iemocap", "device": "cpu"}
```

### Local Python

```bash
source .venv/bin/activate
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### First-run delay

The SpeechBrain IEMOCAP model (~350 MB) downloads from HuggingFace Hub on the
first `/timeline` request. The download is cached in `~/.cache/huggingface/`.
Subsequent requests are fast. The `FEmotionApiClient` sets a 180-second HTTP
timeout to accommodate this.

---

## Building the Unreal project

### Step 1: Generate Xcode project files

Right-click `unreal/EmotionDemo.uproject` in Finder and choose
**Services → Generate Xcode Project Files**.

Or from Terminal:
```bash
UE5_ROOT=~/Library/Epic/UE_5.4   # adjust to your install path
"$UE5_ROOT/Engine/Build/BatchFiles/Mac/GenerateProjectFiles.sh" \
    "$(pwd)/unreal/EmotionDemo.uproject" -game
```

This generates `unreal/EmotionDemo.xcworkspace`.

### Step 2: Build

```bash
open unreal/EmotionDemo.xcworkspace
```

In Xcode:
- Select **EmotionDemoEditor** from the scheme picker.
- Press **⌘B** (Build).

First build: 10–20 min (UE headers and intermediates are compiled).
Incremental build: 30–90 s.

### Step 3: Open in editor

Double-click `unreal/EmotionDemo.uproject` in Finder, or:
```bash
"$UE5_ROOT/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor" \
    "$(pwd)/unreal/EmotionDemo.uproject"
```

---

## Plugin discovery

Unreal discovers the `EmotionBridge` plugin because:

1. `EmotionDemo.uproject` lists it in the `"Plugins"` array with `"Enabled": true`.
2. The plugin resides at `unreal/Plugins/EmotionBridge/EmotionBridge.uplugin`.

No manual symlinks or engine plugin folder copying is required.
The plugin is project-local and travels with the repo.

---

## Opening the editor tab

**Window → Emotion Bridge**

The tab is registered as a "Nomad Tab" by `FEmotionBridgeEditorModule::StartupModule()`.
It can be:
- Docked alongside the Content Browser.
- Floated as a separate window.
- Added to a tab stack with other panels.

If the menu entry is missing, the editor module failed to load — check the Output Log
(`Window → Output Log`) for compile errors.

---

## End-to-end demo

1. Start backend → `docker compose up api`.
2. Open project in Unreal Editor.
3. Create or open a level (follow `MANUAL_ASSET_SETUP.md`).
4. Open **Window → Emotion Bridge**.
5. Click **Health Check** → status turns green.
6. Click **Browse** → select a `.wav` file.
7. Click **Analyze** → wait for the response.
8. Segment list appears with colored emotion labels.
9. Click **Play Demo** → lamp actor changes color in real time.
10. Click **Stop Demo** → lamp resets to white.

---

## Testing with a sample WAV

Any standard WAV file works. The backend auto-converts sample rate and channels.
Recommended test files:

```bash
# Record 10 seconds of speech (macOS)
sox -d -r 16000 -c 1 /tmp/test.wav trim 0 10

# Or download a LibriSpeech sample
curl -L "https://www.openslr.org/resources/12/dev-clean.tar.gz" | \
    tar xz --strip-components=4 -C /tmp '*/1272-128104-0000.flac'
ffmpeg -i /tmp/1272-128104-0000.flac -ar 16000 -ac 1 /tmp/test.wav
```

Alternatively, the `tests/` directory may contain fixture WAV files.

---

## Code walkthrough

### `FEmotionApiClient` (EmotionApiClient.cpp)

The most technically involved class. UE's `FHttpModule` does not have a multipart helper,
so the boundary-delimited body is constructed manually:

```
--boundary\r\n
Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n
Content-Type: audio/wav\r\n\r\n
<raw bytes of WAV file>
\r\n
--boundary\r\n
Content-Disposition: form-data; name="window_sec"\r\n\r\n
2.0000\r\n
...
--boundary--\r\n
```

All string segments are converted to UTF-8 with `FTCHARToUTF8` before appending
to the `TArray<uint8>` body buffer.

### `SEmotionBridgePanel` (SEmotionBridgePanel.cpp)

Pure Slate widget. No Blueprints, no UMG. Key points:
- `SComboBox<TSharedPtr<FString>>` for pad_mode and smoothing_method dropdowns.
- `SListView<TSharedPtr<FEmotionSegmentRow>>` for the segment table.
- Playback uses `FTSTicker::GetCoreTicker().AddTicker(...)` — a core engine ticker
  that fires each frame on the game thread, regardless of which tab is focused.
- Wall-clock time via `FPlatformTime::Seconds()` for accurate timeline simulation.

### `AEmotionLampActor` (EmotionLampActor.cpp)

Standard `AActor` with two components:
- `UStaticMeshComponent` — defaults to `/Engine/BasicShapes/Cube`.
- `UPointLightComponent` — positioned 100 cm above the mesh.
- `UMaterialInstanceDynamic` — created in `BeginPlay()` to tint the mesh if the
  material exposes a `BaseColor` vector parameter.

### `UEmotionBridgeSettings` (EmotionBridgeSettings.cpp)

Inherits `UDeveloperSettings`. The `TMap<FString, FLinearColor> EmotionColors` field
is persisted to `Config/EmotionBridge.ini` and editable in Project Settings without
recompiling.

---

## Troubleshooting

See [unreal/README.md](../unreal/README.md#troubleshooting) for a complete table.

Quick reference:
- **Build error `WorkspaceMenuStructureModule`**: Add `"WorkspaceMenuStructure"` to `PublicDependencyModuleNames` in `EmotionBridgeEditor.Build.cs` (already included).
- **Tab missing from Window menu**: Editor module load failed — check Output Log.
- **HTTP 422 from backend**: The multipart body was malformed — check UE log for the request URL and ensure the file path is absolute.
- **First request times out**: Normal on first run. Set `Request->SetTimeout(180.f)` is already in the code.
- **Actor not changing color**: Ensure a level is open with at least one `AEmotionLampActor`, or let the plugin auto-spawn one.

---

## Future work

| Feature | Complexity | Notes |
|---------|-----------|-------|
| Microphone capture | Medium | `UAudioCaptureComponent` + temp WAV write |
| Runtime in-game HUD | Low | Replace editor panel with `UUserWidget` in UMG |
| Lip sync | High | Needs audio playback + phoneme alignment |
| MetaHuman blendshapes | High | Map emotion → ARKit blend shape weights via LiveLink |
| Multiple characters | Low | Iterate `TActorIterator<AEmotionLampActor>` and assign by tag |
| Cloud backend | Trivial | Change `ApiBaseUrl` in Project Settings |
| Real-time microphone + streaming | High | Server-side streaming API + chunked HTTP |
