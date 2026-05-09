# CLAUDE.md — Project Intelligence File

## Project Name
EmotionBridge (plugin) inside the `EmotionDemo` Unreal project — part of the `speech-emotion-recognition` monorepo.

## What This Project Does
An Unreal Engine 5.7 plugin that:
- Accepts WAV audio input from the editor (file picker / Take Library / runtime).
- Sends the audio to a local FastAPI service (SpeechBrain IEMOCAP model) over HTTP.
- Receives an emotion timeline (`segments: [{start_sec, end_sec, emotion, confidence}]`).
- Maps each emotion segment to MetaHuman facial morph targets / DNA control curves via `UMetaHumanEmotionDriverComponent`.
- Crossfades between emotion presets (brow, cheeks, eye squint, nose wrinkle, mouth corners) so MetaHuman "Amelia" expresses the detected emotion on her face.
- Layer 1 (base speech/lip animation) is owned by MetaHuman's audio-driven AnimBP; Layer 2 (this plugin) is an emotion overlay applied on top.

## Architecture Overview
```
speech-emotion-recognition/              (monorepo root)
├── src/                                 Python backend
│   ├── api/        FastAPI app (main.py, config, schemas, errors, deps)
│   ├── model/      SpeechBrain IEMOCAP wrapper + canonical label mapping
│   ├── timeline/   Windowing, smoothing, merging → segment timelines
│   └── audioio/    Audio loading / resampling
├── apps/streamlit_app/                  Streamlit UI for manual testing
├── scripts/run_api.sh                   uvicorn launcher (PYTHONPATH=src)
├── docker-compose.yml                   Containerised backend
├── requirements/{base,backend,frontend,dev}.txt
└── unreal/                              Unreal Engine 5.7 project
    ├── EmotionDemo.uproject             Engine 5.7, enables EmotionBridge + MetaHumanCharacter
    ├── Content/
    │   ├── MetaHumans/Amelia/           MetaHuman character (Face mesh, Post-Process ABP)
    │   ├── Blueprints/Amelia_FaceMesh.uasset
    │   ├── Maps/EmotionDemoMap.umap
    │   └── Python/SetupAmeliaEmotionDriver.py   Editor-side setup helper
    └── Plugins/EmotionBridge/
        ├── EmotionBridge.uplugin         Depends on AudioCapture; runtime + editor modules
        └── Source/
            ├── EmotionBridge/            Runtime module (C++)
            │   ├── Public/  EmotionApiClient.h, EmotionPlaybackComponent.h,
            │   │            EmotionLampActor.h, EmotionColorComponent.h,
            │   │            EmotionTakeStore.h, EmotionTakeTypes.h,
            │   │            EmotionTimelineTypes.h, EmotionMetaHumanTypes.h,
            │   │            MetaHumanEmotionDriverComponent.h,
            │   │            MetaHumanEmotionAnimInstance.h,
            │   │            EmotionBridgeSettings.h, EmotionBridgeLog.h
            │   └── Private/ matching .cpp implementations
            └── EmotionBridgeEditor/      Editor-only module (Slate panel, Take Library UI)
```

**Data flow:**
`WAV → EmotionApiClient (HTTP multipart POST) → /timeline/unreal → FEmotionTimelineResponse →
 EmotionPlaybackComponent ticks segments → UMetaHumanEmotionDriverComponent::ApplyEmotion(label, confidence) →
 blend + scale morph weights → UMetaHumanEmotionAnimInstance writes CTRL_expressions.* curves →
 MetaHuman Post-Process ABP / RigLogic DNA solver deforms Amelia's face.`

## Key File Locations
- Repo root: [.](.)
- Python backend entrypoint: [src/api/main.py](src/api/main.py)
- Canonical emotion labels: [src/model/labels.py](src/model/labels.py)
- API config: [src/api/config.py](src/api/config.py)
- API launcher: [scripts/run_api.sh](scripts/run_api.sh)
- Unreal project file: [unreal/EmotionDemo.uproject](unreal/EmotionDemo.uproject)
- Plugin descriptor: [unreal/Plugins/EmotionBridge/EmotionBridge.uplugin](unreal/Plugins/EmotionBridge/EmotionBridge.uplugin)
- Plugin runtime source: [unreal/Plugins/EmotionBridge/Source/EmotionBridge/](unreal/Plugins/EmotionBridge/Source/EmotionBridge/)
- Plugin editor source: [unreal/Plugins/EmotionBridge/Source/EmotionBridgeEditor/](unreal/Plugins/EmotionBridge/Source/EmotionBridgeEditor/)
- MetaHuman Amelia assets: [unreal/Content/MetaHumans/Amelia/](unreal/Content/MetaHumans/Amelia/)
- Editor Python setup script: [unreal/Content/Python/SetupAmeliaEmotionDriver.py](unreal/Content/Python/SetupAmeliaEmotionDriver.py)
- Integration guide: [docs/UNREAL_INTEGRATION.md](docs/UNREAL_INTEGRATION.md)
- Phase 2B design notes: [docs/METAHUMAN_PHASE2B.md](docs/METAHUMAN_PHASE2B.md)

## Plugin Module Names
From [EmotionBridge.uplugin](unreal/Plugins/EmotionBridge/EmotionBridge.uplugin):
- `EmotionBridge` — Runtime module (LoadingPhase: Default)
- `EmotionBridgeEditor` — Editor module (LoadingPhase: PostEngineInit)

Plugin dependency: `AudioCapture` (enabled).
Project plugins: `EmotionBridge`, `MetaHumanCharacter`.

## Emotion API
- Runtime: Python 3.x + FastAPI + uvicorn (SpeechBrain IEMOCAP model on CPU by default).
- Base URL: `http://localhost:8000`
- Docs UI: `http://localhost:8000/docs`
- Endpoints:
  - `GET /health` — `{ status, model_id, device }`
  - `POST /predict` — multipart `file=<wav>` → `{ emotion, confidence, scores?, model_name, duration_sec }`
  - `POST /timeline` — full timeline with windowing/smoothing/merging.
  - `POST /timeline/unreal` — simplified UE-friendly contract: `{ duration_sec, segments: [{start_sec, end_sec, emotion, confidence}] }`
- Canonical emotion labels: `neutral, happy, sad, angry, fear, disgust, surprise`.
  Baseline IEMOCAP model only predicts the first four; the others always come back at 0.0.
- How to start (Unix shell — use Git Bash on Windows):
  ```bash
  ./scripts/run_api.sh --reload
  # or: PYTHONPATH=src uvicorn src.api.main:app --host 0.0.0.0 --port 8000
  # or: docker compose up api
  ```

## MetaHuman Integration
- Character: "Amelia" under [unreal/Content/MetaHumans/Amelia/](unreal/Content/MetaHumans/Amelia/).
- Face mesh: `/Game/MetaHumans/Amelia/Face/Amelia_FaceMesh`.
- Post-Process AnimBP: `/Game/MetaHumans/Amelia/Face/ABP_Amelia_FaceMesh_PostProcess` (RigLogic / DNA solver).
- Blendshape convention: **ARKit 52** names in presets (e.g. `browDown_L`, `cheekSquint_R`, `mouthSmile_L`).
  At runtime these are mapped to MetaHuman DNA raw control names of the form
  `CTRL_expressions.<control><Side>` (e.g. `CTRL_expressions.browDownL`) and written as
  animation curves by `UMetaHumanEmotionAnimInstance::NativeUpdateAnimation` via `AddCurveValue`.
- Emotion-to-curve mapping: defined in C++ via `UMetaHumanEmotionDriverComponent::MakeDefaultPresets()` and overridable per-actor on the component's `ExpressionPresets` array.
- Lip sync / speech animation: **Layer 1** uses MetaHuman's built-in audio-driven face AnimBP — not reimplemented here. Presets deliberately avoid jaw / primary lip phoneme curves so they don't fight lip sync (see `EmotionMetaHumanTypes.h` "SPEECH SAFETY RULE").
- Blending: crossfade between `FromEmotion → ToEmotion` over `FEmotionOverlaySettings::BlendDurationSec` (default 0.4 s), optionally weighted by API confidence.

## Build Instructions
- Unreal Engine: **5.7** (`EngineAssociation` in `EmotionDemo.uproject`).
- Windows C++ plugin build:
  1. Right-click `unreal/EmotionDemo.uproject` → **Generate Visual Studio project files**.
  2. Open `unreal/EmotionDemo.sln` in Visual Studio 2022.
  3. Set configuration to `Development Editor | Win64`, build the `EmotionDemo` target.
  4. Launch `UnrealEditor.exe "<abs>/unreal/EmotionDemo.uproject"` (or double-click the uproject).
- Backend (from repo root):
  ```bash
  pip install -r requirements/backend.txt
  ./scripts/run_api.sh --reload           # or docker compose up api
  ```
- End-to-end: start API → open Unreal Editor → Window → Emotion Bridge panel → select WAV → Analyze → Play Demo (optionally Bind Selected Actor = Amelia).

## Current State / Known Issues
- Current branch: `feat/metahuman-integration` (MetaHuman overlay is the active work).
- Uncommitted edits touch: `MetaHumanEmotionDriverComponent.{cpp,h}`, `DefaultEngine.ini`, Amelia assets,
  new `MetaHumanEmotionAnimInstance.{cpp,h}`, and `Content/Python/SetupAmeliaEmotionDriver.py`.
- IEMOCAP model only emits 4 canonical labels; presets for `fear / disgust / surprise` exist in the canonical label list but won't be driven by the current backend.
- The PostProcess ABP on `Amelia_FaceMesh` can be cleared by MetaHuman reimports; re-run the editor Python helper to restore it.
- If MetaHuman face shows zero deformation: verify PostProcess ABP is set, Python plugin is enabled, and the editor panel successfully bound the actor.

## Coding Conventions
- **C++**: Unreal standard (F/U/A/E prefixes, UCLASS/USTRUCT/UPROPERTY/UFUNCTION macros, `MODULE_API` export, `GENERATED_BODY()`). File header format visible in existing headers ("Copyright (c) EmotionDemo Project. All rights reserved.").
- **Python**: PEP8, type hints, FastAPI + pydantic v2, logging via stdlib.
- **Commits**: Conventional-commit style seen in history: `feat(unreal): …`, `feat(api): …`, `fix(...)`.
- Branch for current work: `feat/metahuman-integration`; PRs target `main`.

## What NOT to Touch
- `unreal/Intermediate/`, `unreal/Binaries/`, `unreal/DerivedDataCache/`, `unreal/Saved/` — engine-generated, regenerated on build.
- `unreal/EmotionDemo.sln`, `unreal/.vs/`, `unreal/UpgradeLog.htm` — IDE / upgrader artefacts.
- `unreal/Backup/` — auto-created asset recovery snapshots.
- `unreal/Content/MetaHumans/Common/` and most of `unreal/Content/MetaHumans/Amelia/` — vendor-generated MetaHuman assets; regenerate via MetaHuman Creator, don't hand-edit.
- `.venv/`, `__pycache__/`, `node_modules/`, `.cache/`, `hf_cache/` — caches.
- `*.wav` under repo root (gitignored test audio).
