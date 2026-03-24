# Unreal Engine Integration — EmotionDemo

This folder contains a complete Unreal Engine 5 project that demonstrates the
`speech-emotion-recognition` backend from inside the Unreal Editor.

**Goal:** Select a WAV file → send it to the local Python backend → parse the returned
emotion timeline → watch a lamp actor change color as each emotion segment plays.

---

## What this integration does

1. The Unreal Editor hosts a custom dockable tab called **Emotion Bridge**.
2. You browse for a WAV file and optionally tune the timeline parameters.
3. Clicking **Analyze** POSTs the file to `http://localhost:8000/timeline`
   (multipart/form-data) via the UE HTTP module.
4. The JSON response is parsed into a list of segments, each with start time,
   end time, emotion label, and confidence.
5. The segment list is shown in the UI.
6. Clicking **Play Demo** starts a wall-clock simulation: the panel checks which
   segment is active each frame and calls `ApplyEmotion()` on an
   `AEmotionLampActor`, changing its point-light color.

---

## Why API + Unreal instead of running ML inside Unreal?

- ML inference (SpeechBrain, PyTorch) requires Python and large model weights (~1 GB).
  Running these inside a C++ game engine is not practical.
- The REST API separation means the backend can be improved, swapped, or scaled
  independently without touching Unreal code.
- The Unreal side stays thin: HTTP call + JSON parse + color change.
  This is the pattern used in production real-time entertainment systems.

---

## Folder layout

```
unreal/
├── EmotionDemo.uproject           — UE project descriptor
├── Config/
│   ├── DefaultEngine.ini
│   └── DefaultGame.ini
├── Source/
│   ├── EmotionDemo.Target.cs      — game build target
│   ├── EmotionDemoEditor.Target.cs— editor build target
│   └── EmotionDemo/
│       ├── EmotionDemo.Build.cs
│       ├── EmotionDemo.h
│       └── EmotionDemo.cpp        — IMPLEMENT_PRIMARY_GAME_MODULE
└── Plugins/
    └── EmotionBridge/
        ├── EmotionBridge.uplugin
        ├── README.md
        ├── MANUAL_ASSET_SETUP.md  ← READ THIS before running the demo
        └── Source/
            ├── EmotionBridge/         — runtime module (HTTP, types, actors)
            └── EmotionBridgeEditor/   — editor module (Slate UI, tab)
```

---

## Prerequisites (macOS)

| Tool | Version | Where to get |
|------|---------|--------------|
| Unreal Engine | 5.4 or later | Epic Games Launcher |
| Xcode | 15.x or later | Mac App Store |
| Xcode Command Line Tools | Current | `xcode-select --install` |
| Python 3.11+ | Any | See repo root README |
| Docker (optional) | Latest | docker.com |

> The backend does **not** run inside Unreal. Start it separately before clicking Analyze.

---

## How to start the backend

### Option A — Docker (recommended)

```bash
cd /path/to/speech-emotion-recognition
docker compose up api
```

Wait until you see `Uvicorn running on http://0.0.0.0:8000`.

### Option B — Local Python

```bash
cd /path/to/speech-emotion-recognition
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/backend.txt
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Verify

```bash
curl http://localhost:8000/health
# Expected: {"status": "ok", ...}
```

> **First run note:** The SpeechBrain model (~350 MB) is downloaded from HuggingFace on the
> first `/predict` or `/timeline` request. This can take 2–5 minutes depending on your
> connection. Subsequent requests are fast. The EmotionBridge panel has a 180-second
> HTTP timeout to handle this gracefully.

---

## How to open and build the Unreal project on macOS

### 1. Generate Xcode project files

Right-click `unreal/EmotionDemo.uproject` in Finder and choose
**Services → Generate Xcode Project** (or use the context menu from the launcher).

Alternatively, from a terminal:
```bash
/path/to/UE_5.4/Engine/Build/BatchFiles/Mac/GenerateProjectFiles.sh \
    /path/to/speech-emotion-recognition/unreal/EmotionDemo.uproject -game
```

This creates `unreal/EmotionDemo.xcworkspace`.

### 2. Build in Xcode

```bash
open unreal/EmotionDemo.xcworkspace
```

Select target **EmotionDemoEditor** → scheme **EmotionDemoEditor** → click Build (⌘B).

The first build downloads UE engine intermediates and takes 10–20 minutes.
Subsequent builds are incremental (1–3 minutes).

### 3. Open the project in Unreal Editor

Double-click `unreal/EmotionDemo.uproject`, or run:
```bash
"/path/to/UE_5.4/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor" \
    /path/to/speech-emotion-recognition/unreal/EmotionDemo.uproject
```

---

## How to open and build the Unreal project on Windows

### Prerequisites (Windows)

| Tool | Version | Where to get |
| ---- | ------- | ------------ |
| Unreal Engine | 5.4 or later | Epic Games Launcher |
| Visual Studio | 2022 Community or higher | [visualstudio.microsoft.com](https://visualstudio.microsoft.com) |
| VS Workloads | "Game development with C++" + ".NET desktop" | VS Installer |
| Windows SDK | 10.0.18362.0 or later | VS Installer → Individual Components |
| Python 3.11+ | Any | See repo root README |

> During Visual Studio installation, select the **"Game development with C++"** workload.
> It installs the MSVC compiler, Windows SDK, and CMake — all required by Unreal Build Tool.

### 1. Generate Visual Studio project files

Open a **Developer Command Prompt for VS 2022** (or any terminal with `msbuild` in PATH),
then run:

```bat
"C:\Program Files\Epic Games\UE_5.4\Engine\Build\BatchFiles\GenerateProjectFiles.bat" ^
    "C:\path\to\speech-emotion-recognition\unreal\EmotionDemo.uproject" -game
```

Replace `UE_5.4` with your installed engine version (e.g. `UE_5.7`).

This creates `unreal\EmotionDemo.sln`.

Alternatively, right-click `EmotionDemo.uproject` in File Explorer and choose
**Generate Visual Studio project files** (requires the engine's shell extension to be
registered, which happens automatically during engine installation).

### 2. Build in Visual Studio

```bat
start unreal\EmotionDemo.sln
```

In Visual Studio:

1. Set the **Solution Configuration** to `Development Editor`.
2. Set the **Solution Platform** to `Win64`.
3. In Solution Explorer, right-click **EmotionDemo** → **Build**.

The first build takes 15–30 minutes. Incremental rebuilds are 1–5 minutes.

You can also build from the command line without opening the IDE:

```bat
msbuild unreal\EmotionDemo.sln /p:Configuration="Development Editor" /p:Platform=Win64 /m
```

### 3. Launch the Unreal Editor (Windows)

Double-click `unreal\EmotionDemo.uproject`, or run:

```bat
"C:\Program Files\Epic Games\UE_5.4\Engine\Binaries\Win64\UnrealEditor.exe" ^
    "C:\path\to\speech-emotion-recognition\unreal\EmotionDemo.uproject"
```

### Windows-specific notes

- **Firewall prompt:** Windows may ask to allow `UnrealEditor.exe` network access when the
  panel makes its first HTTP request. Click **Allow** to permit connections to `localhost:8000`.
- **Long path errors:** Enable long path support (`gpedit.msc` →
  Computer Configuration → Administrative Templates → System → Filesystem →
  Enable Win32 long paths) if the build fails with `PathTooLong` errors.
- **Audio playback:** The panel plays WAV files via PowerShell's `System.Media.SoundPlayer`,
  which is built into all Windows versions — no extra software needed.
- **Microphone access:** Windows 10/11 requires microphone permission for the recording
  feature. Check **Settings → Privacy → Microphone** and ensure the toggle is on.

---

## How to open the Emotion Bridge tab

In the main editor menu bar: **Window → Emotion Bridge**

The tab is a "nomad tab" — it can float, dock, or be stacked with other panels.

---

## End-to-end demo walkthrough

1. Start the backend (`docker compose up api`).
2. Open the project in Unreal Editor.
3. Open **Window → Emotion Bridge**.
4. Click **Health Check** — status should say "Backend is healthy".
5. Click **Browse** and select any `.wav` speech file.
6. Leave parameters at defaults (or adjust as needed).
7. Click **Analyze** and wait (first run: ~2 min; subsequent: ~5 s).
8. The **Results** section shows the segment list with emotion labels and timing.
9. In the editor viewport, open or create a level (see `MANUAL_ASSET_SETUP.md`).
10. Click **Play Demo** — the lamp actor's point light cycles through emotion colors.
11. Click **Stop Demo** to halt and reset to white.

---

## Getting a sample WAV

- The repo's `tests/` directory may contain fixtures.
- Record your voice with QuickTime Player (macOS) → Export As → Audio Only → `.m4a`,
  then convert: `ffmpeg -i recording.m4a -ar 16000 -ac 1 sample.wav`
- Download a public speech sample from LibriSpeech or CMU Arctic.

The backend expects **mono, 16 kHz WAV** but will accept any standard WAV and
auto-resample internally.

---

## Running the animation in PIE (Play In Editor)

The **Emotion Bridge panel** drives the lamp in the editor world (no PIE required). However,
if you want the emotion colors to animate while the game is actually running — in PIE or a
packaged build — use the `UEmotionPlaybackComponent` that ships with the plugin.

### What UEmotionPlaybackComponent does

It is a standard `UActorComponent` that:

- Accepts a `FEmotionTimelineResponse` struct (populated by the panel's **Analyze** step).
- Accumulates time each `TickComponent` and calls `ApplyEmotion()` on a target
  `AEmotionLampActor` when the active segment changes.
- Works identically in PIE, packaged builds, and on both macOS and Windows.

### Setup

1. In the editor, analyze a WAV file in the **Emotion Bridge** panel.
1. The panel stores the last parsed timeline in `CurrentTimeline` — it also drives the
   editor-world lamp during **Play Demo**.
1. To animate the lamp during PIE, select the `AEmotionLampActor` in the viewport, click
   **+ Add Component** in the Details panel, and add **EmotionPlaybackComponent**. Then set
   **Target Lamp Actor** to the lamp in the level.
1. In your game's **BeginPlay** (Blueprint or C++), call:

**Blueprint:**

```text
Get EmotionPlaybackComponent → Set Timeline (the FEmotionTimelineResponse from your C++ code)
Get EmotionPlaybackComponent → Start Playback
```

**C++ (in your AGameMode or APawn BeginPlay):**

```cpp
UEmotionPlaybackComponent* Comp = LampActor->FindComponentByClass<UEmotionPlaybackComponent>();
if (Comp)
{
    Comp->SetTimeline(MyParsedTimeline);   // FEmotionTimelineResponse
    Comp->StartPlayback();
}
```

1. Press **Play** (Alt+P) in the Unreal Editor. The lamp cycles through emotion colors in sync
   with the timeline while the game runs. Press **Stop** to end PIE.

### Running the full loop in PIE

The recommended flow for a live demo:

1. Start the backend (`docker compose up api`).
2. Open **Window → Emotion Bridge** in the editor.
3. Record a clip with the **Record** button (or Browse for a WAV).
4. Click **Analyze** — wait for results.
5. Click **Play Demo** to preview in the editor world (no PIE needed).
6. Press **Alt+P** to enter PIE — `UEmotionPlaybackComponent` replays the same
   timeline inside the game world. The audio plays simultaneously via
   `afplay` (macOS) or PowerShell `SoundPlayer` (Windows).
7. Press **Escape** or **Stop** to exit PIE.

> **Tip:** You do not need to re-analyze between PIE sessions.
> The parsed timeline persists in memory until you close the editor or click Analyze again.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `EmotionBridge` not in Window menu | Editor module failed to load | Check Output Log for compile errors; rebuild in Xcode |
| Xcode project won't generate | Engine path not in PATH | Use full path to `GenerateProjectFiles.sh` |
| Build fails: `EmotionBridgeEditor.Build.cs` module not found | Plugin not discovered | Confirm `Plugins/EmotionBridge/EmotionBridge.uplugin` exists |
| Build fails: `WorkspaceMenuStructure` not found | UE version mismatch | Ensure engine is 5.4+; check `Build.cs` dependency list |
| Health Check returns red | Backend not running | `docker compose up api` |
| Analyze hangs for >3 min | First model download | Wait; progress is visible in Docker logs |
| Analyze fails: "Cannot read WAV file" | Wrong path typed | Use Browse button for correct absolute path |
| Analyze fails: HTTP 422 | Malformed request | Check UE Output Log for the raw request details |
| Segment list empty | Backend returned 0 segments | File may be too short (< 2 s); try a longer recording |
| Lamp not changing color | No level open | Create a level; see `MANUAL_ASSET_SETUP.md` |
| Lamp light changes but mesh does not | Material has no `BaseColor` param | See Optional material setup in `MANUAL_ASSET_SETUP.md` |
| Actor spawns underground | Level has terrain at y=0 | Place a lamp actor manually at a visible location |

---

## Future extensions

- **In-game HUD widget:** Replace the editor tab with a `UUserWidget` for use in PIE/packaged builds.
- **MetaHuman / blendshapes:** Map emotion labels to ARKit blend shape curves on a
  MetaHuman face mesh using the `LiveLink` module.
- **Lip sync:** Integrate `OVRLipSync` or UE's built-in audio-driven animation after
  adding audio playback (step 1: play the WAV via `UAudioComponent`).
- **Multi-character:** Iterate over multiple lamp/character actors and apply the timeline
  to each in a round-robin or tagged fashion.
- **Cloud backend:** Change `ApiBaseUrl` to a deployed FastAPI instance;
  no Unreal code changes needed.
