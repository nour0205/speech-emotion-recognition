# EmotionBridge Plugin

An Unreal Engine 5 plugin that connects a running instance of the
`speech-emotion-recognition` Python backend to the Unreal Editor via HTTP.

This plugin is the core of the `unreal/` integration in the monorepo.

---

## Module overview

| Module | Type | Purpose |
|--------|------|---------|
| `EmotionBridge` | Runtime | HTTP client, JSON types, settings, lamp actor, playback component, **Take persistence (Phase 2A)** |
| `EmotionBridgeEditor` | Editor-only | Dockable Slate tab, segment list UI, playback, **Take Library UI (Phase 2A)** |

---

## Phase 2A — Take Library

Phase 2A adds a persistent **Take Library** to the editor panel.
A *Take* is one saved analysis session: source audio + timeline + params + metadata.

**What you can do:**
- Analyze a WAV file → click **Save Take** → give it a name
- Browse all saved Takes in the Take Library panel below the playback area
- Select a Take → click **Load** to restore its timeline without hitting the backend
- Select a Take → click **Play** to load and immediately start playback
- **Reanalyze** a Take using the original (or copied) audio + stored params
- **Duplicate** a Take before reanalysing to preserve the old result
- **Delete** a Take permanently from disk (with confirmation dialog)
- Search by name, sort by created/duration/name, filter by dominant emotion

**Storage location:**
```
<UnrealProject>/Saved/EmotionBridge/Takes/<TakeId>/
    metadata.json      — identity, timestamps, tags, notes, Phase 2B placeholders
    timeline.json      — full /timeline/unreal response
    params.json        — analysis parameters used
    audio/
        source.wav     — optional copy of the source WAV
```

**Reanalysis policy:** reanalysis *overwrites* the existing take in-place
(timeline + params + UpdatedAt replaced; name/notes/tags kept).
Use **Duplicate** first if you want to preserve the current result.

See [TAKE_LIBRARY.md](TAKE_LIBRARY.md) for full documentation.

---

## Runtime classes

### `FEmotionApiClient`
Plain C++ HTTP client. Constructs multipart/form-data requests manually because UE's
`FHttpModule` has no built-in multipart helper. Timeouts: 5 s for `/health`,
180 s for `/timeline` (model download can take ~2 min on first run).

Phase 2A: also parses `model_name` and `sample_rate` from the response and stores
them in `FEmotionTimelineResponse` for take provenance tracking.

### `UEmotionBridgeSettings`
`UDeveloperSettings` subclass stored in `Config/EmotionBridge.ini`.
Editable at **Edit › Project Settings › Plugins › Emotion Bridge**.
Exposes: API base URL, timeline defaults, `TMap<FString, FLinearColor>` emotion colors.

### `AEmotionLampActor`
A simple actor (cube mesh + point light) that reacts to emotion strings.
Call `ApplyEmotion("happy", 0.9f)` — the light and mesh both change color.
The editor panel auto-spawns one if none is found in the level.

### `UEmotionPlaybackComponent`
Attach to any actor. Call `SetTimeline()`, then `StartPlayback()`.
Drives a `TargetLampActor` from the timeline data using `TickComponent`.

### `FEmotionTakeRecord` (Phase 2A)
The central take data structure. Stores audio paths, timeline, params, metadata,
and Phase 2B placeholder fields. Serialised to JSON files on disk.

### `FEmotionTakeStore` (Phase 2A)
Static service class for all take disk I/O (save, load, loadAll, delete, duplicate).
Uses a temp-then-rename write strategy for safe file writes.

---

## Editor UI

Open via **Window › Emotion Bridge**.

The panel is a single `SCompoundWidget` (pure Slate/C++ — no Blueprints) with:
- **Backend** — URL field, Health Check button
- **File** — WAV picker with native file-open dialog, microphone recording
- **Parameters** — window/hop/pad/smoothing controls
- **Analyze** — fires the `/timeline/unreal` HTTP request
- **Results** — metadata + scrollable segment list (color-coded by emotion)
- **Playback** — Play Demo / Stop Demo / Focus in Viewport
- **Save Take** (Phase 2A) — name field + Save Take button (enabled after analysis)
- **Take Library** (Phase 2A) — list, search/filter/sort, Load/Play/Delete/Duplicate/Reanalyze

---

## Color defaults

| Emotion | Color |
|---------|-------|
| angry | Red `(1.0, 0.08, 0.08)` |
| happy | Yellow `(1.0, 0.90, 0.05)` |
| sad | Blue `(0.1, 0.25, 1.0)` |
| neutral | White `(1.0, 1.0, 1.0)` |

Override any of these in Project Settings > Emotion Bridge without recompiling.

---

## Compatibility

- Unreal Engine 5.4, 5.5, 5.6, 5.7
- macOS (Xcode build) — primary development target
- Windows (MSVC) — compile-compatible, not tested

---

## Dependencies (added automatically by Build.cs)

Runtime: `Core`, `CoreUObject`, `Engine`, `HTTP`, `Json`, `JsonUtilities`, `DeveloperSettings`
Editor: `Slate`, `SlateCore`, `UnrealEd`, `ToolMenus`, `DesktopPlatform`, `LevelEditor`

---

See [MANUAL_ASSET_SETUP.md](MANUAL_ASSET_SETUP.md) for the one-time steps required inside the Unreal Editor.
See [TAKE_LIBRARY.md](TAKE_LIBRARY.md) for Phase 2A Take Library documentation.
