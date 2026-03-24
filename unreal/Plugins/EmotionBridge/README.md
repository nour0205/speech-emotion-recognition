# EmotionBridge Plugin

An Unreal Engine 5 plugin that connects a running instance of the
`speech-emotion-recognition` Python backend to the Unreal Editor via HTTP.

This plugin is the core of the `unreal/` integration in the monorepo.

---

## Module overview

| Module | Type | Purpose |
|--------|------|---------|
| `EmotionBridge` | Runtime | HTTP client, JSON types, settings class, lamp actor, playback component |
| `EmotionBridgeEditor` | Editor-only | Dockable Slate tab, segment list UI, playback wiring |

---

## Runtime classes

### `FEmotionApiClient`
Plain C++ HTTP client. Constructs multipart/form-data requests manually because UE's
`FHttpModule` has no built-in multipart helper. Timeouts: 5 s for `/health`,
180 s for `/timeline` (model download can take ~2 min on first run).

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

---

## Editor UI

Open via **Window › Emotion Bridge**.

The panel is a single `SCompoundWidget` (pure Slate/C++ — no Blueprints) with:
- **Backend** — URL field, Health Check button
- **File** — WAV picker with native file-open dialog
- **Parameters** — window/hop/pad/smoothing controls
- **Analyze** — fires the `/timeline` HTTP request
- **Results** — metadata + scrollable segment list (color-coded by emotion)
- **Playback** — Play Demo / Stop Demo

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

See `MANUAL_ASSET_SETUP.md` for the one-time steps required inside the Unreal Editor.
