# EmotionBridge — Phase 2A: Take Library

This document covers everything added in Phase 2A of the EmotionBridge plugin.

Phase 2A is about **persistence, management, and reloadability** of analysis results.

---

## What Phase 2A Adds

Before Phase 2A, you could analyse a WAV file and see the timeline in the editor panel.
When you closed the tab or reopened Unreal, the result was gone.

Phase 2A adds a **Take Library** — a persistent, on-disk store of analysis sessions.

A *Take* captures:

- The source audio file identity
- A copy of the WAV (optional but recommended)
- The full `/timeline/unreal` response
- The exact analysis parameters used
- Timestamps, display name, notes, and tags
- Phase 2B placeholder fields for future MetaHuman integration

You can save multiple Takes, browse them later, reload any Take without calling the
backend again, and replay the emotion timeline in the editor.

---

## Where Takes Are Stored

```
<UnrealProject>/Saved/EmotionBridge/Takes/
    <TakeId>/
        metadata.json      — identity, timestamps, paths, tags, notes, phase2b
        timeline.json      — full serialised timeline response
        params.json        — analysis parameters (window, hop, pad, smoothing…)
        audio/
            source.wav     — optional copy of the source WAV
```

`<UnrealProject>` is the folder containing your `.uproject` file, e.g. `unreal/EmotionDemo/`.

`<TakeId>` is a 32-character hex string derived from `FGuid::NewGuid()` — e.g.
`A1B2C3D4E5F60718293A4B5C6D7E8F90`.

The `Saved/` directory is excluded from version control by the default `.gitignore` for
Unreal projects. This is intentional — takes are **project-local working data**, not
source-controlled assets. If you want to share a take, copy its folder manually.

---

## JSON File Formats

### metadata.json

```json
{
  "schema_version": 1,
  "plugin_version": "1.0",
  "take_id": "A1B2C3D4E5F60718293A4B5C6D7E8F90",
  "display_name": "Actor02_angry_take3",
  "created_at": "2026-03-30T12:00:00.000Z",
  "updated_at": "2026-03-30T12:00:00.000Z",
  "source_audio_path": "/Users/apple/audio/actor02_angry.wav",
  "copied_audio_path": "audio/source.wav",
  "notes": "",
  "tags": [],
  "phase2b": {
    "cleaned_audio_path": "",
    "sound_wave_asset": "",
    "metahuman_performance_asset": "",
    "level_sequence_asset": "",
    "emotion_preset_mapping": ""
  }
}
```

All `phase2b` fields are empty strings in Phase 2A. They exist to make the
storage format forward-compatible with Phase 2B (MetaHuman integration).

### timeline.json

```json
{
  "type": "timeline",
  "source": "ser_api",
  "version": "1.0",
  "model_name": "speechbrain-iemocap",
  "sample_rate": 16000,
  "duration_sec": 10.5,
  "segments": [
    { "start_sec": 0.0,  "end_sec": 3.5,  "emotion": "neutral", "confidence": 0.82 },
    { "start_sec": 3.5,  "end_sec": 7.0,  "emotion": "happy",   "confidence": 0.91 },
    { "start_sec": 7.0,  "end_sec": 10.5, "emotion": "neutral", "confidence": 0.78 }
  ]
}
```

This is the exact response from the backend, extended with `model_name` and `sample_rate`
(Phase 2A additions to `FEmotionTimelineResponse`).

### params.json

```json
{
  "window_sec": 2.0,
  "hop_sec": 0.5,
  "pad_mode": "zero",
  "smoothing_method": "hysteresis",
  "hysteresis_min_run": 3,
  "majority_window": 5,
  "ema_alpha": 0.6
}
```

These are the exact parameters that were sent to `/timeline/unreal`. Storing them
means Reanalyze can reproduce the exact same call, and you always know what settings
produced a given result.

---

## Schema Versioning

Every `metadata.json` has `"schema_version": 1`. When the Phase 2A storage layout
changes in a future release:

1. The version will be incremented.
2. `FEmotionTakeStore::LoadTake()` will check the version and apply a migration.
3. Old takes remain readable; new fields default to safe empty values.

---

## Using the Take Library

### Step 1 — Analyse a WAV

Open the **Emotion Bridge** panel (**Window › Emotion Bridge**).
Select a WAV file, configure parameters, click **Analyze**.
The backend runs; results appear in the segment list.

### Step 2 — Save a Take

After analysis succeeds, the **SAVE TAKE** section at the bottom of the panel
becomes active.

1. Optionally type a name in the **Take Name** field (e.g. `Actor02_angry_take3`).
   Leave it blank for an auto-generated name like `Take_20260330_120015`.
2. Click **Save Take**.

The plugin:
- Creates `Saved/EmotionBridge/Takes/<TakeId>/`
- Writes `metadata.json`, `timeline.json`, `params.json`
- Copies the source WAV to `audio/source.wav`
- Refreshes the Take Library list and selects the new take

A green status message confirms success with the first 8 chars of the Take ID.

### Step 3 — Browse the Take Library

The **TAKE LIBRARY** section is always visible below the Save Take section.

The list shows all saved takes with columns:

| Column | Description |
|--------|-------------|
| Name | Display name you assigned |
| Created | Date of analysis (YYYY-MM-DD) |
| Duration | Total audio length |
| Emotion | Dominant emotion (color-coded) |
| Source File | Clean filename of the source WAV |

**Search** — type in the search box to filter by name (case-insensitive).

**Sort** — choose from: Name (A→Z), Created (newest), Created (oldest), Duration (longest).

**Filter** — filter to only show takes with a specific dominant emotion.

**Refresh** — reload all takes from disk (useful if you manually copied take folders).

### Step 4 — Load a Take

Select a take in the list. Click **Load**.

The plugin restores:
- The full emotion timeline (segments, durations, confidence scores)
- All analysis parameters into the panel controls
- The WAV path (prefers the embedded copy in `audio/source.wav`)
- The metadata display text

No backend call is made. The take is fully self-contained.
Click **Play Demo** after loading to replay the timeline.

### Step 5 — Play a Take Directly

Select a take and click **Play**.
This is equivalent to Load + Play Demo in one click.

### Step 6 — Reanalyze a Take

If you want to re-run analysis on an existing take (e.g. you changed smoothing
settings, or the model was updated):

1. Select the take in the library.
2. Click **Reanalyze**.
3. The plugin calls `/timeline/unreal` using:
   - The take's embedded audio (if present) or the original source path
   - The take's stored analysis parameters
4. When the response arrives, the take is **updated in-place**:
   - `timeline.json` and `params.json` are replaced
   - `updated_at` timestamp is refreshed
   - Display name, notes, and tags are **preserved**
5. The Take Library refreshes.

**Reanalysis design choice: overwrite in-place.**
This keeps the library clean. If you want to preserve the current result,
click **Duplicate** first to create a copy, then reanalyze the original.

**Note:** if the backend is offline, reanalysis will fail with a connection error.
The take is not modified until a successful response arrives.

### Step 7 — Duplicate a Take

Select a take and click **Duplicate**.

A new take is created with:
- A new TakeId
- Display name: `<original name>_Copy`
- Same timeline, params, audio, notes, tags
- `updated_at` set to now; `created_at` copied from the source

The new take appears in the library immediately.

### Step 8 — Delete a Take

Select a take and click **Delete**.

A confirmation dialog appears:
> "Permanently delete take "Actor02_angry_take3"?
> This will remove all files in: ...
> This cannot be undone."

Click **Yes** to delete. The take folder and all its files are removed from disk.

---

## Detail Panel

When a take is selected, the detail panel below the action buttons shows:

```
ID:           A1B2C3D4E5F60718293A4B5C6D7E8F90
Name:         Actor02_angry_take3
Created:      2026-03-30T12:00:00.000Z
Updated:      2026-03-30T12:00:00.000Z
Duration:     10.50 s  |  Sample rate: 16000 Hz  |  Segments: 3
Dominant:     neutral  |  Avg confidence: 84%
Distribution: neutral 52%  happy 33%  angry 15%
Source audio: /Users/apple/audio/actor02_angry.wav  [exists]
Copied audio: .../Saved/EmotionBridge/Takes/.../audio/source.wav  [exists]
Analysis:     window=2.00s  hop=0.50s  pad=zero  smooth=hysteresis  minRun=3
Tags:         (none)
Notes:        (none)
Schema v1  |  Plugin v1.0
```

The `[exists]` / `[MISSING]` indicators tell you immediately whether the audio
files are still accessible.

---

## Phase 2B Readiness

The take format is designed to accommodate future MetaHuman integration without
breaking changes. All Phase 2B fields are stored in the `phase2b` block of
`metadata.json` and are empty strings in Phase 2A.

When Phase 2B is implemented, these fields will be populated:

| Field | Purpose |
|-------|---------|
| `cleaned_audio_path` | Noise-cleaned WAV from the denoising pipeline |
| `sound_wave_asset` | Content-browser path to the imported UE SoundWave |
| `metahuman_performance_asset` | MetaHuman Performance asset generated by facial solve |
| `level_sequence_asset` | Generated Level Sequence (facial + body animation) |
| `emotion_preset_mapping` | Emotion→morph-target preset mapping asset |

Existing takes will gain these fields when loaded and re-saved in Phase 2B.
The `schema_version` will be incremented to mark the transition.

---

## Architecture

### Runtime module (`EmotionBridge`)

| Class | File | Role |
|-------|------|------|
| `FEmotionAnalysisParams` | `EmotionTakeTypes.h` | Parameters sent to `/timeline/unreal` |
| `FEmotionTakeSummary` | `EmotionTakeTypes.h` | Computed stats (dominant emotion, distribution, avg confidence) |
| `FEmotionTakePhase2B` | `EmotionTakeTypes.h` | Phase 2B placeholder fields |
| `FEmotionTakeRecord` | `EmotionTakeTypes.h` | The central take data structure |
| `FEmotionTakeStore` | `EmotionTakeStore.h/.cpp` | All disk I/O (save, load, delete, duplicate) |

`FEmotionTakeSummary` is **never stored on disk**. It is recomputed by
`FEmotionTakeStore::ComputeSummary()` every time a take is loaded.
This avoids stale cached data and keeps `timeline.json` as the single source of truth.

### Editor module (`EmotionBridgeEditor`)

| Class | File | Role |
|-------|------|------|
| `SEmotionTakeLibrary` | `SEmotionTakeLibrary.h/.cpp` | Full Take Library Slate widget |

`SEmotionTakeLibrary` is embedded inside `SEmotionBridgePanel` via
`BuildTakeLibrarySection()`. It fires three delegates back to the panel:

- `OnLoadRequested` → `SEmotionBridgePanel::OnLoadTakeRequested()`
- `OnPlayRequested` → `SEmotionBridgePanel::OnPlayTakeRequested()`
- `OnReanalyzeRequested` → `SEmotionBridgePanel::OnReanalyzeTakeRequested()`

Delete and Duplicate are handled entirely within `SEmotionTakeLibrary` (no panel
involvement needed).

### Write safety

`FEmotionTakeStore::SafeWriteStringToFile()` writes every JSON file using a
temp-then-rename pattern:

1. Write content to `<filename>.json.tmp`
2. Rename `<filename>.json.tmp` → `<filename>.json`

A crash between steps 1 and 2 leaves a `.tmp` file, which is harmless and will be
overwritten on the next save. The final file is never in a partial state.

---

## Troubleshooting

### Take does not save

**Symptom:** clicking Save Take shows a red error status.

**Check:**
1. Output Log (`Window › Output Log`) for an `[EmotionBridge]` error line.
2. Verify that `Saved/EmotionBridge/Takes/` is writable.
   On macOS: check Disk Utility for available space; check file permissions.
3. Verify that the WAV file path in the panel is valid.
   The save can succeed even without audio copy — see next item.

**Audio copy warning (not fatal):**
If the source WAV is on a read-only volume or has been deleted, audio copy will
fail with a `Warning` in the log but the take is still saved without embedded audio.
The three JSON files are what matter for reloading.

---

### Take does not load

**Symptom:** selecting a take shows an error in the status bar, or the take does not
appear in the library after saving.

**Check:**
1. Open the take folder in Finder:
   `<UnrealProject>/Saved/EmotionBridge/Takes/<TakeId>/`
2. Verify that all three files exist: `metadata.json`, `timeline.json`, `params.json`.
3. Open each JSON file in a text editor and verify it is valid JSON.
   Common causes: disk full during write (partial file), manual editing errors.
4. Verify that `take_id` inside `metadata.json` is non-empty and matches the folder name.

If `timeline.json` is missing or malformed, the take cannot be loaded.
Reanalysis is the only recovery option (if audio is available).

---

### Source audio marked MISSING

**Symptom:** detail panel shows `[MISSING]` next to the source audio path.

**What it means:** the original WAV file that was on disk at analysis time
has been moved, renamed, or deleted.

**Recovery:**
- If `copied_audio_path` is non-empty and `[exists]`, the take can still
  be replayed and reanalysed using the embedded copy. No action needed.
- If both paths show `[MISSING]`, Reanalyze will fail.
  You must locate the WAV, set the path manually in the main panel's WAV field,
  and save a new take.

---

### Copied audio marked MISSING

**Symptom:** `copied_audio_path` is set but detail panel shows `[MISSING]`.

**Cause:** the `audio/source.wav` file inside the take folder was deleted externally.

**Fix:**
1. If the original source is still accessible (`[exists]`), click Reanalyze.
   Reanalysis will re-copy the audio on the next save.
2. Otherwise, locate the WAV, set the path in the main panel, reanalyze, and save
   as a new take (or copy the file manually into `audio/source.wav`).

---

### Corrupted JSON

**Symptom:** a take folder exists but the take does not appear in the library,
and the Output Log shows a JSON parse error.

**Check:**
1. Open the file in any text editor.
2. Use a JSON validator (e.g. `python3 -m json.tool metadata.json`).

**Recovery:**
- If `timeline.json` is corrupt: the take cannot be loaded.
  If audio is available, reanalyze and save a new take.
- If `metadata.json` is corrupt: edit it manually to fix the JSON syntax,
  then click Refresh in the library.
- If `params.json` is corrupt: replace with a valid params JSON using the
  default values shown in the JSON format section above.

---

### Duplicate take name

**Symptom:** you saved a take with the same name as an existing take.

**What happens:** take names are NOT enforced to be unique. Two takes can have
the same display name — they will have different TakeIds and different folders.
Both will appear in the library.

**To disambiguate:** add a note, tag, or rename by editing `display_name` in
`metadata.json` directly. A UI rename field will be added in a future version.

---

### Search or filter not updating

**Symptom:** you typed in the search box but the list is not filtering.

**Fix:** the list updates in real time as you type. If it is stuck:
1. Click **Refresh** to reload from disk.
2. Clear the search box and retype.
3. If the filter combo is set to a specific emotion, make sure your takes have that
   dominant emotion in their summary (visible in the detail panel).

---

### Reanalyze fails — backend offline

**Symptom:** clicking Reanalyze shows a yellow/red status "connection failed".

**Fix:**
1. Start the backend: `cd <repo>; docker compose up api`
   or: `uvicorn src.api.main:app --port 8000 --reload`
2. Use **Health Check** in the Backend section to verify the backend is reachable.
3. Click Reanalyze again.

The take is **not modified** until a successful response arrives. A failed
reanalysis attempt leaves the existing take data intact.

---

## File Reference

```
unreal/Plugins/EmotionBridge/
├── Source/
│   ├── EmotionBridge/          (Runtime module)
│   │   ├── Public/
│   │   │   ├── EmotionTakeTypes.h     NEW  — FEmotionTakeRecord, FEmotionAnalysisParams, …
│   │   │   ├── EmotionTakeStore.h     NEW  — FEmotionTakeStore static service
│   │   │   └── EmotionTimelineTypes.h MOD  — added ModelName, SampleRate fields
│   │   └── Private/
│   │       ├── EmotionTakeStore.cpp   NEW  — all disk I/O implementation
│   │       └── EmotionApiClient.cpp   MOD  — parses model_name and sample_rate
│   └── EmotionBridgeEditor/    (Editor module)
│       └── Private/
│           ├── SEmotionTakeLibrary.h  NEW  — Slate widget header
│           ├── SEmotionTakeLibrary.cpp NEW — full Take Library UI implementation
│           ├── SEmotionBridgePanel.h  MOD  — new state and method declarations
│           └── SEmotionBridgePanel.cpp MOD — Save Take + library sections
├── README.md                          MOD  — updated for Phase 2A
└── TAKE_LIBRARY.md                    NEW  — this file
```

Runtime: no new module dependencies required (uses existing `Json`, `Core` deps).
Editor: no new module dependencies required.
