# EmotionBridge Phase 2B — MetaHuman Face Integration

**Audio-Driven Speech + Emotion Timeline Overlay**

---

## Contents

1. [What Phase 2B Adds](#what-phase-2b-adds)
2. [Architecture: Layered Animation](#architecture-layered-animation)
3. [Quick-Start End-to-End Demo](#quick-start-end-to-end-demo)
4. [Step-by-Step: Setting Up a MetaHuman](#step-by-step-setting-up-a-metahuman)
5. [Importing Your Audio](#importing-your-audio)
6. [Connecting Audio-Driven Speech Animation (Manual Steps)](#connecting-audio-driven-speech-animation-manual-steps)
7. [How Emotion Presets Work](#how-emotion-presets-work)
8. [How Smooth Transitions Work](#how-smooth-transitions-work)
9. [What Is Automated vs. Manual](#what-is-automated-vs-manual)
10. [Morph Target Naming on MetaHuman](#morph-target-naming-on-metahuman)
11. [Using the Emotion Bridge Panel — METAHUMAN FACE Section](#using-the-emotion-bridge-panel--metahuman-face-section)
12. [Persisting Phase 2B State in Takes](#persisting-phase-2b-state-in-takes)
13. [Troubleshooting](#troubleshooting)
14. [Future Extension Points](#future-extension-points)
15. [Code Reference](#code-reference)

---

## What Phase 2B Adds

Phase 2B upgrades the EmotionBridge plugin from a "color-changing actor demo" into a
**MetaHuman face demo**.  After Phase 2B, you can:

| Capability | Phase 1 | Phase 2B |
|------------|---------|----------|
| Analyse audio emotion timeline | ✅ | ✅ |
| Drive a color-changing lamp | ✅ | ✅ |
| Import WAV as UE SoundWave | ✗ | ✅ (one click) |
| Bind a MetaHuman actor | ✗ | ✅ |
| Drive facial morph targets from emotion timeline | ✗ | ✅ |
| Smooth crossfade between emotions | ✗ | ✅ |
| Per-emotion intensity multipliers | ✗ | ✅ |
| Confidence-weighted expression intensity | ✗ | ✅ |
| Save MetaHuman state in Take Library | ✗ | ✅ |

**Phase 2B is fully offline, editor-first, and does not require real-time streaming or microphone capture.**

---

## Architecture: Layered Animation

Phase 2B uses a **two-layer architecture** so your MetaHuman speaks naturally
_and_ expresses emotion simultaneously.

```
┌─────────────────────────────────────────────────────────────────┐
│                     MetaHuman Face Result                        │
│         (what you see in the viewport during playback)           │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2 — Emotion Overlay                           (Phase 2B) │
│  • Driven by EmotionBridge API timeline                          │
│  • Applied via UMetaHumanEmotionDriverComponent                  │
│  • Targets: brow, cheeks, eye squint, nose wrinkle               │
│  • Smooth crossfade between segments (configurable duration)     │
│  • Confidence-weighted intensity                                 │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 1 — Base Speech Animation                    (MetaHuman) │
│  • Provided by MetaHuman's Animation Blueprint                   │
│  • Audio-driven: jaw, lips, phonemes from imported SoundWave     │
│  • Set up in MetaHuman Animator or Sequencer (manual steps)      │
└─────────────────────────────────────────────────────────────────┘
```

### Why not just override everything from the API?

MetaHuman's own animation system does lip-sync exceptionally well.
Rebuilding a phoneme solver in the plugin is unnecessary and fragile.

Instead, the EmotionBridge API provides **what MetaHuman can't infer from audio alone**:
- Explicit emotion labels with exact timing
- Confidence scores that let you scale expression strength
- Structured control over which emotion dominates each moment

By treating the API as an **overlay** on top of MetaHuman's base speech,
you preserve high-quality lip sync while adding emotional expressiveness.

---

## Quick-Start End-to-End Demo

This is the fastest path to seeing a result.

**Prerequisites:** Unreal 5.5+, MetaHuman Plugin enabled, EmotionBridge plugin compiled, Python backend running (`python api/server.py`).

```
1. Open your EmotionDemo project in Unreal Editor.
2. Open Window → Emotion Bridge to show the panel.
3. In AUDIO FILE, click Browse... and pick a WAV file.
4. Click Analyze and wait for results.
5. In METAHUMAN FACE:
   a. Place a MetaHuman BP actor in your level.
   b. Select it in the viewport.
   c. Click "Bind Selected Actor".
   d. Click "Import WAV as SoundWave" to import the audio into the content browser.
6. Click Play Demo.
   - The color swatch and lamp still react (Phase 1 behaviour preserved).
   - The MetaHuman face now shows emotion overlaid during playback.
7. Observe brow, cheeks, and eye changes correspond to segments in the timeline.
8. Click Stop Demo to reset.
```

> **Note on speech animation (Layer 1):**
> To also see lip-sync during Play Demo, follow the [manual MetaHuman audio setup](#connecting-audio-driven-speech-animation-manual-steps).
> The emotion overlay (Layer 2) works independently — you will see expression changes
> even before Layer 1 is configured.

---

## Step-by-Step: Setting Up a MetaHuman

### 1. Get a MetaHuman character

If you don't already have one:

1. Open **MetaHuman Creator** at [metahuman.unrealengine.com](https://metahuman.unrealengine.com).
2. Design your character.
3. Export it to your Unreal project via **Quixel Bridge** (Window → Quixel Bridge).
4. The MetaHuman Blueprint lands in `/Game/MetaHumans/<CharacterName>/`.

### 2. Place the MetaHuman in your level

1. In the Content Browser, navigate to `/Game/MetaHumans/<CharacterName>/`.
2. Drag the Blueprint (e.g. `BP_<CharacterName>`) into the viewport.
3. Position it somewhere visible.

### 3. Identify the Face mesh component

MetaHuman actors have multiple SkeletalMeshComponents:
- **Face** — the facial geometry, used for expression. This is what Phase 2B drives.
- Body, Legs, Arms, Torso — body parts, not used for facial expression.

The `UMetaHumanEmotionDriverComponent` auto-detects a component whose name contains
`"Face"` (case-insensitive).  This matches the standard MetaHuman naming convention.

To verify:
1. Select the MetaHuman actor in the level.
2. In **Details**, expand the Components list.
3. Look for a component named `Face` or `Face_Archetype_...`.

If your character uses a non-standard name, set **OverlaySettings → FaceMeshComponentName**
on the driver component after binding.

### 4. Enable the MetaHuman Plugin

> **Window → Plugins → search "MetaHuman" → enable → restart editor.**

If this plugin is not active, the MetaHuman Blueprint will not load correctly.

---

## Importing Your Audio

The Emotion Bridge panel handles audio in two different ways:

| Purpose | What it is | Where |
|---------|-----------|-------|
| **Analysis source** | Absolute filesystem WAV path | Audio File section |
| **Speech animation source** | UE SoundWave content asset | METAHUMAN FACE section |

These can be the same file.  The import step creates a copy as a `.uasset`.

### How to import

1. Select your WAV in the **AUDIO FILE** section.
2. In **METAHUMAN FACE**, click **Import WAV as SoundWave**.
3. The asset is created at `/Game/EmotionBridge/Audio/<filename>`.
4. The path is shown in the panel and saved with the take.

If the asset already exists at that path, it is reused without re-importing.

---

## Connecting Audio-Driven Speech Animation (Manual Steps)

> **This is Layer 1.  The emotion overlay (Layer 2) works without it.**
> Skip this section for a first demo if you just want to see emotion on the face.

MetaHuman's audio-driven facial animation requires the SoundWave to be connected
to the MetaHuman's Animation Blueprint or a Level Sequence.  This cannot be fully
automated from a plugin due to Unreal Editor's interactive asset creation requirements.

### Method A: Audio Component on the MetaHuman (simplest)

1. In the MetaHuman Blueprint, add an **Audio Component**.
2. Set its **Sound** property to your imported SoundWave asset.
3. In the AnimBP's event graph, drive the face rig from the audio component's output.
   MetaHuman ships with an `ABP_<Name>_FacialAnimation` that can accept audio input.

### Method B: Level Sequence with Audio and Face Animation Track

1. Open **Sequencer** (Window → Cinematics → Sequencer).
2. Create a new Level Sequence (`+` → Level Sequence).
3. Add the MetaHuman actor to the sequence.
4. Add an **Audio Track** and select your SoundWave.
5. Add a **Face Animation Track** (requires MetaHuman Performance asset).
6. Press Play in Sequencer to preview synchronized speech + emotion.

### Method C: MetaHuman Animator (most production-quality)

1. Export the WAV to **MetaHuman Animator** (standalone app from Epic).
2. Process the audio to generate a MetaHuman Performance asset.
3. Import the Performance into Unreal.
4. Apply it via the AnimBP or Sequencer.

> **For Phase 2B demo purposes:** Method A (Audio Component) is quickest.
> For production, use MetaHuman Animator for highest-quality lip sync.

---

## How Emotion Presets Work

Each canonical emotion is defined by an `FEmotionExpressionPreset`:
- A list of `(MorphTargetName, Weight)` pairs
- A `BaseIntensity` scalar (global dial for the whole preset)

### Built-in default presets (ARKit blendshape names)

These are applied automatically at first use if you don't override them:

| Emotion | Targets driven | Character |
|---------|----------------|-----------|
| **angry** | `browDown_L/R`, `eyeSquint_L/R`, `noseSneer_L/R` | Furrowed brow, squinting, nose wrinkle |
| **happy** | `cheekSquint_L/R`, `eyeSquint_L/R`, `mouthSmile_L/R`, `browInnerUp` | Raised cheeks, smile squint |
| **sad** | `browInnerUp`, `browDown_L/R`, `eyeWide_L/R`, `mouthFrown_L/R` | Inner brow arch, wide eyes, down-turned corners |
| **neutral** | _(empty)_ | All targets return to 0 |

### Speech safety

The presets intentionally **omit jaw and primary lip targets**.
MetaHuman's animation system owns those for phoneme accuracy.
The emotion presets focus on the upper face + very subtle mouth corners — regions
that carry emotion without conflicting with jaw/lip motion.

### Customising presets

**Method 1 — Details panel:**

1. Select the MetaHuman actor.
2. In **Details**, find `MetaHuman Emotion Driver`.
3. Expand **Expression Presets**.
4. Edit individual `MorphWeights` entries.

**Method 2 — Blueprint/C++:**

```cpp
TArray<FEmotionExpressionPreset> Presets = UMetaHumanEmotionDriverComponent::MakeDefaultPresets();

// Make angry stronger
for (auto& Preset : Presets)
{
    if (Preset.EmotionName == TEXT("angry"))
    {
        Preset.BaseIntensity = 1.3f;
        break;
    }
}

DriverComponent->SetPresets(Presets);
```

**Method 3 — Replace one preset, keep others:**

```cpp
FEmotionExpressionPreset HappyPreset;
HappyPreset.EmotionName   = TEXT("happy");
HappyPreset.BaseIntensity = 1.1f;

FEmotionMorphWeight W;
W.MorphTargetName = TEXT("cheekSquint_L"); W.Weight = 0.7f;
HappyPreset.MorphWeights.Add(W);
W.MorphTargetName = TEXT("cheekSquint_R"); W.Weight = 0.7f;
HappyPreset.MorphWeights.Add(W);
// ... more targets

DriverComponent->SetPresetForEmotion(HappyPreset);
```

---

## How Smooth Transitions Work

When the active emotion segment changes, the driver does NOT snap to the new expression.
Instead it **crossfades** from the current blend state to the new target:

```
Blend alpha timeline:

  0.0 ──── start of crossfade (old emotion fully present)
  0.5 ──── mid-blend (50% old, 50% new)
  1.0 ──── settled (new emotion fully present)

  Blend duration: configurable via panel (default 0.4 s)
  Time to reach 1.0: BlendDurationSec seconds after emotion change
```

Each morph target is linearly interpolated:

```
FinalWeight = Lerp(FromPresetWeight, ToPresetWeight, BlendAlpha)
            × EffectiveIntensity
```

Where `EffectiveIntensity` combines:
- API `confidence` (if "Use confidence as weight" is enabled)
- Per-emotion intensity multiplier from the panel
- `BaseIntensity` from the preset definition

### Tuning tips

| Issue | Fix |
|-------|-----|
| Transitions look abrupt | Increase blend duration (0.5–0.8 s) |
| Expression fades too slowly | Decrease blend duration (0.2–0.3 s) |
| Expression too subtle | Increase intensity multiplier for that emotion |
| Expression too strong | Decrease `BaseIntensity` on the preset |
| Low-confidence segments look noisy | Enable "Use confidence as weight" |
| Neutral segments show residual expression | Check blend duration isn't too long |

---

## What Is Automated vs. Manual

### Fully automated by Phase 2B:

- Parsing the emotion timeline from the API
- Binding a MetaHuman actor to the driver component via one button click
- Adding `UMetaHumanEmotionDriverComponent` to the actor automatically
- Resolving the face SkeletalMeshComponent by name
- Driving morph target weights with smooth blending during Play Demo
- Importing the WAV as a SoundWave content asset (one click)
- Saving SoundWave path, actor label, and overlay settings in Take records
- Restoring overlay settings when a Take is loaded

### Requires manual setup (documented above):

- Placing a MetaHuman actor in the level
- Connecting the SoundWave to the MetaHuman's animation system (Layer 1 speech)
- Tuning expression presets for your specific MetaHuman's morph target names
- Setting `FaceMeshComponentName` if auto-detection picks the wrong mesh

### Requires a future phase:

- Exporting to a Level Sequence for cinematic rendering
- Full body acting / posture changes
- Realtime streaming / LiveLink emotion overlay
- Production-grade preset authoring via DataAsset

---

## Morph Target Naming on MetaHuman

Morph target names vary depending on how the MetaHuman was exported and which
version of the plugin is installed.  If the built-in presets produce no visible
change, your MetaHuman may use different names.

### How to find your MetaHuman's morph target names

1. Open your level.
2. Select the MetaHuman actor.
3. In **Details**, find the `Face` SkeletalMeshComponent.
4. Scroll to **Morph Target Preview** (at the bottom of the Details panel).
5. Drag sliders to find which targets move which regions.
6. Note the exact names (they are case-sensitive).

### Common naming conventions

| Convention | Example names | Used by |
|------------|--------------|---------|
| ARKit camelCase | `browDown_L`, `cheekSquint_R` | MetaHuman (ARKit export) |
| CTRL prefix | `CTRL_R_brow_down`, `CTRL_L_cheek_raise` | MetaHuman Control Rig curves |
| DNA-style | `browInnerUp`, `jawOpen` | Some MetaHuman variants |

The built-in presets use **ARKit camelCase** names.  If your MetaHuman uses CTRL_
prefix names instead, update the presets via `SetPresets()` with corrected names.

---

## Using the Emotion Bridge Panel — METAHUMAN FACE Section

The **METAHUMAN FACE** section appears at the bottom of the Emotion Bridge panel
(scroll down, after the Take Library).

### TARGET subsection

| Control | Purpose |
|---------|---------|
| **Bind Selected Actor** | Reads the viewport selection, validates it has a SkeletalMeshComponent, and adds `UMetaHumanEmotionDriverComponent` (transient). |
| **Clear** | Unlinks the actor and resets its face to neutral. |
| Bound actor display | Shows the label of the currently bound actor. |
| Face mesh display | Shows whether the face mesh was auto-detected. |

### AUDIO ASSET subsection

| Control | Purpose |
|---------|---------|
| **Import WAV as SoundWave** | Imports the currently selected WAV file into `/Game/EmotionBridge/Audio/` as a SoundWave asset. Reuses existing asset if already imported. |
| SoundWave path display | Shows the content-browser path of the imported asset. |

### EMOTION OVERLAY subsection

| Control | Purpose |
|---------|---------|
| Enable overlay checkbox | Master toggle — bypass all morph target changes. |
| Blend duration (s) | How long crossfades between emotion segments take. |
| Use confidence weighting | Scale expression by the API confidence value. |

### EMOTION INTENSITY MULTIPLIERS subsection

Four float inputs (angry / happy / sad / neutral).  Range 0–2.
- `1.0` = use preset values as-is (default).
- `0.0` = suppress this emotion entirely.
- `1.5` = amplify by 50%.

### CURRENT STATE subsection

During playback, shows the emotion currently being applied and the blend completion
percentage.  Updates every rendered frame.

---

## Persisting Phase 2B State in Takes

When you click **Save Take** (Phase 2A Take Library), Phase 2B state is also persisted:

```json
"phase2b": {
    "sound_wave_asset": "/Game/EmotionBridge/Audio/MySpeech",
    "bound_actor_label": "BP_Mia_01",
    "overlay_blend_duration_sec": 0.4,
    "overlay_enabled": true,
    ...
}
```

When you **Load** a Take:
- SoundWave path is restored to the panel.
- Blend duration and overlay-enabled toggle are restored.
- A note is shown if the take had a bound actor (you need to rebind manually — actor
  object references are not persisted across editor sessions).

---

## Troubleshooting

### MetaHuman Plugin not enabled

**Symptom:** The MetaHuman BP won't load; empty viewport after dragging in.

**Fix:** `Edit → Plugins → search "MetaHuman" → enable → restart editor`.

---

### No actor selected / wrong actor bound

**Symptom:** "No actor selected" when clicking Bind.

**Fix:**
1. Click the MetaHuman actor in the viewport.
2. Confirm selection (it should highlight blue).
3. Then click Bind Selected Actor.

---

### No facial motion visible during playback

Most likely cause: the morph target names in the built-in presets don't match
your MetaHuman's morph targets.

**Diagnosis:**
1. Open Output Log (`Window → Output Log`).
2. Filter by `LogEmotionBridge`.
3. Look for: `MetaHumanEmotionDriver: applied built-in ARKit expression presets`.
4. During Play Demo, `SetMorphTarget` calls happen each tick but produce no visible
   change if the names don't exist on the mesh.

**Fix:**
1. Find your MetaHuman's actual morph target names (Details → Morph Target Preview).
2. Update the presets:
   ```cpp
   auto Presets = UMetaHumanEmotionDriverComponent::MakeDefaultPresets();
   // Find the "angry" preset and change its target names to match yours.
   // Then call DriverComponent->SetPresets(Presets);
   ```

---

### Audio imported but speech animation not showing

**Symptom:** Emotion overlay works (face changes expression) but lips don't move.

**Cause:** Layer 1 (MetaHuman speech animation) is not set up yet.
Layer 2 (emotion overlay) works independently.

**Fix:** Follow [Connecting Audio-Driven Speech Animation](#connecting-audio-driven-speech-animation-manual-steps).

---

### Expressions too strong / destroying phoneme motion

**Symptom:** Mouth looks distorted during speech.

**Cause:** The presets include mouth-related targets that conflict with the AnimBP.

**Fix:**
1. Open the `"angry"` / `"happy"` / `"sad"` preset.
2. Remove `mouthSmile_*`, `mouthFrown_*` entries.
3. Focus on brow and cheek targets only.

Alternatively, reduce the emotion intensity multiplier for the offending emotion to
`0.6–0.8` to reduce the magnitude of all targets.

---

### Transitions look abrupt (hard cut between emotions)

**Symptom:** Expression jumps instantly between segments.

**Cause:** Blend duration is 0 or very low.

**Fix:** Set **Blend duration** to `0.4–0.6 s` in the EMOTION OVERLAY section.

---

### Overlay fights the mouth / jaw floats

**Symptom:** Jaw position looks wrong during speech after binding.

**Cause:** The default presets do not touch jaw targets.  If you see jaw issues,
your MetaHuman's animation system may be driving jaw via a morph target that
the plugin is inadvertently zeroing.

**Fix:**
1. Check which targets are in `AllDrivenMorphTargets` (add a log call or debugger).
2. Remove any jaw-related targets (`jawOpen`, `jawLeft`, `jawRight`, etc.) from
   all expression presets.

---

### Missing preset data / expression inactive

**Symptom:** No expression visible despite correct morph target names.

**Checks:**
1. Is **Enable overlay** checked?
2. Is the bound actor valid (not been deleted or re-created)?
3. Are the intensity multipliers all set to `0.0`?
4. Is the timeline analysis valid (run Analyze first)?
5. Check `LogEmotionBridge` for any driver errors.

---

### Wrong actor bound after loading a Take

**Cause:** Actor references are not persisted.  The Take only stores the actor's
**label**, which is informational.

**Fix:** After loading the Take, select the correct MetaHuman actor and click
**Bind Selected Actor** again.

---

### Animation preview not updating in editor viewport

**Symptom:** Face stays still even though Output Log shows `ApplyEmotion` calls.

**Cause:** The SkeletalMeshComponent may not be refreshing its render state.

**Fix:**
1. Confirm `bTickInEditor = true` is set on the driver component (it is by default).
2. Check that `RegisterComponent()` was called after `AddInstanceComponent()` (the
   panel does this automatically on Bind).
3. If still unresponsive, try entering and immediately exiting PIE once to
   re-register components, then return to editor play.

---

## Future Extension Points

### Better preset authoring

Currently presets live in code/Details panel.  A future DataAsset-based preset system
would allow artists to author, save, and share expression libraries without code changes.
`FEmotionTakePhase2B::EmotionPresetMappingPath` is a placeholder for this asset path.

### Take Library integration

Phase 2B already saves SoundWave path, actor label, blend duration, and overlay-enabled
state in every Take record.  When multiple takes are compared in the Library, the Phase 2B
fields will indicate which takes have MetaHuman bindings ready for replay.

### Cleaned audio vs. playback audio

`FEmotionTakePhase2B::CleanedAudioPath` is reserved for a denoising pipeline.
The analysis WAV and the MetaHuman audio-driven animation WAV could eventually differ:
the analysis would use the cleaned version for better emotion accuracy, while the
speaker would play the original for natural sound quality.

### Sequencer / Level Sequence export

A future "Export to Sequence" button would bake the emotion overlay into a
Level Sequence as animation curves, making it renderable and shareable without
the plugin running.  `FEmotionTakePhase2B::LevelSequencePath` reserves this field.

### MetaHuman Performance asset generation

`FEmotionTakePhase2B::MetaHumanPerformancePath` anticipates future integration with
MetaHuman Animator or a similar solve pipeline that would auto-generate a
MetaHuman Performance asset from the audio, eliminating manual setup of Layer 1.

### Body acting

Once MetaHuman Animator's body-pose API matures, emotion labels could drive
basic body posture presets (slouch for sad, upright energy for happy) as Layer 3.

### Realtime / LiveLink streaming

The `UMetaHumanEmotionDriverComponent`'s `ApplyEmotion()` API is intentionally
simple — it accepts an emotion label and confidence, and handles all blending
internally.  A LiveLink-based source that calls `ApplyEmotion()` at video frame rate
would upgrade the system to realtime without changing any downstream code.

---

## Code Reference

| Symbol | Location | Purpose |
|--------|----------|---------|
| `FEmotionMorphWeight` | `EmotionMetaHumanTypes.h` | Single morph target + weight pair |
| `FEmotionExpressionPreset` | `EmotionMetaHumanTypes.h` | One emotion's full face shape |
| `FEmotionOverlaySettings` | `EmotionMetaHumanTypes.h` | Blend/intensity/mesh config |
| `FEmotionBlendState` | `EmotionMetaHumanTypes.h` | Transient blend-in-progress state |
| `UMetaHumanEmotionDriverComponent` | `MetaHumanEmotionDriverComponent.h/.cpp` | Main runtime component |
| `MakeDefaultPresets()` | `MetaHumanEmotionDriverComponent.cpp` | Built-in ARKit preset factory |
| `FEmotionAudioAssetHelper` | `EmotionAudioAssetHelper.h/.cpp` | WAV → SoundWave import (editor only) |
| `SEmotionBridgePanel::BuildMetaHumanSection()` | `SEmotionBridgePanel.cpp` | Phase 2B UI |
| `SEmotionBridgePanel::OnBindSelectedActor()` | `SEmotionBridgePanel.cpp` | Actor binding |
| `SEmotionBridgePanel::OnImportSoundWave()` | `SEmotionBridgePanel.cpp` | SoundWave import |
| `FEmotionTakePhase2B` | `EmotionTakeTypes.h` | Per-take Phase 2B persistence fields |

---

*EmotionBridge Phase 2B — generated 2026-03-30*
