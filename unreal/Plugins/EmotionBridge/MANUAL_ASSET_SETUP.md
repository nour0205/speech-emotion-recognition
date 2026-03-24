# Manual Asset Setup — EmotionBridge

Binary Unreal assets (`.uasset`, `.umap`) cannot be generated as text files and are not
committed to this repository. This document describes every step a student developer needs
to perform inside the Unreal Editor to complete the first working demo.

---

## Why These Assets Are Manual

Unreal Engine stores levels, meshes, and materials in a proprietary binary format. There is
no reliable way to produce valid binary `.uasset`/`.umap` files programmatically or from a
text file. Attempting to do so results in a corrupted project.

All logic, colors, HTTP communication, and playback behavior live in the C++ plugin — zero
Blueprint code is required for the demo to work. The only things you need to do manually are:

1. Create (or open) a level.
2. Optionally place an `AEmotionLampActor` by hand (the plugin will auto-spawn one if none exists).
3. Optionally assign a nicer material so the mesh also changes color.

---

## Prerequisites

- Unreal Engine 5.4+ installed via Epic Games Launcher.
- Project compiles without errors (run `Generate Xcode project` + build in Xcode).
- Python backend is running (`docker compose up api` or `make run`).

---

## Quick Start — See It in the Viewport in 2 Minutes

No level setup required for the panel color swatch. To also see it **in the 3D viewport**:

1. Open any level (File → New Level → Basic).
2. Open **Window → Emotion Bridge**, click **Analyze** on a WAV file.
3. Click **Focus / Spawn in Viewport** — this spawns an `EmotionLampActor` and
   moves the camera to it automatically.
4. Click **Play Demo** — the sphere and its point light change color per segment.

To use **your own character or actor** instead of the lamp, see the
"Link to Any Actor" section below.

---

## Minimum Visual Setup (Required for Demo)

### Step 1 — Open or Create a Level

1. In the **Content Browser**, right-click an empty area.
2. Choose **New Level** → **Empty Level**.
   (Or open the existing `EmotionDemoMap` if already created.)
3. Save it as `Content/Maps/EmotionDemoMap`.
4. In `unreal/Config/DefaultEngine.ini`, ensure the line reads:
   ```ini
   GameDefaultMap=/Game/Maps/EmotionDemoMap
   ```

### Step 2 — Place an AEmotionLampActor (Optional but Recommended)

> **Note:** The Emotion Bridge panel auto-spawns a lamp actor if none is found. Skip this
> step if you want to start quickly; it will appear at the world origin.

1. In the main Unreal Editor toolbar, click **Window > Place Actors**.
2. In the search box type `EmotionLamp`.
3. Drag an **EmotionLampActor** into the viewport and position it somewhere visible.
4. In the Details panel:
   - `MeshComponent` references `/Engine/BasicShapes/Sphere` — the glowing bulb.
   - The **PointLightComponent** sits at the sphere center; its `SourceRadius = 20 cm` makes
     it render as a visible physical light sphere in Lumen / ray-traced mode.
   - `LightIntensityBase` defaults to `5000`. Reduce if the light is too bright.
5. Press **Ctrl+S** to save the level.

### Step 3 — Open the Emotion Bridge Panel

1. In the main editor menu bar: **Window > Emotion Bridge**.
2. The custom tab docks into the editor. Drag it anywhere convenient.

### Step 4 — Run the Demo

1. Click **Health Check** — status should turn green.
2. Click **Browse** and select a `.wav` file
   (try `tests/fixtures/` or any speech WAV from your system).
3. Click **Analyze** and wait for the response.
4. The segment list populates with start time, end time, emotion, and confidence.
5. Click **Play Demo** — the lamp actor's point light changes color as each segment plays.
6. Click **Stop Demo** to halt and reset.

---

## Link to Any Actor

You are not limited to the `EmotionLampActor`. Any actor in the level — a character,
a MetaHuman, a door, a prop — can respond to emotions by adding one component.

### How It Works

The plugin defines `UEmotionColorComponent`, a standard Unreal **Actor Component**.
When the Emotion Bridge panel broadcasts an emotion (Play Demo tick), it searches
every actor in the editor world for this component and calls `ApplyEmotion()` on each
one. No Blueprint, no C++, no extra wiring — just add the component.

The component:

- Auto-detects a `SkeletalMeshComponent` (characters, MetaHumans) or a
  `StaticMeshComponent` (props, shapes) on the same actor.
- Creates a `UMaterialInstanceDynamic` from whichever material is in slot 0.
- Sets `BaseColor`, `Color`, `EmissiveColor`, and `Emissive` parameter values so it
  works with built-in materials, custom materials, and emissive materials alike.
- Optionally tints any `PointLightComponent` on the same actor (enabled by default).

### Step-by-Step: Add to a Static Mesh Actor

1. Place any mesh in the level (e.g., drag a **Cube** or **Sphere** from Place Actors,
   or use an existing prop in your scene).
2. Click the actor in the **viewport** or **Outliner** to select it.
3. In the **Details** panel (right side), scroll to the bottom and click **+ Add**.
4. In the search box, type `Emotion Color`.
5. Click **Emotion Color Component** to add it.
6. The component appears under the actor's component list. Its default settings work
   immediately — no further configuration is needed.
7. Press **Ctrl+S** to save the level.
8. In the Emotion Bridge panel, click **Analyze** then **Play Demo**.
   The mesh will change color as each emotion segment plays.

### Step-by-Step: Add to a Character or MetaHuman

The steps are identical. The component automatically detects the `SkeletalMeshComponent`:

1. Select your character or MetaHuman in the viewport or Outliner.
2. In **Details**, click **+ Add** → search `Emotion Color` → click it.
3. (Optional) In the component's Details, change **Material Slot** if your
   character's skin/face material is not in slot 0.
4. Press **Ctrl+S**.
5. Click **Play Demo** — the character's material tints to the current emotion color.

> **Beginner tip:** With a MetaHuman, the face is usually slot 0 on the `Face` skeletal
> mesh. If the color doesn't show, try selecting the MetaHuman's child `Face` mesh actor
> and add the component there instead of the root actor.

### Tuning the Component

Select the actor, click **Emotion Color Component** in the component list, and adjust
these properties in Details:

| Property | Default | Effect |
| -------- | ------- | ------ |
| `Target Mesh` | (auto) | Pin to a specific mesh if auto-detect picks the wrong one |
| `Material Slot` | `0` | Which material slot receives the color |
| `HDR Multiplier` | `4.0` | Values above 1 make emissive materials bloom; lower for subtlety |
| `Tint Point Light` | `true` | Also changes any `PointLightComponent` on the same actor |

### Using Multiple Actors at Once

You can add `Emotion Color Component` to **as many actors as you like**. All of them
will update simultaneously when Play Demo runs. Useful for:

- A character + the lamp actor together (character body tints, lamp lights the room).
- Multiple props in a scene (every object reacts to the speaker's emotion).
- A background skylight actor (though UE5 `SkyLight` is not a `PointLight` — add a
  `PointLightComponent` via Blueprint for that effect).

### Verify It Is Working

After adding the component, you can quickly verify without running Play Demo:

1. In the Emotion Bridge panel, click **Analyze** on a WAV file.
2. Click a row in the **segment list** — the panel color swatch updates to that emotion.
3. The actor in the viewport should immediately change color (no Play Demo needed).

If the color does not change, check the **Output Log** (Window → Output Log) for lines
starting with `LogEmotionBridge` — they will tell you whether a DynamicMaterial was
created and which parameter was set.

---

## Optional: Emissive Material for Maximum Glow

By default the point light illuminates the surrounding geometry and the sphere itself
receives light from within (via `SourceRadius`). For the richest look — a sphere that
visibly radiates colored light from its surface — assign an **emissive material**.

### Why an emissive material?

The engine's default sphere material does not have an `EmissiveColor` parameter.
`ApplyEmotion()` already tries `BaseColor`, `Color`, `EmissiveColor`, and `Emissive`
parameter names, so any material with one of those vector parameters will respond
automatically.

### Create an Emissive Color Material

1. In the **Content Browser**, right-click an empty area → **Material** → name it
   `M_EmotionEmissive`.
2. Double-click to open the Material Editor.
3. Hold **V** and click in the graph to create a **Vector Parameter** node.
   - Name it exactly `EmissiveColor`.
   - Connect its RGB output to the **Emissive Color** pin of the Material output node.
4. Set **Shading Model** to `Unlit` (no diffuse lighting — the glow is self-contained).
5. Press **Apply** and **Save**.

### Assign the Material

1. Select the `EmotionLampActor` in the viewport.
2. In **Details > Mesh Component > Materials**, slot 0: click the dropdown and select
   `M_EmotionEmissive`.
3. Save the level (Ctrl+S).

`ApplyEmotion()` will now set `EmissiveColor` to an HDR value
(`Color × MeshColorHDRMultiplier`, default ×4), which triggers UE5 Bloom on the
sphere surface — no Blueprint or extra code required.

> **Tip:** With UE5 Lumen + Bloom enabled (the default in a new project), the sphere
> will appear to radiate colored light even without the point light. Keep the point light
> enabled as well so the surrounding floor also changes color.

---

## Optional: Prettier Setup

If you want the demo to look polished:

1. **Sky atmosphere + directional light** — drag in a `SkyAtmosphere` and `DirectionalLight`
   from the Place Actors panel. This gives the room-like context that makes the lamp pop.

2. **Floor plane** — drag in a `Plane` (scale 10×10) beneath the lamp so the colored light
   casts a visible circle on the floor.

3. **Camera** — place a `CineCameraActor` looking at the lamp; press `G` to hide editor
   gizmos, then `Ctrl+Shift+P` to pilot the camera for a clean screenshot.

4. **Multiple lamp actors** — place 3–4 lamp actors and re-use the same panel; the panel
   targets the first one it finds via `TActorIterator`. To target a specific one you can
   rename it in the Details panel to `EmotionLamp_Primary` and adjust
   `FindOrSpawnLampActor` to search by label.

---

## Troubleshooting Asset Setup

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| EmotionLampActor not in Place Actors | Plugin not compiled yet | Build the project in Xcode first |
| Lamp mesh shows as a question mark | BasicShapes plugin disabled | Enable it in Edit > Plugins > "Basic Shapes" |
| Light color does not change | No level open in editor | Create and open a level |
| Mesh color does not change | Material missing `BaseColor` parameter | Follow the material setup steps above |
| Panel tab is missing from Window menu | EditorModule load failed | Check Output Log for compile errors |
