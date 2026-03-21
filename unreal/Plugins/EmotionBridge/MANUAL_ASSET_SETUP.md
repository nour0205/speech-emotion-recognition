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
   - `MeshComponent` should already reference `/Engine/BasicShapes/Cube`.
   - `LightIntensityBase` defaults to `3000`. Increase if the light is too dim.
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

## Optional: Color-Responding Mesh Material

By default only the **point light** color changes (visible on surrounding geometry).
To also tint the **cube mesh** itself:

### Create a Simple Color Material

1. In the **Content Browser**, right-click → **Material** → name it `M_EmotionColor`.
2. Double-click to open the Material Editor.
3. In the graph, hold **V** and click to create a **Vector Parameter** node.
   - Name the parameter exactly `BaseColor`.
   - Connect its output to the **Base Color** input of the Material.
4. Set **Shading Model** to `Unlit` (simpler, more vivid).
5. Press **Apply** and **Save**.

### Assign the Material

1. Select the `EmotionLampActor` in the viewport.
2. In **Details > Mesh Component > Materials**, slot 0: assign `M_EmotionColor`.
3. Save the level.

Now `AEmotionLampActor::ApplyEmotion()` will call
`DynamicMaterial->SetVectorParameterValue("BaseColor", Color)` automatically —
no Blueprint or extra code needed.

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
