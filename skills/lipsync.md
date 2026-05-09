# Skill: Lip Sync Integration

## Method in Use
**MetaHuman audio-driven face AnimBP (Layer 1)** — the stock MetaHuman face rig has a built-in Post-Process Animation Blueprint (`ABP_Amelia_FaceMesh_PostProcess`) whose RigLogic / DNA solver consumes audio and produces phoneme-driven jaw + lip curves. This plugin does **not** implement its own viseme mapping, OVR Lip Sync, or MetaSound lip driver. Emotion is an *overlay only*.

## How It Works (end-to-end)
1. In the editor panel or at runtime, an audio asset is played on the MetaHuman actor (typically via an `UAudioComponent` or the `SoundWave` referenced by the Take Library).
2. MetaHuman's Post-Process ABP, attached to `Amelia_FaceMesh`, reads the currently playing audio buffer and drives the speech-related `CTRL_expressions.*` curves (jaw, mouth shape).
3. **In parallel**, `UMetaHumanEmotionDriverComponent` calls `ApplyEmotion(label, confidence)` with the currently-active segment from `/timeline/unreal`.
4. The driver writes emotion curves (brow, cheeks, eye squint, nose wrinkle, mouth corners) into `UMetaHumanEmotionAnimInstance::CurveValues`.
5. `NativeUpdateAnimation` pushes both sets of curves into the animation pose. The Post-Process ABP's RigLogic node evaluates the combined pose — so emotion and lip sync merge in the DNA solver, not in the game-thread component.

## Viseme-to-Curve Mapping
Not maintained here — it's internal to the MetaHuman Post-Process ABP. Do not duplicate viseme logic in the plugin.

## Coexistence Rules (critical)
- Emotion presets **must not** drive jaw / primary lip phoneme curves (`jawOpen`, `mouthFunnel`, `mouthPucker`, `mouthClose`, `mouthRoll*`, `mouthPress_*`, etc.) at meaningful weight. Let Layer 1 own them.
- Mouth-corner curves (`mouthSmile_L/R`, `mouthFrown_L/R`, `mouthDimple_L/R`) can be driven at moderate weight (≤ 0.6) — they influence expression without masking phonemes.
- If a future requirement demands additive speech control (e.g. shouting vs. whispering), introduce it as a **separate layer** rather than folding it into `FEmotionExpressionPreset`.

## Threading
- `UMetaHumanEmotionDriverComponent::TickComponent` runs on the **game thread**.
- `EmotionApiClient` HTTP requests run async; the response callback marshals segments back to the game thread before touching the driver.
- `UMetaHumanEmotionAnimInstance::NativeUpdateAnimation` runs on the **animation thread** — but only reads the `CurveValues` map (written on the game thread). The map is a plain `TMap` with no explicit lock; UE's anim graph evaluation ordering makes this safe in practice because the driver writes before the anim instance reads in the same frame. If you ever see flicker, consider double-buffering.

## Future Work
- Optional OVR LipSync fallback for non-MetaHuman characters (out of scope for current milestone).
- Finer confidence smoothing between segments (already partly handled by the API's `smoothing_method`, but emotion intensity could ramp separately from label blend).
