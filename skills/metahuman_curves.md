# Skill: MetaHuman Facial Curve Reference

## Blendshape System
**ARKit 52-blendshape** naming for presets, mapped to **MetaHuman DNA raw controls** at runtime.

- Preset-facing names (what you author in `FEmotionExpressionPreset::MorphWeights`): ARKit style — e.g. `browDown_L`, `cheekSquint_R`, `mouthSmile_L`.
- Runtime curve names (what the MetaHuman DNA solver reads): `CTRL_expressions.<control><Side>` — e.g. `CTRL_expressions.browDownL`, `CTRL_expressions.cheekSquintR`.
- Translation from ARKit → CTRL is done inside [MetaHumanEmotionDriverComponent.cpp](../unreal/Plugins/EmotionBridge/Source/EmotionBridge/Private/MetaHumanEmotionDriverComponent.cpp). Animation curves are then pushed via `UMetaHumanEmotionAnimInstance::NativeUpdateAnimation → AddCurveValue` so the Post-Process ABP's RigLogic node sees them.

## Layer Model (important — emotion overlay must NOT fight speech)
| Layer | Owner | Drives |
|-------|-------|--------|
| 1 — Speech / lip sync | MetaHuman audio-driven AnimBP | Jaw, primary lip phoneme curves |
| 2 — Emotion overlay (this plugin) | `UMetaHumanEmotionDriverComponent` | Brow, cheeks, eye squint, nose wrinkle, mouth corners |
| 3 — Tooling | `SEmotionBridgePanel` (editor) | Binds actor, drives playback |

**Speech-safety rule** (from [EmotionMetaHumanTypes.h](../unreal/Plugins/EmotionBridge/Source/EmotionBridge/Public/EmotionMetaHumanTypes.h)): do NOT include jaw or primary lip phoneme targets in emotion presets (or keep weight < 0.1). Stick to upper-face + mouth corners only.

## Key Curves for Emotion (safe set)
Upper face:
- `browDown_L`, `browDown_R`, `browInnerUp`, `browOuterUp_L`, `browOuterUp_R`
- `cheekSquint_L`, `cheekSquint_R`, `cheekPuff`
- `eyeSquint_L`, `eyeSquint_R`, `eyeWide_L`, `eyeWide_R`, `eyeBlink_L`, `eyeBlink_R`
- `noseSneer_L`, `noseSneer_R`

Mouth corners (safe to drive at moderate weight):
- `mouthSmile_L`, `mouthSmile_R`
- `mouthFrown_L`, `mouthFrown_R`
- `mouthDimple_L`, `mouthDimple_R`

Avoid unless intentionally stylising (owned by Layer 1):
- `jawOpen`, `jawForward`, `jawLeft`, `jawRight`
- `mouthFunnel`, `mouthPucker`, `mouthClose`, `mouthRoll*`, `mouthShrug*`, `mouthPress_*`, `mouthStretch_*`

## Discovering Real Curve Names on Your MetaHuman
1. In the Unreal Editor, open `/Game/MetaHumans/Amelia/Face/Amelia_FaceMesh`.
2. Details panel → **Morph Target Preview** section lists every morph target on the mesh — those are the *exact* names to use in `FEmotionMorphWeight::MorphTargetName`.
3. DNA raw control names (`CTRL_expressions.*`) can be listed from the MetaHuman DNA asset or via the editor Python console:
   ```python
   import unreal
   mesh = unreal.load_asset('/Game/MetaHumans/Amelia/Face/Amelia_FaceMesh')
   # Inspect morph target names:
   for mt in mesh.morph_targets:
       unreal.log(mt.get_name())
   ```

## How to Drive Curves from C++
The driver component does this for you — but the underlying operations are:

```cpp
// Direct morph target write (works on any SkeletalMeshComponent):
USkeletalMeshComponent* FaceMesh = ...;
FaceMesh->SetMorphTarget(FName("mouthSmile_L"), Value, /*bRemoveZeroWeight=*/true);

// MetaHuman DNA path (required so RigLogic blends it correctly):
// 1) Store the target value:
EmotionAnimInstance->CurveValues.Add(FName("CTRL_expressions.browDownL"), Value);
// 2) NativeUpdateAnimation re-pushes it every frame:
void UMetaHumanEmotionAnimInstance::NativeUpdateAnimation(float Dt) {
    Super::NativeUpdateAnimation(Dt);
    for (const auto& Pair : CurveValues) {
        AddCurveValue(Pair.Key, Pair.Value);
    }
}
```

## Blend Strategy (implemented in UMetaHumanEmotionDriverComponent)
- Each `ApplyEmotion(Label, Confidence)` call sets `BlendState.FromEmotion = prev, ToEmotion = new, BlendAlpha = 0`.
- Every tick, `BlendAlpha += DeltaTime / BlendDurationSec` (default 0.4 s), clamped to 1.
- Output weight per morph target:
  `w = Lerp(FromPreset.FindWeight(name), ToPreset.FindWeight(name), Alpha) × BaseIntensity × Multiplier × (bUseConfidenceAsWeight ? Confidence : 1.0)`.
- Targets present in any preset but absent from the current preset blend to 0 — so expressions "un-wind" cleanly when switching.

## Thread Safety
All face-mesh writes happen on the game thread from `TickComponent`. Keep it that way — do not write morph targets or curves from async HTTP callbacks; marshal them back to the game thread (the API client already does this).
