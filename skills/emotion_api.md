# Skill: Emotion API Integration

## What it does
Sends WAV audio to the local Speech-Emotion-Recognition FastAPI service and parses the response. The Unreal plugin (`EmotionApiClient`) targets the UE-friendly variant of the timeline endpoint.

## Base URL
`http://localhost:8000` (configurable — `HOST` / `PORT` env vars in [scripts/run_api.sh](../scripts/run_api.sh); Unreal side reads from `UEmotionBridgeSettings` / project config).

## Endpoints

### `GET /health`
Response:
```json
{ "status": "ok", "model_id": "baseline", "device": "cpu" }
```

### `POST /predict` (single-clip)
Content-Type: `multipart/form-data`
Fields:
- `file` — WAV bytes (required)
- `include_scores` — bool (optional)

Response:
```json
{
  "emotion": "happy",
  "confidence": 0.91,
  "scores": { "neutral": 0.05, "happy": 0.91, "sad": 0.02, "angry": 0.02, "fear": 0.0, "disgust": 0.0, "surprise": 0.0 },
  "model_name": "speechbrain-iemocap",
  "duration_sec": 3.2
}
```

### `POST /timeline`
Full timeline with per-window data. Fields (all optional except `file`):
- `file` — WAV bytes
- `window_sec` (default 2.0), `hop_sec` (default 0.5)
- `pad_mode` — `"none" | "zero" | "reflect"`
- `smoothing_method` — `"none" | "majority" | "hysteresis" | "ema"`
- `hysteresis_min_run`, `majority_window`, `ema_alpha`
- `include_windows`, `include_scores`

### `POST /timeline/unreal` (used by the plugin)
Same inputs; simplified response shape intended for in-engine consumption:
```json
{
  "duration_sec": 12.34,
  "segments": [
    { "start_sec": 0.0,  "end_sec": 2.4, "emotion": "neutral", "confidence": 0.72 },
    { "start_sec": 2.4,  "end_sec": 5.1, "emotion": "happy",   "confidence": 0.88 },
    { "start_sec": 5.1,  "end_sec": 8.0, "emotion": "angry",   "confidence": 0.81 }
  ]
}
```
Unreal deserialises this into `FEmotionTimelineResponse` (see [EmotionTimelineTypes.h](../unreal/Plugins/EmotionBridge/Source/EmotionBridge/Public/EmotionTimelineTypes.h)).

## Canonical Emotion Labels
Defined in [src/model/labels.py](../src/model/labels.py):

| Label    | Supported by baseline (IEMOCAP) |
|----------|---------------------------------|
| neutral  | yes                             |
| happy    | yes                             |
| sad      | yes                             |
| angry    | yes                             |
| fear     | no — always 0.0                 |
| disgust  | no — always 0.0                 |
| surprise | no — always 0.0                 |

Raw model label mapping (`IEMOCAP_TO_CANONICAL`): `neu→neutral`, `hap→happy`, `sad→sad`, `ang→angry`.

## Curve / Morph Mapping (ARKit 52 naming, MetaHuman face)
Presets live in `UMetaHumanEmotionDriverComponent::MakeDefaultPresets()` — change them there or via the `ExpressionPresets` Details panel array. Presets deliberately avoid jaw/lip phoneme curves to stay compatible with Layer-1 speech animation.

| Emotion  | Primary curves (upper face)           | Secondary curves (mouth corners) | Notes                                   |
|----------|---------------------------------------|----------------------------------|-----------------------------------------|
| neutral  | — (empty; all driven targets → 0)     | —                                | Use an empty `MorphWeights` array       |
| happy    | `cheekSquint_L/R`, `eyeSquint_L/R`    | `mouthSmile_L/R`                 | Keep mouth weight ≤ 0.5 to avoid LS fight|
| sad      | `browDown_L/R`, `browInnerUp`         | `mouthFrown_L/R`                 | Low intensity reads best                |
| angry    | `browDown_L/R`, `noseSneer_L/R`       | (none)                           | Strong brow, minimal mouth              |

(Presets for `fear / disgust / surprise` are optional — currently unused because the baseline model never emits them.)

## Error Handling (plugin side)
- API unreachable / timeout → log `EmotionBridge` error, fall back to `neutral` with confidence 0.
- Non-200 response → surface API `detail` via the editor panel's status label; don't crash playback.
- Empty segments array → do nothing (no face drive).
- If `confidence < FEmotionOverlaySettings::ConfidenceThreshold` (conceptual; driver uses `bUseConfidenceAsWeight` to scale intensity), expression proportionally blends toward neutral.
- Unknown emotion label → treated as `neutral` by `UMetaHumanEmotionDriverComponent::ApplyEmotion`.
