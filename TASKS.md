# TASKS.md

## Active
- [ ] Finish MetaHuman integration branch (`feat/metahuman-integration`): commit pending edits to `MetaHumanEmotionDriverComponent.{cpp,h}`, new `MetaHumanEmotionAnimInstance.{cpp,h}`, `DefaultEngine.ini`, Amelia assets, and `Content/Python/SetupAmeliaEmotionDriver.py`.
- [ ] Verify Amelia face drives end-to-end on the current branch: API running → Analyze → Play Demo shows blended brow/cheeks/mouth-corner motion on top of MetaHuman lip sync.
- [ ] Confirm `UMetaHumanEmotionAnimInstance` is installed automatically by the driver (no manual Blueprint edits required).

## Backlog
- [ ] Emotion smoothing / temporal blending tuning (crossfade length, confidence ramping independent of label switch).
- [ ] Confidence threshold behaviour — explicit hard cutoff vs. current proportional-intensity model.
- [ ] Expand model support beyond IEMOCAP's 4 labels (fear/disgust/surprise presets currently unused).
- [ ] Multi-speaker support (diarisation → per-speaker emotion tracks).
- [ ] Richer Blueprint-exposed API for designers (expose preset editing / live preview from BP).
- [ ] Editor preview tool: scrub a timeline asset and preview emotion poses without audio playback.
- [ ] Graceful API-down fallback (auto-retry, status banner in the panel).
- [ ] Unit-test coverage for `EmotionApiClient` JSON parsing against the `/timeline/unreal` schema.

## Completed
- [x] Python FastAPI backend with `/health`, `/predict`, `/timeline`, `/timeline/unreal` endpoints.
- [x] SpeechBrain IEMOCAP baseline model + canonical label mapping (`src/model/labels.py`).
- [x] Windowing / smoothing / merging pipeline (`src/timeline/`).
- [x] `EmotionBridge` runtime module: HTTP client, timeline types, playback component, emotion lamp demo actor.
- [x] `EmotionBridgeEditor` module: Slate tab, WAV picker, Take Library UI.
- [x] Persistent Take Library (`EmotionTakeStore`).
- [x] Create "Amelia" MetaHuman character for emotion display.
- [x] `UMetaHumanEmotionDriverComponent` — emotion overlay layer with blended morph-target driving (Phase 2B).
- [x] ARKit-named default expression presets targeting upper face + mouth corners only (speech-safe).
- [x] Custom `UMetaHumanEmotionAnimInstance` injecting `CTRL_expressions.*` curves into the Post-Process ABP.
- [x] Editor Python helper `SetupAmeliaEmotionDriver.py` to restore the Post-Process ABP after MetaHuman reimports.
- [x] `docs/UNREAL_INTEGRATION.md`, `docs/METAHUMAN_PHASE2B.md` integration guides.
- [x] M5 — editor-time Level Sequence playback: "Play" button on the panel runs the bound sequence via a transient `ULevelSequencePlayer` without opening the Sequencer UI.
- [x] Panel polish — each section (Backend, Audio File, Parameters, Results, Playback, Save Take, Take Library, MetaHuman, Sequencer Export) is now a collapsible `SExpandableArea`; inner duplicate headers removed.
