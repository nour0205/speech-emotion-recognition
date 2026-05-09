// Copyright (c) EmotionDemo Project. All rights reserved.
// M1 — Bakes emotion control curves into a UAnimSequence so they ride
// alongside the lip-sync curves produced by MetaHuman Performance.
#pragma once

#include "CoreMinimal.h"

class UAnimSequence;
struct FEmotionTimelineResponse;
struct FEmotionExpressionPreset;

/**
 * Static helper that writes per-control float curves into an AnimSequence
 * based on an emotion timeline.
 *
 * Why this exists:
 *   MetaHuman Performance (the "Process and Export to Anim Sequence" path
 *   you used in Phase 1) produces an AnimSequence whose curves drive the
 *   face's lip sync via RigLogic.  We want to overlay emotional expression
 *   on the same animation — same delivery mechanism, no runtime curve
 *   injection nonsense.  This baker reads the API timeline, looks up each
 *   emotion's preset, and writes the matching control values as curve
 *   keyframes on the existing AnimSequence asset.
 *
 * Curve naming:
 *   Curves are added with names that match the rig's CTRL_<L|R>_<feature>
 *   convention (e.g. "CTRL_R_brow_down").  The PostProcess ABP's RigLogic
 *   solver evaluates them automatically.
 *
 * Keyframe strategy:
 *   For each unique control across all presets, walks the timeline and
 *   places linear-interp keyframes at each segment boundary with a
 *   BlendDurationSec ramp on either side.  Result: smooth crossfades
 *   between emotions, no popping.
 */
class FEmotionAnimSequenceBaker
{
public:
	/**
	 * Modifies AnimSeq in place: adds float curves for every control name
	 * referenced by Presets, with keyframes derived from Timeline.
	 *
	 * Existing curves with the same names are removed and rewritten — safe
	 * to re-bake the same AnimSequence multiple times with different
	 * timelines.
	 *
	 * @param AnimSeq             Target AnimSequence (typically the one
	 *                            produced by MetaHuman Performance).
	 *                            Must not be null.  Will be marked dirty.
	 * @param Timeline            Parsed /timeline/unreal response.
	 * @param Presets             Emotion-name → control-weight mapping
	 *                            (typically from
	 *                            UMetaHumanEmotionDriverComponent::MakeDefaultPresets()).
	 * @param BlendDurationSec    Crossfade duration at segment boundaries.
	 *                            Default 0.4 — the same value the runtime
	 *                            driver uses.
	 * @param bUseConfidenceAsWeight  When true, multiplies each segment's
	 *                            preset values by its confidence in [0,1].
	 *
	 * @return true if at least one curve was written, false on bad inputs.
	 */
	static bool BakeEmotionCurves(
		UAnimSequence* AnimSeq,
		const FEmotionTimelineResponse& Timeline,
		const TArray<FEmotionExpressionPreset>& Presets,
		float BlendDurationSec = 0.4f,
		bool bUseConfidenceAsWeight = true);

private:
	/**
	 * Gathers every distinct morph/control name that appears in any of the
	 * given presets, in stable iteration order.  Used as the set of curves
	 * the baker emits on the AnimSequence.
	 */
	static void CollectAllControlNames(
		const TArray<FEmotionExpressionPreset>& Presets,
		TArray<FName>& OutControlNames);

	/**
	 * Returns the value for ControlName in the preset matching EmotionLabel,
	 * or 0 if no match (case-insensitive).  Includes the preset's
	 * BaseIntensity multiplier.  Confidence scaling is applied separately
	 * at the keyframe level so we can have different confidence per segment.
	 */
	static float GetPresetValue(
		const FName& ControlName,
		const FString& EmotionLabel,
		const TArray<FEmotionExpressionPreset>& Presets);
};
