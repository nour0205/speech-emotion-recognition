// Copyright (c) EmotionDemo Project. All rights reserved.
// M1 — Implementation.

#include "EmotionAnimSequenceBaker.h"

#include "EmotionTimelineTypes.h"
#include "EmotionMetaHumanTypes.h"
#include "EmotionBridgeLog.h"

#include "Animation/AnimSequence.h"
#include "Animation/AnimData/IAnimationDataController.h"
#include "Animation/AnimData/IAnimationDataModel.h"
#include "Animation/AnimCurveTypes.h"
#include "Curves/RichCurve.h"

#define LOCTEXT_NAMESPACE "EmotionAnimSequenceBaker"

namespace
{
	/** A keyframe being staged before we hand it to the controller. */
	struct FStagedKey
	{
		float Time;
		float Value;
	};

	/**
	 * Removes redundant adjacent keys whose value didn't change AND whose
	 * neighbours don't change either — i.e. flat runs collapse to two keys
	 * (start + end).  Keeps the curve compact without losing information.
	 */
	void CompactKeys(TArray<FStagedKey>& Keys)
	{
		if (Keys.Num() <= 2) return;

		TArray<FStagedKey> Out;
		Out.Reserve(Keys.Num());
		Out.Add(Keys[0]);
		for (int32 i = 1; i < Keys.Num() - 1; ++i)
		{
			const float Prev = Keys[i - 1].Value;
			const float Cur  = Keys[i].Value;
			const float Next = Keys[i + 1].Value;
			if (FMath::IsNearlyEqual(Prev, Cur, 1e-5f)
				&& FMath::IsNearlyEqual(Cur, Next, 1e-5f))
			{
				continue; // middle of a flat run
			}
			Out.Add(Keys[i]);
		}
		Out.Add(Keys.Last());
		Keys = MoveTemp(Out);
	}

	/**
	 * Ensures keys are strictly time-ordered.  When two would land at the
	 * exact same time (rare, happens at duration_sec boundary), the second
	 * is nudged forward by a frame.  This keeps the controller happy.
	 */
	void EnforceMonotonicTimes(TArray<FStagedKey>& Keys)
	{
		const float Epsilon = 1.0f / 240.0f; // sub-frame
		for (int32 i = 1; i < Keys.Num(); ++i)
		{
			if (Keys[i].Time <= Keys[i - 1].Time)
			{
				Keys[i].Time = Keys[i - 1].Time + Epsilon;
			}
		}
	}
}

bool FEmotionAnimSequenceBaker::BakeEmotionCurves(
	UAnimSequence* AnimSeq,
	const FEmotionTimelineResponse& Timeline,
	const TArray<FEmotionExpressionPreset>& Presets,
	float BlendDurationSec,
	bool bUseConfidenceAsWeight)
{
#if !WITH_EDITOR
	return false;
#else
	if (!AnimSeq)
	{
		UE_LOG(LogEmotionBridge, Error,
			TEXT("BakeEmotionCurves: AnimSequence is null."));
		return false;
	}
	if (!Timeline.bIsValid)
	{
		UE_LOG(LogEmotionBridge, Error,
			TEXT("BakeEmotionCurves: timeline is invalid: %s"), *Timeline.ErrorMessage);
		return false;
	}
	if (Timeline.Segments.IsEmpty())
	{
		UE_LOG(LogEmotionBridge, Warning,
			TEXT("BakeEmotionCurves: timeline has zero segments — nothing to bake."));
		return false;
	}
	if (Presets.IsEmpty())
	{
		UE_LOG(LogEmotionBridge, Error,
			TEXT("BakeEmotionCurves: no presets supplied."));
		return false;
	}

	const float SafeBlend     = FMath::Max(0.f, BlendDurationSec);
	const float HalfBlend     = SafeBlend * 0.5f;
	const float TimelineEnd   = FMath::Max(Timeline.DurationSec, Timeline.Segments.Last().EndSec);

	TArray<FName> ControlNames;
	CollectAllControlNames(Presets, ControlNames);

	if (ControlNames.IsEmpty())
	{
		UE_LOG(LogEmotionBridge, Warning,
			TEXT("BakeEmotionCurves: presets reference 0 control names — nothing to write."));
		return false;
	}

	// ── Diagnostic: inventory the curves already present + find (mis)matches
	//    This is critical because the rig's display names ("CTRL_L_brow_down")
	//    differ from the animation curve names ("ctrl_expressions_browdownl")
	//    that RigLogic actually reads.  A preset name that isn't in the
	//    AnimSequence's curve list is baked but silently ignored at playback.
	{
		TSet<FName> ExistingCurves;
		if (const IAnimationDataModel* Model = AnimSeq->GetDataModel())
		{
			for (const FFloatCurve& C : Model->GetFloatCurves())
			{
				ExistingCurves.Add(C.GetName());
			}
		}

		TArray<FName> Matched;
		TArray<FName> Unmatched;
		for (const FName& Name : ControlNames)
		{
			if (ExistingCurves.Contains(Name)) { Matched.Add(Name); }
			else                               { Unmatched.Add(Name); }
		}

		UE_LOG(LogEmotionBridge, Log,
			TEXT("BakeEmotionCurves: existing AnimSequence has %d float curves. "
			     "Preset controls: %d matched, %d unmatched."),
			ExistingCurves.Num(), Matched.Num(), Unmatched.Num());

		if (Unmatched.Num() > 0)
		{
			FString UnmatchedBlob;
			int32 Shown = 0;
			for (const FName& N : Unmatched)
			{
				if (Shown++ >= 16) { UnmatchedBlob += TEXT(", ..."); break; }
				if (!UnmatchedBlob.IsEmpty()) UnmatchedBlob += TEXT(", ");
				UnmatchedBlob += N.ToString();
			}
			UE_LOG(LogEmotionBridge, Warning,
				TEXT("BakeEmotionCurves: preset names NOT found in '%s' — will be baked but ignored by RigLogic:\n  %s"),
				*AnimSeq->GetName(), *UnmatchedBlob);

			// For each unmatched preset name, dump every existing curve in the
			// same region (first word after "ctrl_expressions_") so the correct
			// name is obvious at a glance.  Example: a miss on
			//   ctrl_expressions_mouthcornerdepressorl
			// prints every "ctrl_expressions_mouth*" curve on the mesh.
			for (const FName& Missing : Unmatched)
			{
				const FString MissingStr = Missing.ToString();

				// Extract the region word (everything after "ctrl_expressions_"
				// up to the first 4 alpha chars, which is enough to pick the
				// body region: mouth / nose / brow / eye / cheek / jaw).
				FString Region;
				{
					static const FString kPfx = TEXT("ctrl_expressions_");
					if (MissingStr.StartsWith(kPfx))
					{
						const FString Tail = MissingStr.RightChop(kPfx.Len());
						// Grab the longest alpha prefix (the region word).
						int32 AlphaEnd = 0;
						while (AlphaEnd < Tail.Len() && FChar::IsAlpha(Tail[AlphaEnd])) { ++AlphaEnd; }
						// Take first 4–5 chars as the region key.
						Region = Tail.Left(FMath::Min(AlphaEnd, 5));
					}
				}

				if (Region.IsEmpty())
				{
					UE_LOG(LogEmotionBridge, Warning,
						TEXT("  '%s' — no region extracted, skipping suggestions."),
						*MissingStr);
					continue;
				}

				const FString RegionPrefix = FString::Printf(TEXT("ctrl_expressions_%s"), *Region);

				TArray<FString> RegionCurves;
				for (const FName& Existing : ExistingCurves)
				{
					const FString ExistingStr = Existing.ToString();
					if (ExistingStr.StartsWith(RegionPrefix, ESearchCase::IgnoreCase))
					{
						RegionCurves.Add(ExistingStr);
					}
				}
				RegionCurves.Sort();

				UE_LOG(LogEmotionBridge, Warning,
					TEXT("  '%s' — all %d curves in region '%s*':"),
					*MissingStr, RegionCurves.Num(), *RegionPrefix);
				for (const FString& C : RegionCurves)
				{
					UE_LOG(LogEmotionBridge, Warning, TEXT("      %s"), *C);
				}
			}
		}
	}

	IAnimationDataController& Controller = AnimSeq->GetController();
	Controller.OpenBracket(LOCTEXT("BakeBracket", "Bake EmotionBridge curves"));

	int32 CurvesWritten = 0;

	for (const FName& ControlName : ControlNames)
	{
		// ── Build keyframe list for this control ──────────────────────────
		TArray<FStagedKey> Keys;
		Keys.Reserve(Timeline.Segments.Num() * 2 + 2);

		// Start at neutral.
		Keys.Add({ 0.f, 0.f });

		float PrevTargetValue = 0.f;
		for (int32 SegIdx = 0; SegIdx < Timeline.Segments.Num(); ++SegIdx)
		{
			const FEmotionSegment& Seg = Timeline.Segments[SegIdx];
			const float PresetVal = GetPresetValue(ControlName, Seg.Emotion, Presets);
			const float ConfScale = bUseConfidenceAsWeight
				? FMath::Clamp(Seg.Confidence, 0.f, 1.f)
				: 1.f;
			const float TargetValue = PresetVal * ConfScale;

			// Crossfade ramp centred on the segment-start boundary.
			const float RampStart = FMath::Max(0.f, Seg.StartSec - HalfBlend);
			const float RampEnd   = FMath::Min(TimelineEnd, Seg.StartSec + HalfBlend);

			// Anchor the previous-segment value just before the ramp.
			Keys.Add({ RampStart, PrevTargetValue });
			// And the new target just after.
			Keys.Add({ RampEnd, TargetValue });

			PrevTargetValue = TargetValue;
		}

		// Decay the final value back to neutral over BlendDurationSec.
		const float DecayStart = FMath::Max(0.f, TimelineEnd - HalfBlend);
		Keys.Add({ DecayStart, PrevTargetValue });
		Keys.Add({ TimelineEnd, 0.f });

		EnforceMonotonicTimes(Keys);
		CompactKeys(Keys);

		// ── Write the curve via the controller ────────────────────────────
		const FAnimationCurveIdentifier CurveId(ControlName, ERawCurveTrackTypes::RCT_Float);

		// Remove any previous version (re-bakes are idempotent).
		if (AnimSeq->GetDataModel()->FindFloatCurve(CurveId))
		{
			Controller.RemoveCurve(CurveId);
		}
		Controller.AddCurve(CurveId);

		TArray<FRichCurveKey> RichKeys;
		RichKeys.Reserve(Keys.Num());
		for (const FStagedKey& K : Keys)
		{
			FRichCurveKey RK(K.Time, K.Value);
			RK.InterpMode = RCIM_Linear;
			RichKeys.Add(RK);
		}
		Controller.SetCurveKeys(CurveId, RichKeys);
		++CurvesWritten;
	}

	Controller.CloseBracket();
	AnimSeq->MarkPackageDirty();

	UE_LOG(LogEmotionBridge, Log,
		TEXT("BakeEmotionCurves: wrote %d curves to '%s' from %d segments (duration=%.2fs, blend=%.2fs)."),
		CurvesWritten, *AnimSeq->GetPathName(), Timeline.Segments.Num(),
		TimelineEnd, SafeBlend);

	return CurvesWritten > 0;
#endif
}

void FEmotionAnimSequenceBaker::CollectAllControlNames(
	const TArray<FEmotionExpressionPreset>& Presets,
	TArray<FName>& OutControlNames)
{
	TSet<FName> Seen;
	OutControlNames.Reset();
	for (const FEmotionExpressionPreset& P : Presets)
	{
		for (const FEmotionMorphWeight& MW : P.MorphWeights)
		{
			if (MW.MorphTargetName.IsNone()) continue;
			bool bAlready = false;
			Seen.Add(MW.MorphTargetName, &bAlready);
			if (!bAlready)
			{
				OutControlNames.Add(MW.MorphTargetName);
			}
		}
	}
}

float FEmotionAnimSequenceBaker::GetPresetValue(
	const FName& ControlName,
	const FString& EmotionLabel,
	const TArray<FEmotionExpressionPreset>& Presets)
{
	const FString LabelLower = EmotionLabel.ToLower();
	for (const FEmotionExpressionPreset& P : Presets)
	{
		if (P.EmotionName.ToLower() != LabelLower) continue;
		const float W = P.FindWeight(ControlName);
		return W * FMath::Clamp(P.BaseIntensity, 0.f, 2.f);
	}
	return 0.f;
}

#undef LOCTEXT_NAMESPACE
