// Copyright (c) EmotionDemo Project. All rights reserved.
// Phase 2B — MetaHuman emotion overlay types.
//
// This file defines the data structures for Layer 2 (Emotion Overlay) of the
// EmotionBridge facial animation system.
//
// LAYER MODEL:
//   Layer 1 — Base speech animation (MetaHuman audio-driven AnimBP — not in this file)
//   Layer 2 — Emotion overlay: this file + UMetaHumanEmotionDriverComponent
//   Layer 3 — Tooling/Control: SEmotionBridgePanel (editor module)
#pragma once

#include "CoreMinimal.h"
#include "EmotionMetaHumanTypes.generated.h"

// ---------------------------------------------------------------------------
// FEmotionMorphWeight
//
// One morph target name and its target weight for a given emotional state.
// Used inside FEmotionExpressionPreset to describe a face shape.
// ---------------------------------------------------------------------------

USTRUCT(BlueprintType)
struct EMOTIONBRIDGE_API FEmotionMorphWeight
{
	GENERATED_BODY()

	/**
	 * The morph target name exactly as it appears on the SkeletalMeshComponent.
	 *
	 * For MetaHuman faces, common names follow the ARKit 52 blendshape convention:
	 *   browDown_L, browDown_R, browInnerUp, cheekSquint_L, cheekSquint_R,
	 *   eyeSquint_L, eyeSquint_R, noseSneer_L, noseSneer_R,
	 *   mouthSmile_L, mouthSmile_R, mouthFrown_L, mouthFrown_R …
	 *
	 * To discover morph target names on your MetaHuman:
	 *   Select the Face mesh component → Details → Morph Target Preview section.
	 *   Every listed curve name is a valid MorphTargetName.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge|MetaHuman")
	FName MorphTargetName;

	/**
	 * Target weight in [0, 1] when this emotion is fully active at full intensity.
	 * Values outside [0, 1] are valid for additive correction but uncommon.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge|MetaHuman",
		meta=(ClampMin="0.0", ClampMax="1.0"))
	float Weight = 0.f;
};

// ---------------------------------------------------------------------------
// FEmotionExpressionPreset
//
// The complete facial configuration for ONE canonical emotion.
//
// SPEECH SAFETY RULE:
//   Do NOT include jaw or primary lip phoneme targets here unless you use very
//   low weights (< 0.1).  The base MetaHuman speech animation owns those targets.
//   Focus this preset on: brow, cheeks, eye squint, nose wrinkle, and the very
//   corners of the mouth.  These areas carry emotion without fighting lip sync.
//
// NEUTRAL PRESET:
//   Leave MorphWeights empty.  The driver returns all targets to 0 over the
//   BlendDurationSec when neutral is the active emotion.
// ---------------------------------------------------------------------------

USTRUCT(BlueprintType)
struct EMOTIONBRIDGE_API FEmotionExpressionPreset
{
	GENERATED_BODY()

	/**
	 * Canonical emotion label.  Must match labels returned by the API exactly:
	 *   "angry" | "happy" | "sad" | "neutral"
	 * Case-insensitive comparison is used when looking up presets.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge|MetaHuman")
	FString EmotionName;

	/**
	 * All morph targets and their target weights for this expression.
	 * Only list targets this emotion actually drives.  Targets listed in OTHER
	 * presets but absent here will automatically blend back to 0 during this emotion.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge|MetaHuman")
	TArray<FEmotionMorphWeight> MorphWeights;

	/**
	 * Global intensity scale applied on top of weight values and confidence scaling.
	 * Range [0, 1].  Use this to globally dial down a preset that is too strong
	 * without changing individual weight values.
	 * Default: 1.0 (full intensity).
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge|MetaHuman",
		meta=(ClampMin="0.0", ClampMax="1.0"))
	float BaseIntensity = 1.f;

	/** Helper: find the weight for a given morph target name. Returns 0 if not found. */
	float FindWeight(FName InName) const
	{
		for (const FEmotionMorphWeight& MW : MorphWeights)
			if (MW.MorphTargetName == InName) return MW.Weight;
		return 0.f;
	}
};

// ---------------------------------------------------------------------------
// FEmotionOverlaySettings
//
// Runtime-configurable settings governing the emotion overlay layer.
// Exposed on UMetaHumanEmotionDriverComponent and editable in the Details panel
// or from the Emotion Bridge editor tab.
// ---------------------------------------------------------------------------

USTRUCT(BlueprintType)
struct EMOTIONBRIDGE_API FEmotionOverlaySettings
{
	GENERATED_BODY()

	/**
	 * Master enable/bypass switch for the entire emotion overlay layer.
	 * When false, the component still ticks but writes nothing to the face mesh,
	 * allowing the AnimBP speech animation to run unmodified.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge|MetaHuman")
	bool bEnabled = true;

	/**
	 * Duration of the crossfade between one emotion and the next, in seconds.
	 * Recommended range: 0.2–0.8 s.
	 *   0.0 = hard cut (not recommended — looks mechanical).
	 *   0.4 = good default for most speech performances.
	 *   0.8+ = very slow, "dreamy" feel — useful for ambient visuals.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge|MetaHuman",
		meta=(ClampMin="0.0", ClampMax="5.0"))
	float BlendDurationSec = 0.4f;

	/**
	 * When true, the API confidence value [0,1] is used as an additional
	 * intensity multiplier.  High-confidence segments get full expression;
	 * low-confidence segments appear more subtle.
	 *
	 * When false, confidence is ignored and expression intensity is driven
	 * purely by the preset BaseIntensity and EmotionIntensityMultipliers.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge|MetaHuman")
	bool bUseConfidenceAsWeight = true;

	/**
	 * Per-emotion intensity multiplier.  Key = lowercase emotion label.
	 * Applied AFTER confidence scaling and preset BaseIntensity.
	 *
	 * Use this to make certain emotions stronger or weaker relative to others.
	 * Example: { "angry": 1.4, "happy": 1.0, "sad": 0.8, "neutral": 1.0 }
	 *
	 * Default: empty map (all multipliers default to 1.0).
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge|MetaHuman")
	TMap<FString, float> EmotionIntensityMultipliers;

	/**
	 * Name of the SkeletalMeshComponent on the owner actor to drive.
	 * If None, the component auto-detects the best face mesh:
	 *   1. First component whose name contains "Face" (MetaHuman convention).
	 *   2. First SkeletalMeshComponent found on the actor.
	 *
	 * Override this if auto-detection picks the wrong mesh on actors
	 * that have multiple SkeletalMeshComponents (body, hair, teeth, etc.).
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge|MetaHuman")
	FName FaceMeshComponentName = NAME_None;
};

// ---------------------------------------------------------------------------
// FEmotionBlendState
//
// Transient runtime state tracking the in-progress blend between two emotions.
// This struct lives inside UMetaHumanEmotionDriverComponent and is NOT persisted
// to disk (it resets to neutral on component creation).
// ---------------------------------------------------------------------------

struct EMOTIONBRIDGE_API FEmotionBlendState
{
	/** Emotion we are blending FROM (the previous or partially blended state). */
	FString FromEmotion = TEXT("neutral");

	/** Emotion we are blending TO (the newly requested state). */
	FString ToEmotion = TEXT("neutral");

	/**
	 * Blend progress: 0.0 = fully at FromEmotion, 1.0 = fully at ToEmotion.
	 * Advances at (DeltaTime / BlendDurationSec) per tick, clamped to [0,1].
	 */
	float BlendAlpha = 1.f;

	/** API confidence of the current ToEmotion segment. */
	float ToConfidence = 1.f;

	/** True when a blend is actively in progress (BlendAlpha has not yet reached 1). */
	bool IsBlending() const { return BlendAlpha < 1.f - KINDA_SMALL_NUMBER; }

	/** The settled emotion (valid only when BlendAlpha == 1). */
	const FString& GetCurrentEmotion() const { return ToEmotion; }
};
