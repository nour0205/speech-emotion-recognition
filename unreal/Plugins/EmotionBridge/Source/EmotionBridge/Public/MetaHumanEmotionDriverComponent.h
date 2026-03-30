// Copyright (c) EmotionDemo Project. All rights reserved.
// Phase 2B — drives MetaHuman face morph targets from the EmotionBridge timeline.
#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "EmotionMetaHumanTypes.h"
#include "EmotionTimelineTypes.h"
#include "MetaHumanEmotionDriverComponent.generated.h"

class USkeletalMeshComponent;

/**
 * UMetaHumanEmotionDriverComponent
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * WHAT IT DOES
 * ─────────────────────────────────────────────────────────────────────────────
 * Attach this component to a MetaHuman Actor (or any actor with a face
 * SkeletalMeshComponent).  The component drives facial morph targets to express
 * emotions from the EmotionBridge API timeline with smooth crossfades.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * LAYER ARCHITECTURE
 * ─────────────────────────────────────────────────────────────────────────────
 * This component implements LAYER 2 (Emotion Overlay) only.
 *
 *   Layer 1 — MetaHuman audio-driven speech animation
 *             Provided by the MetaHuman Animation Blueprint + audio-driven rig.
 *             Handles jaw, lips, phoneme-based facial motion.
 *
 *   Layer 2 — Emotion overlay (THIS COMPONENT)
 *             Drives brow, cheeks, eye squint, nose wrinkle, and mouth corners.
 *             These regions carry emotional character without interfering with
 *             the phoneme motion owned by Layer 1.
 *
 *   Layer 3 — Editor tooling
 *             The Emotion Bridge editor panel configures and drives this component.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * USAGE — editor panel (external drive mode, recommended for Phase 2B)
 * ─────────────────────────────────────────────────────────────────────────────
 *   1. Select your MetaHuman Actor in the viewport.
 *   2. In the Emotion Bridge panel → METAHUMAN FACE section, click
 *      "Bind Selected Actor". The component is added automatically.
 *   3. Click Analyze, then Play Demo.  Emotion is applied each frame.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * USAGE — runtime / Blueprint (autonomous playback mode)
 * ─────────────────────────────────────────────────────────────────────────────
 *   1. Add this component to your MetaHuman Actor Blueprint.
 *   2. After receiving the API response, call SetTimeline(TimelineResponse).
 *   3. Call StartPlayback() to begin autonomous, self-ticking playback.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * EXPRESSION PRESETS
 * ─────────────────────────────────────────────────────────────────────────────
 * Built-in ARKit-compatible presets are applied at BeginPlay / first tick if
 * ExpressionPresets is empty.  You can:
 *   a) Override ExpressionPresets in the Details panel (per-actor customisation).
 *   b) Call SetPresets() / SetPresetForEmotion() from C++ or Blueprint.
 *   c) Consult METAHUMAN_PHASE2B.md for the full tuning guide.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * MORPH TARGET COMPATIBILITY
 * ─────────────────────────────────────────────────────────────────────────────
 * Default presets use ARKit blendshape names (browDown_L, cheekSquint_R, …).
 * MetaHuman characters export these if set up for ARKit face capture.
 * To check which names your MetaHuman exposes:
 *   Select face mesh → Details → Morph Target Preview → note all curve names.
 * If none match, update ExpressionPresets with your actual morph target names.
 */
UCLASS(ClassGroup=(EmotionBridge),
       meta=(BlueprintSpawnableComponent,
             DisplayName="MetaHuman Emotion Driver"))
class EMOTIONBRIDGE_API UMetaHumanEmotionDriverComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UMetaHumanEmotionDriverComponent();

	// ─────────────────────────────────────────────────────────────────────────
	// Configuration — editable in Details panel
	// ─────────────────────────────────────────────────────────────────────────

	/**
	 * Controls blending, confidence scaling, per-emotion multipliers, and which
	 * SkeletalMeshComponent on the owner actor is treated as the face mesh.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge|MetaHuman",
		meta=(ShowOnlyInnerProperties))
	FEmotionOverlaySettings OverlaySettings;

	/**
	 * Expression presets for each canonical emotion label.
	 * If empty at BeginPlay / first tick, built-in ARKit defaults are applied.
	 * Override these per-actor in the Details panel to tune the facial shapes.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge|MetaHuman")
	TArray<FEmotionExpressionPreset> ExpressionPresets;

	// ─────────────────────────────────────────────────────────────────────────
	// External drive API — called by the editor panel's BroadcastEmotion()
	// ─────────────────────────────────────────────────────────────────────────

	/**
	 * Begin blending toward the given emotion.
	 *
	 * This is the primary method when the Emotion Bridge panel drives playback.
	 * Call it each time the active emotion segment changes.  The component
	 * handles all blending internally on subsequent ticks.
	 *
	 * Calling with the same emotion that is already active updates the
	 * confidence value but does NOT restart the blend.
	 *
	 * @param InEmotion    One of: "angry" | "happy" | "sad" | "neutral"
	 *                     Unknown labels are treated as "neutral".
	 * @param InConfidence API confidence value [0,1]. Scales intensity when
	 *                     OverlaySettings.bUseConfidenceAsWeight is true.
	 */
	UFUNCTION(BlueprintCallable, Category="EmotionBridge|MetaHuman")
	void ApplyEmotion(const FString& InEmotion, float InConfidence);

	/**
	 * Immediately zero all driven morph targets and reset the blend state to
	 * neutral.  Call this when playback stops or the actor is unbound.
	 */
	UFUNCTION(BlueprintCallable, Category="EmotionBridge|MetaHuman")
	void ResetToNeutral();

	// ─────────────────────────────────────────────────────────────────────────
	// Autonomous playback API — for runtime / Blueprint use without the panel
	// ─────────────────────────────────────────────────────────────────────────

	/**
	 * Store a timeline for autonomous playback.
	 * Resets the internal playback cursor to 0.  Does not start playback.
	 */
	UFUNCTION(BlueprintCallable, Category="EmotionBridge|MetaHuman|Playback")
	void SetTimeline(const FEmotionTimelineResponse& InTimeline);

	/**
	 * Start autonomous timeline playback.
	 * The component advances its own elapsed time each tick and calls
	 * ApplyEmotion() from the timeline data.  Audio is NOT played — launch
	 * a platform audio player separately if needed (see SEmotionBridgePanel).
	 * Does nothing when the timeline is empty or invalid.
	 */
	UFUNCTION(BlueprintCallable, Category="EmotionBridge|MetaHuman|Playback")
	void StartPlayback();

	/** Stop autonomous playback and reset the face to neutral. */
	UFUNCTION(BlueprintCallable, Category="EmotionBridge|MetaHuman|Playback")
	void StopPlayback();

	/** Returns true while autonomous playback is running. */
	UFUNCTION(BlueprintPure, Category="EmotionBridge|MetaHuman|Playback")
	bool IsPlaying() const { return bIsAutonomousPlaying; }

	// ─────────────────────────────────────────────────────────────────────────
	// Introspection
	// ─────────────────────────────────────────────────────────────────────────

	/** Returns the current target emotion label (the "to" side of any ongoing blend). */
	UFUNCTION(BlueprintPure, Category="EmotionBridge|MetaHuman")
	FString GetCurrentEmotion() const { return BlendState.ToEmotion; }

	/**
	 * Returns blend progress in [0,1]:
	 *   0.0 = just started blending from previous emotion.
	 *   1.0 = fully settled at current emotion.
	 */
	UFUNCTION(BlueprintPure, Category="EmotionBridge|MetaHuman")
	float GetBlendAlpha() const { return BlendState.BlendAlpha; }

	/**
	 * Returns true when a face SkeletalMeshComponent has been successfully
	 * resolved on the owner actor (either auto-detected or explicitly set).
	 */
	UFUNCTION(BlueprintPure, Category="EmotionBridge|MetaHuman")
	bool HasValidFaceMesh() const { return CachedFaceMesh.IsValid(); }

	// ─────────────────────────────────────────────────────────────────────────
	// Direct mesh binding
	// ─────────────────────────────────────────────────────────────────────────

	/**
	 * Explicitly set which SkeletalMeshComponent to drive.
	 * Bypasses auto-detection.  Call this if the auto-detected mesh is wrong.
	 */
	UFUNCTION(BlueprintCallable, Category="EmotionBridge|MetaHuman")
	void SetFaceMeshComponent(USkeletalMeshComponent* InMesh);

	// ─────────────────────────────────────────────────────────────────────────
	// Preset management
	// ─────────────────────────────────────────────────────────────────────────

	/**
	 * Replace all expression presets.  The new set takes effect immediately
	 * on the next tick.
	 */
	UFUNCTION(BlueprintCallable, Category="EmotionBridge|MetaHuman")
	void SetPresets(const TArray<FEmotionExpressionPreset>& InPresets);

	/**
	 * Add or replace a single preset by emotion name.
	 * Useful for hot-tuning one emotion without touching the others.
	 */
	UFUNCTION(BlueprintCallable, Category="EmotionBridge|MetaHuman")
	void SetPresetForEmotion(const FEmotionExpressionPreset& InPreset);

	// ─────────────────────────────────────────────────────────────────────────
	// Default presets — also useful as a starting point for custom presets
	// ─────────────────────────────────────────────────────────────────────────

	/**
	 * Returns built-in conservative presets for the four canonical emotions.
	 * Based on ARKit 52 blendshape names (MetaHuman standard).
	 * These weights target the upper face (brow/cheeks/eyes) and keep mouth
	 * influence minimal to preserve speech intelligibility.
	 *
	 * Call this from Blueprint to get a starting point, then modify:
	 *   TArray<FEmotionExpressionPreset> P = UMetaHumanEmotionDriverComponent::MakeDefaultPresets();
	 *   // tweak P[0].MorphWeights...
	 *   DriverComponent->SetPresets(P);
	 */
	UFUNCTION(BlueprintCallable, Category="EmotionBridge|MetaHuman")
	static TArray<FEmotionExpressionPreset> MakeDefaultPresets();

	// ─────────────────────────────────────────────────────────────────────────
	// UActorComponent overrides
	// ─────────────────────────────────────────────────────────────────────────

	virtual void BeginPlay() override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType,
		FActorComponentTickFunction* ThisTickFunction) override;

#if WITH_EDITOR
	virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

protected:
	// Internal helpers.
	void TickBlend(float DeltaTime);
	void ApplyBlendStateToMesh();
	USkeletalMeshComponent* ResolveFaceMesh();
	const FEmotionExpressionPreset* FindPreset(const FString& EmotionName) const;
	void EnsurePresetsInitialized();
	void RebuildMorphTargetSet();

	/**
	 * Compute the final blended + scaled weight for a single morph target.
	 * @param MorphTargetName  Name to look up in From and To presets.
	 * @param FromPreset       Outgoing emotion preset (may be nullptr → treat as neutral).
	 * @param ToPreset         Incoming emotion preset (may be nullptr → treat as neutral).
	 * @param Alpha            Blend progress [0,1].
	 * @param EffectiveIntensity Combined intensity scale (confidence × preset × multiplier).
	 */
	float ComputeBlendedWeight(
		FName MorphTargetName,
		const FEmotionExpressionPreset* FromPreset,
		const FEmotionExpressionPreset* ToPreset,
		float Alpha,
		float EffectiveIntensity) const;

private:
	/** Resolved face SkeletalMeshComponent. Cached lazily; invalidated on actor changes. */
	UPROPERTY(Transient)
	TWeakObjectPtr<USkeletalMeshComponent> CachedFaceMesh;

	/** Current blend state — transient, not persisted. */
	FEmotionBlendState BlendState;

	/** Union of all morph target names used by any preset.  Rebuilt when presets change. */
	TSet<FName> AllDrivenMorphTargets;

	bool bPresetsInitialized = false;

	// --- Autonomous playback ---
	FEmotionTimelineResponse AutonomousTimeline;
	float                    AutonomousElapsedSec       = 0.f;
	bool                     bIsAutonomousPlaying       = false;
	int32                    LastAutonomousSegmentIndex = -1;
};
