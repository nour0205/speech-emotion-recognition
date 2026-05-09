// Copyright (c) EmotionDemo Project. All rights reserved.
// M2 — Populates a bound Level Sequence with the audio + face animation
// produced by the EmotionBridge bake pipeline.
#pragma once

#include "CoreMinimal.h"

class AActor;
class ULevelSequence;
class UAnimSequence;
class USoundBase;

/**
 * Replaces the audio + face-animation tracks of a user-bound Level Sequence
 * with new content from an EmotionBridge "Bake & Export" run.
 *
 * Workflow:
 *   1. The user picks a Level Sequence asset once in the Emotion Bridge
 *      panel.  We don't create new sequences per take — we keep updating
 *      the same one.
 *   2. Each "Bake & Export" call populates that sequence with:
 *        - the take's SoundWave on a master audio track,
 *        - the baked AnimSequence on a skeletal-animation track bound to
 *          the MetaHuman actor's Face SkeletalMeshComponent.
 *   3. Existing audio + face-animation tracks are removed first so re-runs
 *      are idempotent and don't accumulate duplicates.
 *
 * The MetaHuman's body / hair / outfit bindings are left untouched if the
 * user has set them up separately.
 */
class FEmotionSequenceExporter
{
public:
	/**
	 * @param TargetSeq      Level Sequence asset to update.  Must not be null.
	 * @param MetaHumanActor Actor in the level whose Face SkeletalMeshComponent
	 *                       will receive the animation track.  Must not be null
	 *                       and must currently exist in a world.
	 * @param AudioAsset     SoundWave (or any USoundBase) to play.
	 *                       Pass null to skip the audio track.
	 * @param FaceAnim       AnimSequence to play on the Face component.
	 *                       Pass null to skip the animation track.
	 * @param OutErrorMsg    Filled with a human-readable reason when the
	 *                       function returns false.
	 *
	 * @return true on success.  Sequence is marked dirty on success; caller
	 *         must save the package if persistence is desired.
	 */
	static bool PopulateSequence(
		ULevelSequence* TargetSeq,
		AActor* MetaHumanActor,
		USoundBase* AudioAsset,
		UAnimSequence* FaceAnim,
		FString& OutErrorMsg);
};
