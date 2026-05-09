// Copyright (c) EmotionDemo Project. All rights reserved.
// Phase B — Wraps MetaHuman Performance's audio-to-face pipeline so the
// Emotion Bridge panel can produce a face AnimSequence from a SoundWave
// without the user ever opening the Content Browser.
#pragma once

#include "CoreMinimal.h"

class UAnimSequence;
class USkeletalMesh;
class USoundWave;

/**
 * Result of a Performance run.  AnimSequence is non-null on success and
 * already saved to /Game on disk.  ErrorMessage is human-readable when
 * AnimSequence is null.
 */
struct FEmotionPerformanceRunResult
{
	UAnimSequence* AnimSequence = nullptr;
	FString        ErrorMessage;
};

/**
 * Editor-only utility that drives MetaHuman Performance's audio-to-face
 * pipeline directly from C++.  This is the same pipeline the engine uses
 * when the user right-clicks a SoundWave → MetaHuman Performance →
 * "Process and Export to Anim Sequence" — we just bypass the UI.
 *
 * The implementation mirrors UMetaHumanBatchOperation in the engine plugin
 * (E:\Program Files\Epic Games\UE_5.7\Engine\Plugins\MetaHuman\
 *  MetaHumanAnimator\Source\MetaHumanBatchProcessor\
 *  Private\MetaHumanBatchOperation.cpp), specifically the SetupPerformance
 * → ProcessPerformanceAsset → ExportAnimationSequence sequence, but uses
 * the public StartPipeline + SetBlockingProcessing path instead of building
 * an inline FSpeechToAnimNode pipeline (avoids private engine headers).
 */
class FEmotionPerformanceRunner
{
public:
	/**
	 * Runs MetaHuman Performance synchronously on the game thread and
	 * returns the saved face AnimSequence.
	 *
	 * @param InSoundWave           The audio source.  Must be a real
	 *                              SoundWave asset (Performance loads its
	 *                              decoded audio internally).  Required.
	 * @param InTargetFaceMesh      Skeletal mesh whose skeleton drives the
	 *                              exported curves — typically the bound
	 *                              MetaHuman actor's "Face" component
	 *                              SkeletalMesh.  Required.
	 * @param InOutputPackagePath   Content path for the output AnimSequence,
	 *                              e.g. "/Game/EmotionBridge/Generated/<takeId>".
	 * @param InOutputAssetName     Name for the AnimSequence,
	 *                              e.g. "Anim_FaceEmotion_<takeId>".
	 * @param bGenerateBlinks       Forwarded to UMetaHumanPerformance::bGenerateBlinks.
	 * @param bDownmixChannels      Forwarded to UMetaHumanPerformance::bDownmixChannels.
	 *
	 * @return AnimSequence + empty error on success, null + error on failure.
	 */
	static FEmotionPerformanceRunResult RunAudioToFace(
		USoundWave*    InSoundWave,
		USkeletalMesh* InTargetFaceMesh,
		const FString& InOutputPackagePath,
		const FString& InOutputAssetName,
		bool           bGenerateBlinks  = true,
		bool           bDownmixChannels = true);
};
