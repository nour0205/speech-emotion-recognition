// Copyright (c) EmotionDemo Project. All rights reserved.
// Phase B — Implementation.  Replicates the canonical audio-to-face flow
// from UMetaHumanBatchOperation::SetupPerformance + ProcessPerformanceAsset
// + ExportAnimationSequence (engine source), but without the inline
// FSpeechToAnimNode pipeline (we use the public StartPipeline path so we
// don't need MetaHumanPipeline private includes).

#include "EmotionPerformanceRunner.h"

#include "EmotionBridgeLog.h"

#include "MetaHumanPerformance.h"
#include "MetaHumanPerformanceExportUtils.h"

#include "Animation/AnimSequence.h"
#include "Engine/SkeletalMesh.h"
#include "Sound/SoundWave.h"
#include "Misc/ScopedSlowTask.h"
#include "UObject/Package.h"
#include "UObject/UnrealType.h"

#define LOCTEXT_NAMESPACE "EmotionPerformanceRunner"

FEmotionPerformanceRunResult FEmotionPerformanceRunner::RunAudioToFace(
	USoundWave*    InSoundWave,
	USkeletalMesh* InTargetFaceMesh,
	const FString& InOutputPackagePath,
	const FString& InOutputAssetName,
	bool           bGenerateBlinks,
	bool           bDownmixChannels)
{
#if !WITH_EDITOR
	FEmotionPerformanceRunResult Result;
	Result.ErrorMessage = TEXT("EmotionPerformanceRunner requires editor build.");
	return Result;
#else
	FEmotionPerformanceRunResult Result;

	// ── Validate inputs ──────────────────────────────────────────────────────
	if (!InSoundWave)
	{
		Result.ErrorMessage = TEXT("SoundWave is null.");
		return Result;
	}
	if (!InTargetFaceMesh)
	{
		Result.ErrorMessage = TEXT("Target face mesh is null. Bind a MetaHuman actor with a 'Face' SkeletalMeshComponent.");
		return Result;
	}
	if (InOutputPackagePath.IsEmpty() || InOutputAssetName.IsEmpty())
	{
		Result.ErrorMessage = TEXT("Output package path/name is empty.");
		return Result;
	}

	// ── Progress dialog (non-blocking but visible during the long-running step)
	FScopedSlowTask Progress(3.0f, FText::Format(
		LOCTEXT("Slow", "MetaHuman Performance: '{0}'"),
		FText::FromName(InSoundWave->GetFName())));
	Progress.MakeDialog(/*bCanCancel=*/false);

	UE_LOG(LogEmotionBridge, Log,
		TEXT("EmotionPerformanceRunner: starting Performance for SoundWave='%s' duration=%.2fs faceMesh='%s'"),
		*InSoundWave->GetName(), InSoundWave->GetDuration(),
		*InTargetFaceMesh->GetName());

	// ── Step 1: create a transient Performance asset ────────────────────────
	Progress.EnterProgressFrame(1.0f, LOCTEXT("Setup", "Configuring audio-to-face pipeline"));

	UMetaHumanPerformance* Performance = NewObject<UMetaHumanPerformance>(
		GetTransientPackage(), NAME_None, RF_Transient);
	if (!Performance)
	{
		Result.ErrorMessage = TEXT("Failed to NewObject<UMetaHumanPerformance>().");
		return Result;
	}

	// ── Step 2: configure for audio-only mode ───────────────────────────────
	Performance->InputType                       = EDataInputType::Audio;
	Performance->Audio                           = InSoundWave;
	Performance->VisualizationMesh               = InTargetFaceMesh;
	Performance->bGenerateBlinks                 = bGenerateBlinks;
	Performance->bDownmixChannels                = bDownmixChannels;
	Performance->HeadMovementMode                = EPerformanceHeadMovementMode::Disabled;
	Performance->AudioDrivenAnimationOutputControls = EAudioDrivenAnimationOutputControls::FullFace;

	// ── Step 3: trigger PostEditChangeProperty for Audio so internal frame
	//    ranges init.  Mirrors MetaHumanBatchOperation.cpp:261-264.
	{
		static const FName AudioPropertyName = GET_MEMBER_NAME_STRING_CHECKED(UMetaHumanPerformance, Audio);
		FProperty* AudioProperty = UMetaHumanPerformance::StaticClass()->FindPropertyByName(AudioPropertyName);
		if (AudioProperty)
		{
			FPropertyChangedEvent AudioChangedEvent(AudioProperty);
			Performance->PostEditChangeProperty(AudioChangedEvent);
		}
	}

	// ── Step 4: ask Performance to run synchronously on the game thread ────
	Performance->SetBlockingProcessing(true);

	if (!Performance->CanProcess())
	{
		const FText Why = Performance->GetCannotProcessTooltipText();
		Result.ErrorMessage = FString::Printf(
			TEXT("Performance refused to process: %s"), *Why.ToString());
		UE_LOG(LogEmotionBridge, Error, TEXT("%s"), *Result.ErrorMessage);
		return Result;
	}

	// ── Step 5: run the pipeline ────────────────────────────────────────────
	Progress.EnterProgressFrame(1.0f, LOCTEXT("Pipeline", "Running audio-to-face neural pipeline"));

	UE_LOG(LogEmotionBridge, Log,
		TEXT("EmotionPerformanceRunner: pipeline started (blocking)"));

	const EStartPipelineErrorType StartErr =
		Performance->StartPipeline(/*bIsScriptedProcessing=*/true);

	if (StartErr != EStartPipelineErrorType::None)
	{
		Result.ErrorMessage = FString::Printf(
			TEXT("StartPipeline returned error code %d (NoFrames=%d, Disabled=%d)."),
			static_cast<int32>(StartErr),
			static_cast<int32>(EStartPipelineErrorType::NoFrames),
			static_cast<int32>(EStartPipelineErrorType::Disabled));
		UE_LOG(LogEmotionBridge, Error, TEXT("%s"), *Result.ErrorMessage);
		return Result;
	}

	// SetBlockingProcessing(true) makes StartPipeline run synchronously, so
	// processing is done by the time we get here.  Defensive check:
	if (Performance->IsProcessing())
	{
		Result.ErrorMessage = TEXT("Pipeline is still IsProcessing() after blocking call — engine inconsistency.");
		UE_LOG(LogEmotionBridge, Error, TEXT("%s"), *Result.ErrorMessage);
		return Result;
	}

	if (!Performance->ContainsAnimationData())
	{
		Result.ErrorMessage = TEXT("Pipeline finished but produced no animation data.");
		UE_LOG(LogEmotionBridge, Error, TEXT("%s"), *Result.ErrorMessage);
		return Result;
	}

	if (!Performance->CanExportAnimation())
	{
		Result.ErrorMessage = TEXT("Pipeline finished but Performance refuses to export animation.");
		UE_LOG(LogEmotionBridge, Error, TEXT("%s"), *Result.ErrorMessage);
		return Result;
	}

	UE_LOG(LogEmotionBridge, Log,
		TEXT("EmotionPerformanceRunner: pipeline finished, frames=%d, exporting to '%s/%s'"),
		Performance->GetNumberOfProcessedFrames(),
		*InOutputPackagePath, *InOutputAssetName);

	// ── Step 6: export to AnimSequence ──────────────────────────────────────
	Progress.EnterProgressFrame(1.0f, LOCTEXT("Export", "Exporting AnimSequence"));

	UMetaHumanPerformanceExportAnimationSettings* Settings =
		NewObject<UMetaHumanPerformanceExportAnimationSettings>();
	Settings->bShowExportDialog              = false;
	Settings->bAutoSaveAnimSequence          = true;
	Settings->bEnableHeadMovement            = false;
	Settings->TargetSkeletonOrSkeletalMesh   = InTargetFaceMesh;
	Settings->ExportRange                    = EPerformanceExportRange::WholeSequence;
	Settings->CurveInterpolation             = RCIM_Linear;
	Settings->bRemoveRedundantKeys           = false;
	Settings->PackagePath                    = InOutputPackagePath;
	Settings->AssetName                      = InOutputAssetName;

	UAnimSequence* ExportedAnim = UMetaHumanPerformanceExportUtils::ExportAnimationSequence(
		Performance, Settings);

	if (!ExportedAnim)
	{
		Result.ErrorMessage = FString::Printf(
			TEXT("ExportAnimationSequence returned null for output '%s/%s'."),
			*InOutputPackagePath, *InOutputAssetName);
		UE_LOG(LogEmotionBridge, Error, TEXT("%s"), *Result.ErrorMessage);
		return Result;
	}

	UE_LOG(LogEmotionBridge, Log,
		TEXT("EmotionPerformanceRunner: exported '%s' (path='%s')."),
		*ExportedAnim->GetName(), *ExportedAnim->GetPathName());

	Result.AnimSequence = ExportedAnim;
	return Result;
#endif
}

#undef LOCTEXT_NAMESPACE
