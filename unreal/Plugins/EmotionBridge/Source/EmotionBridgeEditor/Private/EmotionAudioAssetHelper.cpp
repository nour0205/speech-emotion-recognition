// Copyright (c) EmotionDemo Project. All rights reserved.
// Phase 2B — FEmotionAudioAssetHelper implementation.

#include "EmotionAudioAssetHelper.h"
#include "EmotionBridgeLog.h"

#include "AssetToolsModule.h"
#include "IAssetTools.h"
#include "AssetImportTask.h"
#include "Misc/PackageName.h"
#include "Misc/Paths.h"
#include "Modules/ModuleManager.h"

// ---------------------------------------------------------------------------
// GetDefaultAudioContentPath
// ---------------------------------------------------------------------------

FString FEmotionAudioAssetHelper::GetDefaultAudioContentPath()
{
	return TEXT("/Game/EmotionBridge/Audio");
}

// ---------------------------------------------------------------------------
// FindExistingSoundWave
// ---------------------------------------------------------------------------

FString FEmotionAudioAssetHelper::FindExistingSoundWave(
	const FString& AbsoluteWavPath,
	const FString& ContentDestPath)
{
	const FString BaseName        = FPaths::GetBaseFilename(AbsoluteWavPath);
	const FString CandidatePkg    = ContentDestPath / BaseName;

	// FPackageName::DoesPackageExist() checks both the in-memory package list
	// and the on-disk .uasset file at the derived path.
	if (FPackageName::DoesPackageExist(CandidatePkg))
	{
		UE_LOG(LogEmotionBridge, Verbose,
			TEXT("FEmotionAudioAssetHelper: found existing SoundWave '%s'."), *CandidatePkg);
		return CandidatePkg;
	}

	return FString{};
}

// ---------------------------------------------------------------------------
// ImportWavAsSoundWave
// ---------------------------------------------------------------------------

FString FEmotionAudioAssetHelper::ImportWavAsSoundWave(
	const FString& AbsoluteWavPath,
	const FString& ContentDestPath,
	FString& OutErrorMessage)
{
	// ── Validate source ──────────────────────────────────────────────────────
	if (AbsoluteWavPath.IsEmpty())
	{
		OutErrorMessage = TEXT("WAV path is empty.");
		return FString{};
	}

	if (!FPaths::FileExists(AbsoluteWavPath))
	{
		OutErrorMessage = FString::Printf(
			TEXT("WAV file not found: '%s'"), *AbsoluteWavPath);
		return FString{};
	}

	if (FPaths::GetExtension(AbsoluteWavPath).ToLower() != TEXT("wav"))
	{
		OutErrorMessage = TEXT("Only .wav files can be imported as SoundWave assets.");
		return FString{};
	}

	// ── Already exists? ──────────────────────────────────────────────────────
	const FString ExistingPath = FindExistingSoundWave(AbsoluteWavPath, ContentDestPath);
	if (!ExistingPath.IsEmpty())
	{
		UE_LOG(LogEmotionBridge, Log,
			TEXT("FEmotionAudioAssetHelper: reusing existing SoundWave '%s'."), *ExistingPath);
		return ExistingPath;
	}

	// ── Import via AssetTools ────────────────────────────────────────────────
	if (!FModuleManager::Get().IsModuleLoaded(TEXT("AssetTools")))
	{
		OutErrorMessage = TEXT("AssetTools module is not available.");
		UE_LOG(LogEmotionBridge, Error, TEXT("FEmotionAudioAssetHelper: %s"), *OutErrorMessage);
		return FString{};
	}

	FAssetToolsModule& ATModule =
		FModuleManager::LoadModuleChecked<FAssetToolsModule>(TEXT("AssetTools"));
	IAssetTools& AssetTools = ATModule.Get();

	// Build an automated import task — no dialogs, save on complete.
	UAssetImportTask* ImportTask = NewObject<UAssetImportTask>();
	ImportTask->Filename         = AbsoluteWavPath;
	ImportTask->DestinationPath  = ContentDestPath;
	ImportTask->DestinationName  = FPaths::GetBaseFilename(AbsoluteWavPath);
	ImportTask->bAutomated       = true;   // suppress all import dialogs
	ImportTask->bReplaceExisting = false;
	ImportTask->bSave            = true;   // persist the .uasset to disk immediately

	TArray<UAssetImportTask*> Tasks = { ImportTask };
	AssetTools.ImportAssetTasks(Tasks);

	// ── Verify success ───────────────────────────────────────────────────────
	// After a synchronous automated import, the .uasset file should exist on disk.
	const FString ExpectedPkg = ContentDestPath / FPaths::GetBaseFilename(AbsoluteWavPath);

	if (FPackageName::DoesPackageExist(ExpectedPkg))
	{
		UE_LOG(LogEmotionBridge, Log,
			TEXT("FEmotionAudioAssetHelper: imported SoundWave → '%s'."), *ExpectedPkg);
		return ExpectedPkg;
	}

	// Import may have succeeded but the package name doesn't match expectations.
	// Check via the task's imported object list (if populated by the factory).
	if (ImportTask->ImportedObjectPaths.Num() > 0)
	{
		const FString FirstPath = ImportTask->ImportedObjectPaths[0];
		// Object path is "PackageName.AssetName" — strip after the dot.
		const FString PkgPath = FPackageName::ObjectPathToPackageName(FirstPath);
		if (!PkgPath.IsEmpty())
		{
			UE_LOG(LogEmotionBridge, Log,
				TEXT("FEmotionAudioAssetHelper: imported SoundWave (from task) → '%s'."), *PkgPath);
			return PkgPath;
		}
	}

	OutErrorMessage = FString::Printf(
		TEXT("Import pipeline ran but no SoundWave asset was created for '%s'. "
			 "Check Output Log → AssetTools for details."),
		*FPaths::GetCleanFilename(AbsoluteWavPath));
	UE_LOG(LogEmotionBridge, Error,
		TEXT("FEmotionAudioAssetHelper: %s"), *OutErrorMessage);
	return FString{};
}
