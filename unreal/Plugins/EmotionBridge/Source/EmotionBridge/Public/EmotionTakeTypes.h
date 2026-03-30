// Copyright (c) EmotionDemo Project. All rights reserved.
// Phase 2A — Take Library persistence types.
#pragma once

#include "CoreMinimal.h"
#include "EmotionTimelineTypes.h"

// ---------------------------------------------------------------------------
// FEmotionAnalysisParams
//
// All parameters sent to POST /timeline/unreal.
// Stored per-take so every result is fully reproducible.
// ---------------------------------------------------------------------------

struct EMOTIONBRIDGE_API FEmotionAnalysisParams
{
	float   WindowSec         = 2.0f;
	float   HopSec            = 0.5f;
	FString PadMode           = TEXT("none");
	FString SmoothingMethod   = TEXT("none");
	int32   HysteresisMinRun  = 3;
	int32   MajorityWindow    = 5;
	float   EmaAlpha          = 0.6f;
};

// ---------------------------------------------------------------------------
// FEmotionTakeSummary
//
// Derived statistics computed from a timeline — NOT stored on disk.
// Recomputed by FEmotionTakeStore::ComputeSummary() every time a take loads.
// ---------------------------------------------------------------------------

struct EMOTIONBRIDGE_API FEmotionTakeSummary
{
	/** Emotion with the greatest total on-screen duration. */
	FString DominantEmotion;

	/**
	 * Fraction of timeline covered by each emotion, summing to ~1.0.
	 * Key: lowercase emotion label (e.g. "happy"), Value: fraction [0,1].
	 */
	TMap<FString, float> EmotionDistribution;

	int32 SegmentCount      = 0;
	float TotalDurationSec  = 0.f;
	float AverageConfidence = 0.f;

	/** Clean filename extracted from SourceAudioPath (e.g. "speech_01.wav"). */
	FString SourceFilename;

	bool IsValid() const { return SegmentCount > 0; }
};

// ---------------------------------------------------------------------------
// FEmotionTakePhase2B
//
// Forward-compatible placeholder fields for the MetaHuman integration phase.
// All fields are empty strings until Phase 2B is implemented.
// None of these are populated or used in Phase 2A.
// ---------------------------------------------------------------------------

struct EMOTIONBRIDGE_API FEmotionTakePhase2B
{
	/** Path to a noise-cleaned copy of the source WAV (future: denoising pipeline). */
	FString CleanedAudioPath;

	/** Content-browser path to the imported UE SoundWave asset, e.g. "/Game/Sounds/Take01". */
	FString SoundWaveAssetPath;

	/**
	 * Content-browser path to the MetaHuman Performance asset.
	 * Populated in Phase 2B by the facial solve pipeline.
	 */
	FString MetaHumanPerformancePath;

	/** Content-browser path to the generated Level Sequence (facial + body animation). */
	FString LevelSequencePath;

	/**
	 * Path or asset ref to an emotion→morph-target preset mapping asset.
	 * Defines which facial blend shapes correspond to each emotion label.
	 */
	FString EmotionPresetMappingPath;
};

// ---------------------------------------------------------------------------
// FEmotionTakeRecord
//
// One persisted analysis unit — the central data type of Phase 2A.
//
// Disk layout (under <ProjectDir>/Saved/EmotionBridge/Takes/<TakeId>/):
//
//   metadata.json          — identity, timestamps, paths, tags, notes, phase2b
//   timeline.json          — full /timeline/unreal response
//   params.json            — analysis parameters used
//   audio/source.wav       — optional copied source audio
//
// All string paths are absolute except CopiedAudioPath, which is relative to
// the take folder root (e.g. "audio/source.wav").
// ---------------------------------------------------------------------------

struct EMOTIONBRIDGE_API FEmotionTakeRecord
{
	// -----------------------------------------------------------------------
	// Identity
	// -----------------------------------------------------------------------

	/** UUID-like identifier generated at save time (FGuid-based). Immutable after creation. */
	FString TakeId;

	/** User-visible label, e.g. "Actor_02_angry_take3". May be renamed. */
	FString DisplayName;

	// -----------------------------------------------------------------------
	// Timestamps  (ISO 8601 UTC — e.g. "2026-03-30T12:00:00.000Z")
	// -----------------------------------------------------------------------

	FString CreatedAt;
	FString UpdatedAt;

	// -----------------------------------------------------------------------
	// Audio paths
	// -----------------------------------------------------------------------

	/** Absolute path to the WAV at the time of analysis. May no longer exist. */
	FString SourceAudioPath;

	/**
	 * Path RELATIVE to the take folder root (e.g. "audio/source.wav").
	 * Empty if the audio was not copied at save time.
	 * Use GetCopiedAudioFullPath() to get the absolute path.
	 */
	FString CopiedAudioPath;

	// -----------------------------------------------------------------------
	// Audio metadata (from the backend response)
	// -----------------------------------------------------------------------

	float DurationSec = 0.f;
	int32 SampleRate  = 16000;

	// -----------------------------------------------------------------------
	// Analysis
	// -----------------------------------------------------------------------

	FEmotionAnalysisParams      Params;
	FEmotionTimelineResponse    Timeline;

	// -----------------------------------------------------------------------
	// User annotations
	// -----------------------------------------------------------------------

	FString         Notes;
	TArray<FString> Tags;

	// -----------------------------------------------------------------------
	// Schema versioning
	// -----------------------------------------------------------------------

	/**
	 * Incremented when the on-disk JSON layout changes in a breaking way.
	 * Readers should check this before loading unknown versions.
	 * Current version: 1
	 */
	int32   SchemaVersion = 1;
	FString PluginVersion = TEXT("1.0");

	// -----------------------------------------------------------------------
	// Phase 2B placeholders (empty in Phase 2A)
	// -----------------------------------------------------------------------

	FEmotionTakePhase2B Phase2B;

	// -----------------------------------------------------------------------
	// Computed fields — NOT stored on disk; filled by FEmotionTakeStore::LoadTake
	// -----------------------------------------------------------------------

	FEmotionTakeSummary Summary;

	// -----------------------------------------------------------------------
	// Path helpers
	// -----------------------------------------------------------------------

	/** Returns <ProjectDir>/Saved/EmotionBridge/Takes/<TakeId>. May not exist yet. */
	FString GetFolderPath() const
	{
		return FPaths::Combine(
			FPaths::ProjectSavedDir(),
			TEXT("EmotionBridge"), TEXT("Takes"), TakeId);
	}

	/**
	 * Returns the absolute path to the copied audio file.
	 * Does NOT verify the file exists — call FPaths::FileExists() if needed.
	 * Returns empty string if CopiedAudioPath is empty.
	 */
	FString GetCopiedAudioFullPath() const
	{
		if (CopiedAudioPath.IsEmpty()) return FString{};
		return FPaths::Combine(GetFolderPath(), CopiedAudioPath);
	}

	/**
	 * Returns the best available audio path for this take:
	 *   1. Copied audio inside the take folder (preferred — self-contained)
	 *   2. Original source audio path (fallback — may be gone)
	 *   3. Empty string if neither exists
	 */
	FString GetBestAudioPath() const
	{
		if (!CopiedAudioPath.IsEmpty())
		{
			const FString Copied = GetCopiedAudioFullPath();
			if (FPaths::FileExists(Copied)) return Copied;
		}
		if (!SourceAudioPath.IsEmpty() && FPaths::FileExists(SourceAudioPath))
			return SourceAudioPath;
		return FString{};
	}

	/** True when TakeId is set and Timeline parsed successfully. */
	bool IsValid() const { return !TakeId.IsEmpty() && Timeline.bIsValid; }
};
