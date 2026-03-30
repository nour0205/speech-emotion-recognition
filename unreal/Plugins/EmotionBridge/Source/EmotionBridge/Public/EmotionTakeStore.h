// Copyright (c) EmotionDemo Project. All rights reserved.
// Phase 2A — Take Library persistence service.
#pragma once

#include "CoreMinimal.h"
#include "EmotionTakeTypes.h"

/**
 * FEmotionTakeStore
 *
 * Static service layer for all take persistence.
 * Every disk read/write goes through this class — no other code touches the
 * Saved/EmotionBridge/Takes/ directory directly.
 *
 * Storage root:
 *   <ProjectDir>/Saved/EmotionBridge/Takes/
 *
 * Per-take directory layout:
 *   <TakeId>/
 *     metadata.json          — identity, timestamps, paths, tags, notes, phase2b
 *     timeline.json          — full serialised FEmotionTimelineResponse
 *     params.json            — FEmotionAnalysisParams used for this run
 *     audio/
 *       source.wav           — optional copy of the source WAV
 *
 * Write safety:
 *   Each JSON file is written to a <filename>.tmp file first, then renamed
 *   to the final path.  A partially-written tmp file left by a crash is
 *   harmless and will be overwritten on the next save.
 *
 * Reanalysis policy (Phase 2A design decision):
 *   Reanalysis OVERWRITES the existing take in-place.
 *   The take ID, display name, notes, and tags are preserved.
 *   The timeline, params, and UpdatedAt timestamp are replaced.
 *   To keep the old result, use DuplicateTake() before reanalysing.
 *
 * Thread safety:
 *   Currently single-threaded (editor-only use).
 *   Do not call from a background thread without adding a mutex.
 */
class EMOTIONBRIDGE_API FEmotionTakeStore
{
public:
	// -----------------------------------------------------------------------
	// Path helpers
	// -----------------------------------------------------------------------

	/** Returns <ProjectDir>/Saved/EmotionBridge/Takes/  (no trailing slash). */
	static FString GetTakesRootDirectory();

	/**
	 * Returns the absolute path to a specific take folder.
	 * The folder may or may not exist on disk.
	 */
	static FString GetTakeFolderPath(const FString& TakeId);

	// -----------------------------------------------------------------------
	// CRUD
	// -----------------------------------------------------------------------

	/**
	 * Persist a take to disk.
	 *
	 * @param InOutRecord   The take to save.  CopiedAudioPath is updated in-place
	 *                      when bCopyAudio=true and the copy succeeds.
	 * @param bCopyAudio    When true AND SourceAudioPath points to an existing file,
	 *                      the WAV is copied into <TakeId>/audio/source.wav.
	 * @return              True on full success (all three JSON files written).
	 *                      Audio copy failure is logged as a warning but does NOT
	 *                      cause a false return.
	 */
	static bool SaveTake(FEmotionTakeRecord& InOutRecord, bool bCopyAudio = true);

	/**
	 * Load a single take by TakeId.
	 *
	 * Reads metadata.json, timeline.json, and params.json from the take folder.
	 * Recomputes Summary from the loaded timeline.
	 *
	 * @return  False if the folder does not exist, any required JSON file is
	 *          missing or malformed, or take_id inside metadata.json is empty.
	 */
	static bool LoadTake(const FString& TakeId, FEmotionTakeRecord& OutRecord);

	/**
	 * Enumerate and load all takes from the takes root directory.
	 *
	 * Skips (and logs) any take that fails to load.  Partial loads are safe —
	 * a corrupted take does not prevent other takes from loading.
	 *
	 * @return  Number of successfully loaded takes appended to OutTakes.
	 */
	static int32 LoadAllTakes(TArray<FEmotionTakeRecord>& OutTakes);

	/**
	 * Delete a take folder and all its contents from disk.
	 *
	 * @return  True if the folder was deleted or did not exist in the first place.
	 */
	static bool DeleteTake(const FString& TakeId);

	/**
	 * Duplicate a take under a new display name (and new auto-generated TakeId).
	 *
	 * Copies the entire take folder tree.  UpdatedAt is reset to now.
	 * CreatedAt is copied from the source so the origin is traceable.
	 *
	 * @param SourceTakeId      TakeId of the existing take to copy.
	 * @param NewDisplayName    Display name for the duplicate.
	 * @param OutNewRecord      Filled with the newly created take on success.
	 * @return                  True on success.
	 */
	static bool DuplicateTake(
		const FString& SourceTakeId,
		const FString& NewDisplayName,
		FEmotionTakeRecord& OutNewRecord);

	// -----------------------------------------------------------------------
	// Helpers
	// -----------------------------------------------------------------------

	/** Generate a unique TakeId string (GUIDv4-derived, 32 hex chars). */
	static FString GenerateTakeId();

	/**
	 * Generate a default display name based on the current UTC timestamp,
	 * e.g. "Take_20260330_120000".
	 */
	static FString GenerateDefaultDisplayName();

	/**
	 * Compute summary statistics from a timeline response.
	 * This is a pure function — it does not touch the disk.
	 *
	 * @param Timeline      The parsed timeline response.
	 * @param SourcePath    Used only to derive SourceFilename in the summary.
	 */
	static FEmotionTakeSummary ComputeSummary(
		const FEmotionTimelineResponse& Timeline,
		const FString& SourcePath);

	/** True if a take folder with this TakeId exists on disk. */
	static bool TakeExists(const FString& TakeId);

	/**
	 * Check whether a display name is already in use.
	 * Non-blocking advisory check — does not enforce uniqueness at save time.
	 * Useful for showing a warning in the UI before the user clicks Save.
	 */
	static bool IsDisplayNameUnique(
		const FString& DisplayName,
		const TArray<FEmotionTakeRecord>& ExistingTakes);

private:
	// -----------------------------------------------------------------------
	// Serialisation (private — call SaveTake / LoadTake instead)
	// -----------------------------------------------------------------------

	static bool WriteMetadataJson(const FEmotionTakeRecord& Record,      const FString& FolderPath);
	static bool WriteTimelineJson(const FEmotionTimelineResponse& TL,    const FString& FolderPath);
	static bool WriteParamsJson  (const FEmotionAnalysisParams& Params,  const FString& FolderPath);

	static bool ReadMetadataJson(const FString& FilePath, FEmotionTakeRecord& OutRecord);
	static bool ReadTimelineJson(const FString& FilePath, FEmotionTimelineResponse& OutTimeline);
	static bool ReadParamsJson  (const FString& FilePath, FEmotionAnalysisParams& OutParams);

	/**
	 * Write Content to FilePath using a temp-then-rename strategy.
	 * Writes to <FilePath>.tmp first; renames to <FilePath> only on success.
	 * On failure, the .tmp file is deleted.
	 */
	static bool SafeWriteStringToFile(const FString& FilePath, const FString& Content);
};
