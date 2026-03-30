// Copyright (c) EmotionDemo Project. All rights reserved.
// Phase 2B — Editor helper for importing WAV files as SoundWave content assets.
//
// This is an editor-only helper (lives in EmotionBridgeEditor module).
// All methods are static — no instantiation required.
#pragma once

#include "CoreMinimal.h"

/**
 * FEmotionAudioAssetHelper
 *
 * Wraps the UE5 asset import pipeline to import an absolute WAV path as a
 * USoundWave asset in the Unreal content browser.
 *
 * WHY THIS EXISTS
 *   The MetaHuman audio-driven facial animation workflow requires the audio to
 *   exist as a content-browser SoundWave asset.  This helper automates that
 *   step so the Emotion Bridge panel can offer a one-click "Import WAV" button
 *   instead of requiring the user to drag-and-drop into the browser manually.
 *
 * LIMITATIONS
 *   - Only .wav files are supported (SoundWave import factory accepts WAV).
 *   - Import is synchronous and suppresses dialogs (bAutomated = true).
 *   - The resulting asset is automatically saved to disk.
 *   - If an asset already exists at the destination path, it is returned
 *     without re-importing.
 */
struct FEmotionAudioAssetHelper
{
	/**
	 * Import a WAV file as a SoundWave content asset.
	 *
	 * @param AbsoluteWavPath      Absolute filesystem path to the source .wav file.
	 * @param ContentDestPath      Content-browser destination path in /Game/… form.
	 *                             Example: "/Game/EmotionBridge/Audio"
	 *                             The asset name is derived from the WAV filename.
	 * @param OutErrorMessage      Human-readable error description when returning empty.
	 * @return                     Content-browser asset package path on success,
	 *                             e.g. "/Game/EmotionBridge/Audio/MySpeech".
	 *                             Empty string on failure.
	 */
	static FString ImportWavAsSoundWave(
		const FString& AbsoluteWavPath,
		const FString& ContentDestPath,
		FString& OutErrorMessage);

	/**
	 * Returns the default content-browser destination for EmotionBridge audio.
	 * Value: "/Game/EmotionBridge/Audio"
	 */
	static FString GetDefaultAudioContentPath();

	/**
	 * Check whether a SoundWave asset already exists in the content browser for
	 * the given WAV file and destination path.
	 *
	 * @param AbsoluteWavPath  Filesystem path; the base filename is used as the asset name.
	 * @param ContentDestPath  Content-browser folder, e.g. "/Game/EmotionBridge/Audio".
	 * @return                 Asset package path if it exists; empty string otherwise.
	 */
	static FString FindExistingSoundWave(
		const FString& AbsoluteWavPath,
		const FString& ContentDestPath);
};
