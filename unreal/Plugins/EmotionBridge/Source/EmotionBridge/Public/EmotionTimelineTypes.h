// Copyright (c) EmotionDemo Project. All rights reserved.
#pragma once

#include "CoreMinimal.h"
#include "EmotionTimelineTypes.generated.h"

/**
 * One contiguous segment of a detected emotion.
 * Matches the UnrealSegmentSchema returned by POST /timeline/unreal.
 */
USTRUCT(BlueprintType)
struct EMOTIONBRIDGE_API FEmotionSegment
{
	GENERATED_BODY()

	/** Start time in seconds from the beginning of the audio. */
	UPROPERTY(BlueprintReadOnly, Category="EmotionBridge")
	float StartSec = 0.f;

	/** End time in seconds. */
	UPROPERTY(BlueprintReadOnly, Category="EmotionBridge")
	float EndSec = 0.f;

	/** Emotion label: "angry", "happy", "sad", or "neutral". */
	UPROPERTY(BlueprintReadOnly, Category="EmotionBridge")
	FString Emotion;

	/** Model confidence in [0,1]. */
	UPROPERTY(BlueprintReadOnly, Category="EmotionBridge")
	float Confidence = 0.f;
};

/**
 * Parsed response from POST /timeline/unreal.
 *
 * Envelope:
 *   { "type": "timeline", "source": "ser_api", "version": "1.0",
 *     "model_name": "speechbrain-iemocap", "sample_rate": 16000,
 *     "duration_sec": 10.5, "segments": [ {...}, ... ] }
 *
 * bIsValid is false when the request failed or JSON was malformed;
 * inspect ErrorMessage for a human-readable reason.
 *
 * Phase 2A additions: ModelName and SampleRate are stored in take records
 * so takes are fully self-describing without re-calling the backend.
 */
USTRUCT(BlueprintType)
struct EMOTIONBRIDGE_API FEmotionTimelineResponse
{
	GENERATED_BODY()

	/** Always "timeline" when valid. */
	UPROPERTY(BlueprintReadOnly, Category="EmotionBridge")
	FString Type;

	/** Always "ser_api" when valid. */
	UPROPERTY(BlueprintReadOnly, Category="EmotionBridge")
	FString Source;

	/** Schema version string, e.g. "1.0". */
	UPROPERTY(BlueprintReadOnly, Category="EmotionBridge")
	FString Version;

	/**
	 * Model identifier returned by the backend, e.g. "speechbrain-iemocap".
	 * Populated by Phase 2A — stored in take records for provenance tracking.
	 */
	UPROPERTY(BlueprintReadOnly, Category="EmotionBridge")
	FString ModelName;

	/**
	 * Sample rate of the audio file as reported by the backend (Hz).
	 * Typically 16000.  Stored in take records for future audio pipeline use.
	 */
	UPROPERTY(BlueprintReadOnly, Category="EmotionBridge")
	int32 SampleRate = 16000;

	/** Total audio duration in seconds. */
	UPROPERTY(BlueprintReadOnly, Category="EmotionBridge")
	float DurationSec = 0.f;

	/** All detected emotion segments in chronological order. */
	UPROPERTY(BlueprintReadOnly, Category="EmotionBridge")
	TArray<FEmotionSegment> Segments;

	/** True when the HTTP request succeeded and JSON parsed cleanly. */
	bool bIsValid = false;

	/** Human-readable error description when bIsValid == false. */
	FString ErrorMessage;
};
