// Copyright (c) EmotionDemo Project. All rights reserved.
#pragma once

#include "CoreMinimal.h"
#include "EmotionTimelineTypes.generated.h"

/**
 * One contiguous segment of a detected emotion returned by the /timeline endpoint.
 * Maps directly to a single object in the "segments" JSON array.
 */
USTRUCT(BlueprintType)
struct EMOTIONBRIDGE_API FEmotionSegment
{
	GENERATED_BODY()

	/** Start time of this segment in seconds from the beginning of the audio. */
	UPROPERTY(BlueprintReadOnly, Category="EmotionBridge")
	float StartSec = 0.f;

	/** End time of this segment in seconds. */
	UPROPERTY(BlueprintReadOnly, Category="EmotionBridge")
	float EndSec = 0.f;

	/** Emotion label, e.g. "angry", "happy", "sad", "neutral". */
	UPROPERTY(BlueprintReadOnly, Category="EmotionBridge")
	FString Emotion;

	/** Model confidence in [0,1]. */
	UPROPERTY(BlueprintReadOnly, Category="EmotionBridge")
	float Confidence = 0.f;
};

/**
 * Full response from the /timeline endpoint.
 * bIsValid is false if parsing failed or the request failed; check ErrorMessage in that case.
 */
USTRUCT(BlueprintType)
struct EMOTIONBRIDGE_API FEmotionTimelineResponse
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category="EmotionBridge")
	FString ModelName;

	UPROPERTY(BlueprintReadOnly, Category="EmotionBridge")
	int32 SampleRate = 0;

	UPROPERTY(BlueprintReadOnly, Category="EmotionBridge")
	float DurationSec = 0.f;

	UPROPERTY(BlueprintReadOnly, Category="EmotionBridge")
	float WindowSec = 2.0f;

	UPROPERTY(BlueprintReadOnly, Category="EmotionBridge")
	float HopSec = 0.5f;

	UPROPERTY(BlueprintReadOnly, Category="EmotionBridge")
	TArray<FEmotionSegment> Segments;

	/** True when the HTTP request succeeded and JSON was parsed without errors. */
	bool bIsValid = false;

	/** Human-readable error description when bIsValid == false. */
	FString ErrorMessage;
};
