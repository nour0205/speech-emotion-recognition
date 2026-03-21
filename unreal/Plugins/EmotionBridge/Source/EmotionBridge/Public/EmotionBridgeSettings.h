// Copyright (c) EmotionDemo Project. All rights reserved.
#pragma once

#include "CoreMinimal.h"
#include "Engine/DeveloperSettings.h"
#include "EmotionBridgeSettings.generated.h"

/**
 * Project settings for the EmotionBridge plugin.
 * Accessible via Edit > Project Settings > Plugins > Emotion Bridge.
 * Persisted to Config/EmotionBridge.ini.
 */
UCLASS(config=EmotionBridge, defaultconfig, meta=(DisplayName="Emotion Bridge"))
class EMOTIONBRIDGE_API UEmotionBridgeSettings : public UDeveloperSettings
{
	GENERATED_BODY()

public:
	UEmotionBridgeSettings();

	// -----------------------------------------------------------------------
	// Connection
	// -----------------------------------------------------------------------

	/** Base URL of the local speech emotion recognition backend. */
	UPROPERTY(config, EditAnywhere, Category="Connection", meta=(DisplayName="API Base URL"))
	FString ApiBaseUrl = TEXT("http://localhost:8000");

	// -----------------------------------------------------------------------
	// Timeline Defaults
	// -----------------------------------------------------------------------

	/** Default window size in seconds for the /timeline request. */
	UPROPERTY(config, EditAnywhere, Category="Timeline Defaults", meta=(ClampMin="0.1", ClampMax="30.0"))
	float DefaultWindowSec = 2.0f;

	/** Default hop size in seconds. */
	UPROPERTY(config, EditAnywhere, Category="Timeline Defaults", meta=(ClampMin="0.1", ClampMax="10.0"))
	float DefaultHopSec = 0.5f;

	/** Default padding mode ("zero", "reflect", or "edge"). */
	UPROPERTY(config, EditAnywhere, Category="Timeline Defaults")
	FString DefaultPadMode = TEXT("zero");

	/** Default smoothing method ("hysteresis" or "none"). */
	UPROPERTY(config, EditAnywhere, Category="Timeline Defaults")
	FString DefaultSmoothingMethod = TEXT("hysteresis");

	/** Minimum run length for hysteresis smoothing. */
	UPROPERTY(config, EditAnywhere, Category="Timeline Defaults", meta=(ClampMin="1", ClampMax="20"))
	int32 DefaultHysteresisMinRun = 3;

	// -----------------------------------------------------------------------
	// Emotion Color Mapping
	// -----------------------------------------------------------------------

	/**
	 * Maps lowercase emotion labels to linear colors for the lamp actor.
	 * Add custom rows here to support additional emotion labels from future models.
	 * Unknown labels fall back to neutral (white).
	 */
	UPROPERTY(config, EditAnywhere, Category="Emotion Colors", meta=(DisplayName="Emotion → Color"))
	TMap<FString, FLinearColor> EmotionColors;

	// -----------------------------------------------------------------------
	// Helpers
	// -----------------------------------------------------------------------

	/** Returns the singleton settings object (never null in editor). */
	static UEmotionBridgeSettings* Get();

	/** Looks up a color for the given emotion label (case-insensitive). Returns White if not found. */
	FLinearColor GetColorForEmotion(const FString& Emotion) const;

	// UDeveloperSettings interface
	virtual FName GetCategoryName() const override;
	virtual FName GetSectionName() const override;
};
