// Copyright (c) EmotionDemo Project. All rights reserved.
#pragma once

#include "CoreMinimal.h"
#include "Engine/DeveloperSettings.h"
#include "EmotionBridgeSettings.generated.h"

/**
 * Project settings for the EmotionBridge plugin.
 * Edit at: Edit > Project Settings > Plugins > Emotion Bridge
 * Persisted to Config/EmotionBridge.ini
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
	// Timeline Defaults (forwarded to POST /timeline/unreal)
	// -----------------------------------------------------------------------

	UPROPERTY(config, EditAnywhere, Category="Timeline Defaults", meta=(ClampMin="0.1", ClampMax="30.0"))
	float DefaultWindowSec = 2.0f;

	UPROPERTY(config, EditAnywhere, Category="Timeline Defaults", meta=(ClampMin="0.05", ClampMax="10.0"))
	float DefaultHopSec = 0.5f;

	/** "zero", "reflect", or "none". */
	UPROPERTY(config, EditAnywhere, Category="Timeline Defaults")
	FString DefaultPadMode = TEXT("none");

	/** "hysteresis", "majority", "ema", or "none". */
	UPROPERTY(config, EditAnywhere, Category="Timeline Defaults")
	FString DefaultSmoothingMethod = TEXT("none");

	UPROPERTY(config, EditAnywhere, Category="Timeline Defaults", meta=(ClampMin="1", ClampMax="20"))
	int32 DefaultHysteresisMinRun = 3;

	/** Used when SmoothingMethod == "majority". Must be odd. */
	UPROPERTY(config, EditAnywhere, Category="Timeline Defaults", meta=(ClampMin="1", ClampMax="21"))
	int32 DefaultMajorityWindow = 5;

	/** EMA alpha in (0,1]. Used when SmoothingMethod == "ema". */
	UPROPERTY(config, EditAnywhere, Category="Timeline Defaults", meta=(ClampMin="0.01", ClampMax="1.0"))
	float DefaultEmaAlpha = 0.6f;

	// -----------------------------------------------------------------------
	// Emotion Color Mapping
	// -----------------------------------------------------------------------

	/**
	 * Maps lowercase emotion labels to colors for the lamp actor.
	 * Add rows here to support future model emotion labels.
	 * Unknown labels fall back to neutral white.
	 */
	UPROPERTY(config, EditAnywhere, Category="Emotion Colors", meta=(DisplayName="Emotion → Color"))
	TMap<FString, FLinearColor> EmotionColors;

	// -----------------------------------------------------------------------
	// Helpers
	// -----------------------------------------------------------------------

	static UEmotionBridgeSettings* Get();

	/** Returns the color mapped to the given emotion (case-insensitive). Falls back to white. */
	FLinearColor GetColorForEmotion(const FString& Emotion) const;

	virtual FName GetCategoryName() const override;
	virtual FName GetSectionName() const override;
};
