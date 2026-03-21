// Copyright (c) EmotionDemo Project. All rights reserved.

#include "EmotionBridgeSettings.h"
#include "EmotionBridgeLog.h"

UEmotionBridgeSettings::UEmotionBridgeSettings()
{
	// Default color mapping for the four baseline emotions.
	// These are stored in Config/EmotionBridge.ini and can be overridden
	// in Edit > Project Settings > Plugins > Emotion Bridge.
	EmotionColors.Add(TEXT("angry"),   FLinearColor(1.0f, 0.08f, 0.08f, 1.0f)); // red
	EmotionColors.Add(TEXT("happy"),   FLinearColor(1.0f, 0.90f, 0.05f, 1.0f)); // yellow
	EmotionColors.Add(TEXT("sad"),     FLinearColor(0.1f, 0.25f, 1.0f,  1.0f)); // blue
	EmotionColors.Add(TEXT("neutral"), FLinearColor(1.0f, 1.0f,  1.0f,  1.0f)); // white
}

UEmotionBridgeSettings* UEmotionBridgeSettings::Get()
{
	return GetMutableDefault<UEmotionBridgeSettings>();
}

FLinearColor UEmotionBridgeSettings::GetColorForEmotion(const FString& Emotion) const
{
	const FString Key = Emotion.ToLower();
	const FLinearColor* Found = EmotionColors.Find(Key);
	if (Found)
	{
		return *Found;
	}

	// Unknown label: warn once and fall back to neutral white.
	UE_LOG(LogEmotionBridge, Warning,
		TEXT("EmotionBridgeSettings: no color mapping for emotion '%s' — using neutral white."), *Emotion);
	return FLinearColor::White;
}

FName UEmotionBridgeSettings::GetCategoryName() const
{
	return FName(TEXT("Plugins"));
}

FName UEmotionBridgeSettings::GetSectionName() const
{
	return FName(TEXT("Emotion Bridge"));
}
