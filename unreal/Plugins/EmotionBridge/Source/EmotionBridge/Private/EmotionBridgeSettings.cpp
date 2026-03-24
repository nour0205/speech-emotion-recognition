// Copyright (c) EmotionDemo Project. All rights reserved.

#include "EmotionBridgeSettings.h"
#include "EmotionBridgeLog.h"

UEmotionBridgeSettings::UEmotionBridgeSettings()
{
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
	const FLinearColor* Found = EmotionColors.Find(Emotion.ToLower());
	if (Found) return *Found;

	UE_LOG(LogEmotionBridge, Warning,
		TEXT("No color for emotion '%s' — using neutral white."), *Emotion);
	return FLinearColor::White;
}

FName UEmotionBridgeSettings::GetCategoryName() const { return FName(TEXT("Plugins")); }
FName UEmotionBridgeSettings::GetSectionName() const  { return FName(TEXT("Emotion Bridge")); }
