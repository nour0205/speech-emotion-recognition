// Copyright (c) EmotionDemo Project. All rights reserved.

#include "EmotionBridge.h"
#include "EmotionBridgeLog.h"

// Define the shared log category declared in EmotionBridgeLog.h
DEFINE_LOG_CATEGORY(LogEmotionBridge);

#define LOCTEXT_NAMESPACE "FEmotionBridgeModule"

void FEmotionBridgeModule::StartupModule()
{
	UE_LOG(LogEmotionBridge, Log, TEXT("EmotionBridge runtime module started."));
}

void FEmotionBridgeModule::ShutdownModule()
{
	UE_LOG(LogEmotionBridge, Log, TEXT("EmotionBridge runtime module shut down."));
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FEmotionBridgeModule, EmotionBridge)
