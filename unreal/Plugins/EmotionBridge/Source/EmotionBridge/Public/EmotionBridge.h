// Copyright (c) EmotionDemo Project. All rights reserved.
#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

/**
 * FEmotionBridgeModule
 *
 * Runtime module entry point for the EmotionBridge plugin.
 * Registers the log category and plugin settings on startup.
 * No game-thread objects are owned here; per-actor/per-component logic lives
 * in AEmotionLampActor and UEmotionPlaybackComponent respectively.
 */
class EMOTIONBRIDGE_API FEmotionBridgeModule : public IModuleInterface
{
public:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;
};
