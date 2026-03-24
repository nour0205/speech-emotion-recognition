// Copyright (c) EmotionDemo Project. All rights reserved.
#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

/**
 * FEmotionBridgeEditorModule
 *
 * Editor-only module for the EmotionBridge plugin.
 *
 * Responsibilities:
 *   - Register the "Emotion Bridge" nomad tab spawner.
 *   - Add a "Emotion Bridge" entry under the Window menu.
 *   - Unregister everything on shutdown.
 */
class FEmotionBridgeEditorModule : public IModuleInterface
{
public:
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;

	/** Returns the singleton module instance. */
	static FEmotionBridgeEditorModule& Get()
	{
		return FModuleManager::GetModuleChecked<FEmotionBridgeEditorModule>("EmotionBridgeEditor");
	}

private:
	void RegisterMenus();
	TSharedRef<class SDockTab> SpawnTab(const class FSpawnTabArgs& Args);

	/** Tab identifier used by the global tab manager. */
	static const FName EmotionBridgeTabName;
};
