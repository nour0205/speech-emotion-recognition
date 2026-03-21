// Copyright (c) EmotionDemo Project. All rights reserved.

#include "EmotionBridgeEditor.h"
#include "EmotionBridgeLog.h"
#include "SEmotionBridgePanel.h"

#include "Framework/Docking/TabManager.h"
#include "Widgets/Docking/SDockTab.h"
#include "ToolMenus.h"
#include "WorkspaceMenuStructure.h"
#include "WorkspaceMenuStructureModule.h"

#define LOCTEXT_NAMESPACE "FEmotionBridgeEditorModule"

const FName FEmotionBridgeEditorModule::EmotionBridgeTabName = FName("EmotionBridgeTab");

// ---------------------------------------------------------------------------
// Startup / Shutdown
// ---------------------------------------------------------------------------

void FEmotionBridgeEditorModule::StartupModule()
{
	UE_LOG(LogEmotionBridge, Log, TEXT("EmotionBridgeEditor module started."));

	// Register the tab spawner. "Nomad" means it can float anywhere in the editor.
	FGlobalTabmanager::Get()->RegisterNomadTabSpawner(
		EmotionBridgeTabName,
		FOnSpawnTab::CreateRaw(this, &FEmotionBridgeEditorModule::SpawnTab))
		.SetDisplayName(LOCTEXT("TabTitle",   "Emotion Bridge"))
		.SetTooltipText(LOCTEXT("TabTooltip", "Analyze speech emotion timelines and drive actor color changes."))
		.SetGroup(WorkspaceMenu::GetMenuStructure().GetToolsCategory())
		.SetMenuType(ETabSpawnerMenuType::Enabled);

	// Defer menu registration until ToolMenus are ready.
	UToolMenus::RegisterStartupCallback(
		FSimpleMulticastDelegate::FDelegate::CreateRaw(
			this, &FEmotionBridgeEditorModule::RegisterMenus));
}

void FEmotionBridgeEditorModule::ShutdownModule()
{
	UToolMenus::UnRegisterStartupCallback(this);
	UToolMenus::UnregisterOwner(this);

	FGlobalTabmanager::Get()->UnregisterNomadTabSpawner(EmotionBridgeTabName);

	UE_LOG(LogEmotionBridge, Log, TEXT("EmotionBridgeEditor module shut down."));
}

// ---------------------------------------------------------------------------
// Menu entry
// ---------------------------------------------------------------------------

void FEmotionBridgeEditorModule::RegisterMenus()
{
	// Add "Emotion Bridge" under Window > ... in the main editor menu bar.
	FToolMenuOwnerScoped OwnerScoped(this);
	UToolMenu* Menu = UToolMenus::Get()->ExtendMenu("LevelEditor.MainMenu.Window");
	FToolMenuSection& Section = Menu->FindOrAddSection("WindowLayout");
	Section.AddMenuEntry(
		"EmotionBridgeEntry",
		LOCTEXT("MenuLabel",   "Emotion Bridge"),
		LOCTEXT("MenuTooltip", "Open the Emotion Bridge analysis and demo panel."),
		FSlateIcon(),
		FUIAction(FExecuteAction::CreateLambda([]()
		{
			FGlobalTabmanager::Get()->TryInvokeTab(FName("EmotionBridgeTab"));
		}))
	);
}

// ---------------------------------------------------------------------------
// Tab factory
// ---------------------------------------------------------------------------

TSharedRef<SDockTab> FEmotionBridgeEditorModule::SpawnTab(const FSpawnTabArgs& Args)
{
	return SNew(SDockTab)
		.TabRole(ETabRole::NomadTab)
		[
			SNew(SEmotionBridgePanel)
		];
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FEmotionBridgeEditorModule, EmotionBridgeEditor)
