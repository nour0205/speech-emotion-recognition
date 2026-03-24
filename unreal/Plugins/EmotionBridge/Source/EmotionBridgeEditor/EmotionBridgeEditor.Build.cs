// Copyright (c) EmotionDemo Project. All rights reserved.

using UnrealBuildTool;

public class EmotionBridgeEditor : ModuleRules
{
	public EmotionBridgeEditor(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"Slate",
			"SlateCore",
			"UnrealEd",
			"ToolMenus",
			"WorkspaceMenuStructure",  // WorkspaceMenu::GetMenuStructure()
			"EmotionBridge",
		});

		PrivateDependencyModuleNames.AddRange(new string[]
		{
			"InputCore",
			"LevelEditor",
			"DesktopPlatform",
			"EditorStyle",
			"ApplicationCore",    // FSlateApplication
			"AudioCaptureCore",   // Audio::FAudioCapture implementation
			"AudioCapture",       // Audio::FAudioCapture — microphone recording
			"EditorFramework",    // FLevelEditorViewportClient (viewport focus)
		});

		PrivateIncludePaths.AddRange(new string[]
		{
			"EmotionBridgeEditor/Private",
		});
	}
}
