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
			"AssetTools",         // Phase 2B: IAssetTools::ImportAssetTasks (SoundWave import)
			// IAnimationDataController + IAnimationDataModel headers live under
			// Engine/Animation/AnimData — no separate module required.
			"LevelSequence",       // M2: ULevelSequence (Initialize() lives here, no editor module needed)
			"MovieScene",          // M2: UMovieScene + binding/track APIs
			"MovieSceneTracks",    // M2: UMovieSceneAudioTrack, UMovieSceneSkeletalAnimationTrack
			"Sequencer",           // M4: opening sequencer UI for the bound sequence
			"PropertyEditor",      // M4: SObjectPropertyEntryBox for asset pickers
			// ── Phase B: drive MetaHuman Performance audio-to-face from C++ ──
			// All confirmed against MetaHumanPerformance.Build.cs in the engine
			// install (MetaHumanAnimator/Source/MetaHumanPerformance/).  The
			// MetaHumanPerformance public header pulls in all the *Pipeline*
			// and *CaptureData* + *CoreTech* modules transitively, so we have
			// to list them too.
			"MetaHumanPerformance",     // UMetaHumanPerformance + ExportUtils
			"MetaHumanCaptureData",     // CaptureData.h transitive include
			"MetaHumanPipelineCore",    // Pipeline/PipelineData.h
			"MetaHumanPipeline",        // Pipeline/Pipeline.h
			"MetaHumanCoreTech",        // FrameAnimationData
			"MetaHumanCoreTechLib",     // realtime smoothing/calibration
			"MetaHumanCoreEditor",      // editor-only public dep of Performance
			"CaptureDataEditor",        // editor-only public dep of Performance
		});

		PrivateIncludePaths.AddRange(new string[]
		{
			"EmotionBridgeEditor/Private",
		});
	}
}
