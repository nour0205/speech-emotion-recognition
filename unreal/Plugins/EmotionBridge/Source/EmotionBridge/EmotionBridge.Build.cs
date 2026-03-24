// Copyright (c) EmotionDemo Project. All rights reserved.

using UnrealBuildTool;

public class EmotionBridge : ModuleRules
{
	public EmotionBridge(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"HTTP",
			"Json",
			"JsonUtilities",
			"DeveloperSettings",
		});

		PrivateDependencyModuleNames.AddRange(new string[]
		{
			// nothing private yet
		});

		// Ensure private headers are findable within this module
		PrivateIncludePaths.AddRange(new string[]
		{
			"EmotionBridge/Private",
		});
	}
}
