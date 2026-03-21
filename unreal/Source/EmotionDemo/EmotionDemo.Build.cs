// Copyright (c) EmotionDemo Project. All rights reserved.

using UnrealBuildTool;

public class EmotionDemo : ModuleRules
{
	public EmotionDemo(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"InputCore",
		});

		PrivateDependencyModuleNames.AddRange(new string[]
		{
			"EmotionBridge",
		});
	}
}
