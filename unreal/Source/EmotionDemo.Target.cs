// Copyright (c) EmotionDemo Project. All rights reserved.

using UnrealBuildTool;

public class EmotionDemoTarget : TargetRules
{
	public EmotionDemoTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Game;
		DefaultBuildSettings = BuildSettingsVersion.V6;
		IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_7;
		ExtraModuleNames.Add("EmotionDemo");
	}
}
