// Copyright (c) EmotionDemo Project. All rights reserved.

using UnrealBuildTool;

public class EmotionDemoEditorTarget : TargetRules
{
	public EmotionDemoEditorTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Editor;
		DefaultBuildSettings = BuildSettingsVersion.V6;
		IncludeOrderVersion = EngineIncludeOrderVersion.Unreal5_7;
		ExtraModuleNames.Add("EmotionDemo");
	}
}
