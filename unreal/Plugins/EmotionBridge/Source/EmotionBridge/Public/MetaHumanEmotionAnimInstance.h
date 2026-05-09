// Copyright (c) EmotionDemo Project. All rights reserved.
// Phase 2B — Custom AnimInstance that injects CTRL_ animation curves into the
// MetaHuman PostProcess ABP every frame.
//
// The MetaHuman DNA solver reads curves named "CTRL_expressions.<control><Side>"
// (e.g. "CTRL_expressions.browDownL").  UMetaHumanEmotionDriverComponent writes
// target values into CurveValues each tick; NativeUpdateAnimation forwards them
// via AddCurveValue so the PostProcess ABP's RigLogic node can read them.
#pragma once

#include "CoreMinimal.h"
#include "Animation/AnimInstance.h"
#include "MetaHumanEmotionAnimInstance.generated.h"

UCLASS(meta=(DisplayName="MetaHuman Emotion AnimInstance"))
class EMOTIONBRIDGE_API UMetaHumanEmotionAnimInstance : public UAnimInstance
{
	GENERATED_BODY()

public:
	/**
	 * Animation curve values to inject each frame.
	 *
	 * Key   = exact DNA raw control name, e.g. FName("CTRL_expressions.browDownL").
	 * Value = target weight [0,1].
	 *
	 * Written by UMetaHumanEmotionDriverComponent::ApplyBlendStateToMesh()
	 * every tick.  NativeUpdateAnimation iterates this map and calls
	 * AddCurveValue for each entry so they become part of the animation pose
	 * that the PostProcess ABP's RigLogic / DNA solver reads.
	 */
	TMap<FName, float> CurveValues;

	virtual void NativeUpdateAnimation(float DeltaSeconds) override;
};
