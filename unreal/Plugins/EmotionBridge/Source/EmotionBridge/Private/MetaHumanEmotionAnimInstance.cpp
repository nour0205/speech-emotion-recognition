// Copyright (c) EmotionDemo Project. All rights reserved.
// Phase 2B — MetaHumanEmotionAnimInstance implementation.

#include "MetaHumanEmotionAnimInstance.h"
#include "EmotionBridgeLog.h"

void UMetaHumanEmotionAnimInstance::NativeUpdateAnimation(float DeltaSeconds)
{
	Super::NativeUpdateAnimation(DeltaSeconds);

	// First-call log — proves the anim graph is actually being evaluated.
	// If you never see this, the anim pipeline is skipped (usually because
	// bUpdateAnimationInEditor is false outside PIE) and no curve injection
	// will have any effect on the face.
	static bool bLoggedOnce = false;
	if (!bLoggedOnce)
	{
		bLoggedOnce = true;
		UE_LOG(LogEmotionBridge, Log,
			TEXT("MetaHumanEmotionAnimInstance: NativeUpdateAnimation FIRST CALL — "
			     "CurveValues.Num()=%d (if 0, driver hasn't pushed anything yet)."),
			CurveValues.Num());
	}

	// Throttled detailed dump (once / second) — shows the top non-zero curves
	// actually being injected into the pose this frame.  Useful to confirm the
	// driver → anim-instance handshake is live, and to see that values change
	// during emotion blends.
	static double LastCurveDumpTime = 0.0;
	const double Now = FPlatformTime::Seconds();
	if (Now - LastCurveDumpTime > 1.0 && CurveValues.Num() > 0)
	{
		LastCurveDumpTime = Now;

		int32 NonZero = 0;
		FName Top1, Top2, Top3;
		float W1 = 0.f, W2 = 0.f, W3 = 0.f;
		for (const auto& Pair : CurveValues)
		{
			if (Pair.Value > KINDA_SMALL_NUMBER) { ++NonZero; }
			if (Pair.Value > W1) { W3 = W2; Top3 = Top2; W2 = W1; Top2 = Top1; W1 = Pair.Value; Top1 = Pair.Key; }
			else if (Pair.Value > W2) { W3 = W2; Top3 = Top2; W2 = Pair.Value; Top2 = Pair.Key; }
			else if (Pair.Value > W3) { W3 = Pair.Value; Top3 = Pair.Key; }
		}

		UE_LOG(LogEmotionBridge, Log,
			TEXT("AnimInst  inject %d curves (%d non-zero). Top3: '%s'=%.3f, '%s'=%.3f, '%s'=%.3f"),
			CurveValues.Num(), NonZero,
			Top1.IsNone() ? TEXT("<none>") : *Top1.ToString(), W1,
			Top2.IsNone() ? TEXT("<none>") : *Top2.ToString(), W2,
			Top3.IsNone() ? TEXT("<none>") : *Top3.ToString(), W3);
	}

	for (const auto& Pair : CurveValues)
	{
		AddCurveValue(Pair.Key, Pair.Value);
	}
}
