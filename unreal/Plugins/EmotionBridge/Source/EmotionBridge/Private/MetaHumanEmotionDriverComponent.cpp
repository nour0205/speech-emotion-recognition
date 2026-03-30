// Copyright (c) EmotionDemo Project. All rights reserved.
// Phase 2B — UMetaHumanEmotionDriverComponent implementation.

#include "MetaHumanEmotionDriverComponent.h"
#include "EmotionBridgeLog.h"

#include "Components/SkeletalMeshComponent.h"
#include "GameFramework/Actor.h"

// ===========================================================================
// Constructor
// ===========================================================================

UMetaHumanEmotionDriverComponent::UMetaHumanEmotionDriverComponent()
{
	PrimaryComponentTick.bCanEverTick      = true;
	PrimaryComponentTick.bStartWithTickEnabled = false; // enabled on demand
	bTickInEditor = true;   // preview in editor viewport without PIE

	// Default overlay settings are set in FEmotionOverlaySettings constructor.
}

// ===========================================================================
// UActorComponent overrides
// ===========================================================================

void UMetaHumanEmotionDriverComponent::BeginPlay()
{
	Super::BeginPlay();
	EnsurePresetsInitialized();
	ResolveFaceMesh(); // warm the cache early
}

void UMetaHumanEmotionDriverComponent::TickComponent(
	float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	EnsurePresetsInitialized();

	// ---- Autonomous playback ---------------------------------------------------
	if (bIsAutonomousPlaying)
	{
		AutonomousElapsedSec += DeltaTime;

		if (AutonomousElapsedSec >= AutonomousTimeline.DurationSec)
		{
			bIsAutonomousPlaying = false;
			ResetToNeutral();
			SetComponentTickEnabled(false);
			UE_LOG(LogEmotionBridge, Log,
				TEXT("MetaHumanEmotionDriver: autonomous playback finished."));
			return;
		}

		// Find active segment.
		int32 ActiveIndex = -1;
		for (int32 i = 0; i < AutonomousTimeline.Segments.Num(); ++i)
		{
			const FEmotionSegment& Seg = AutonomousTimeline.Segments[i];
			if (AutonomousElapsedSec >= Seg.StartSec && AutonomousElapsedSec < Seg.EndSec)
			{
				ActiveIndex = i;
				break;
			}
		}

		if (ActiveIndex != LastAutonomousSegmentIndex)
		{
			LastAutonomousSegmentIndex = ActiveIndex;
			if (ActiveIndex >= 0)
			{
				const FEmotionSegment& Seg = AutonomousTimeline.Segments[ActiveIndex];
				ApplyEmotion(Seg.Emotion, Seg.Confidence);
			}
			else
			{
				ApplyEmotion(TEXT("neutral"), 1.0f);
			}
		}
	}

	// ---- Advance blend --------------------------------------------------------
	TickBlend(DeltaTime);

	// ---- Apply to mesh --------------------------------------------------------
	ApplyBlendStateToMesh();
}

// ===========================================================================
// External drive API
// ===========================================================================

void UMetaHumanEmotionDriverComponent::ApplyEmotion(
	const FString& InEmotion, float InConfidence)
{
	const FString NormEmotion = InEmotion.ToLower().TrimStartAndEnd();

	// If already transitioning to this same emotion, just update confidence.
	if (NormEmotion == BlendState.ToEmotion.ToLower())
	{
		BlendState.ToConfidence = FMath::Clamp(InConfidence, 0.f, 1.f);
		return;
	}

	// Compute the current "from" state.  If a blend is in progress, we snapshot
	// the mid-blend state as a synthetic neutral so the FROM position is accurate.
	// Since we store FromEmotion / ToEmotion (not per-target weights), we simply
	// let the ComputeBlendedWeight function handle the soft snapshot by keeping
	// the current alpha embedded in the new From start.
	//
	// Implementation note: we start the new blend from the FULLY SETTLED
	// FromEmotion but at whatever alpha we're at.  For smooth appearance, the
	// blend duration is short enough that an instantaneous From-reset is not
	// visible in practice.
	BlendState.FromEmotion  = BlendState.ToEmotion;   // snap from = previous target
	BlendState.ToEmotion    = NormEmotion;
	BlendState.BlendAlpha   = 0.f;
	BlendState.ToConfidence = FMath::Clamp(InConfidence, 0.f, 1.f);

	// Enable ticking (safe to call repeatedly).
	SetComponentTickEnabled(true);
}

void UMetaHumanEmotionDriverComponent::ResetToNeutral()
{
	BlendState.FromEmotion  = TEXT("neutral");
	BlendState.ToEmotion    = TEXT("neutral");
	BlendState.BlendAlpha   = 1.f;
	BlendState.ToConfidence = 1.f;

	// Zero all driven morph targets immediately.
	if (USkeletalMeshComponent* FaceMesh = ResolveFaceMesh())
	{
		for (const FName& TargetName : AllDrivenMorphTargets)
		{
			FaceMesh->SetMorphTarget(TargetName, 0.f, /*bRemoveZeroWeight=*/true);
		}
	}

	SetComponentTickEnabled(false);
}

// ===========================================================================
// Autonomous playback API
// ===========================================================================

void UMetaHumanEmotionDriverComponent::SetTimeline(const FEmotionTimelineResponse& InTimeline)
{
	AutonomousTimeline             = InTimeline;
	AutonomousElapsedSec           = 0.f;
	LastAutonomousSegmentIndex     = -1;
	bIsAutonomousPlaying           = false;
}

void UMetaHumanEmotionDriverComponent::StartPlayback()
{
	if (!AutonomousTimeline.bIsValid || AutonomousTimeline.Segments.IsEmpty())
	{
		UE_LOG(LogEmotionBridge, Warning,
			TEXT("MetaHumanEmotionDriver::StartPlayback — timeline is empty or invalid."));
		return;
	}
	AutonomousElapsedSec       = 0.f;
	LastAutonomousSegmentIndex = -1;
	bIsAutonomousPlaying       = true;
	SetComponentTickEnabled(true);
	UE_LOG(LogEmotionBridge, Log,
		TEXT("MetaHumanEmotionDriver: autonomous playback started (%.2f s, %d segments)."),
		AutonomousTimeline.DurationSec, AutonomousTimeline.Segments.Num());
}

void UMetaHumanEmotionDriverComponent::StopPlayback()
{
	bIsAutonomousPlaying = false;
	ResetToNeutral();
}

// ===========================================================================
// Direct mesh binding
// ===========================================================================

void UMetaHumanEmotionDriverComponent::SetFaceMeshComponent(USkeletalMeshComponent* InMesh)
{
	CachedFaceMesh = InMesh;
	if (InMesh)
	{
		UE_LOG(LogEmotionBridge, Log,
			TEXT("MetaHumanEmotionDriver: face mesh explicitly set to '%s'."),
			*InMesh->GetName());
	}
}

// ===========================================================================
// Preset management
// ===========================================================================

void UMetaHumanEmotionDriverComponent::SetPresets(const TArray<FEmotionExpressionPreset>& InPresets)
{
	ExpressionPresets    = InPresets;
	bPresetsInitialized  = true; // user provided — do not overwrite with defaults
	RebuildMorphTargetSet();
	UE_LOG(LogEmotionBridge, Log,
		TEXT("MetaHumanEmotionDriver: %d expression presets loaded."), ExpressionPresets.Num());
}

void UMetaHumanEmotionDriverComponent::SetPresetForEmotion(const FEmotionExpressionPreset& InPreset)
{
	const FString NormName = InPreset.EmotionName.ToLower();
	for (FEmotionExpressionPreset& EP : ExpressionPresets)
	{
		if (EP.EmotionName.ToLower() == NormName)
		{
			EP = InPreset;
			RebuildMorphTargetSet();
			return;
		}
	}
	// Not found — add it.
	ExpressionPresets.Add(InPreset);
	RebuildMorphTargetSet();
}

// ===========================================================================
// Default presets
// ===========================================================================

TArray<FEmotionExpressionPreset> UMetaHumanEmotionDriverComponent::MakeDefaultPresets()
{
	// These presets use ARKit 52 blendshape names as exported by MetaHuman when
	// the character is configured for ARKit face capture in UE5.
	//
	// SPEECH SAFETY: brow, cheek, eye, and nose targets are safe.
	// Mouth corner targets (mouthSmile/mouthFrown) are included at LOW weights
	// (< 0.5) so they add mood without dominating phoneme shapes.
	//
	// If SetMorphTarget() calls are silently ignored, your MetaHuman's face mesh
	// does not have these morph targets.  Use Details → Morph Target Preview on
	// the face mesh to find your actual target names and call SetPresets() with
	// corrected names.

	auto MakePreset = [](FString Name, TArray<TPair<FName, float>> Weights,
		float Intensity = 1.f) -> FEmotionExpressionPreset
	{
		FEmotionExpressionPreset P;
		P.EmotionName   = MoveTemp(Name);
		P.BaseIntensity = Intensity;
		for (auto& KV : Weights)
		{
			FEmotionMorphWeight MW;
			MW.MorphTargetName = KV.Key;
			MW.Weight          = KV.Value;
			P.MorphWeights.Add(MW);
		}
		return P;
	};

	TArray<FEmotionExpressionPreset> Presets;

	// ─── ANGRY ───────────────────────────────────────────────────────────────
	// Furrowed brows, eye squint, nose wrinkle.  No jaw / no lips.
	Presets.Add(MakePreset(TEXT("angry"), {
		{ TEXT("browDown_L"),    0.65f },
		{ TEXT("browDown_R"),    0.65f },
		{ TEXT("eyeSquint_L"),   0.35f },
		{ TEXT("eyeSquint_R"),   0.35f },
		{ TEXT("noseSneer_L"),   0.30f },
		{ TEXT("noseSneer_R"),   0.30f },
	}, 1.0f));

	// ─── HAPPY ───────────────────────────────────────────────────────────────
	// Raised cheeks, smile squint, light brow lift.  Mouth corners subtle.
	Presets.Add(MakePreset(TEXT("happy"), {
		{ TEXT("cheekSquint_L"), 0.55f },
		{ TEXT("cheekSquint_R"), 0.55f },
		{ TEXT("eyeSquint_L"),   0.30f },
		{ TEXT("eyeSquint_R"),   0.30f },
		{ TEXT("mouthSmile_L"),  0.40f },
		{ TEXT("mouthSmile_R"),  0.40f },
		{ TEXT("browInnerUp"),   0.10f },
	}, 1.0f));

	// ─── SAD ─────────────────────────────────────────────────────────────────
	// Inner brow raise (pleading arch), slight wide eyes, downturn corners.
	Presets.Add(MakePreset(TEXT("sad"), {
		{ TEXT("browInnerUp"),   0.55f },
		{ TEXT("browDown_L"),    0.20f },
		{ TEXT("browDown_R"),    0.20f },
		{ TEXT("eyeWide_L"),     0.15f },
		{ TEXT("eyeWide_R"),     0.15f },
		{ TEXT("mouthFrown_L"),  0.40f },
		{ TEXT("mouthFrown_R"),  0.40f },
	}, 1.0f));

	// ─── NEUTRAL ─────────────────────────────────────────────────────────────
	// Empty preset — all targets return to 0.
	// The driver zeroes out all previously driven targets when transitioning here.
	Presets.Add(MakePreset(TEXT("neutral"), {}, 1.0f));

	return Presets;
}

// ===========================================================================
// Editor support
// ===========================================================================

#if WITH_EDITOR
void UMetaHumanEmotionDriverComponent::PostEditChangeProperty(
	FPropertyChangedEvent& PropertyChangedEvent)
{
	Super::PostEditChangeProperty(PropertyChangedEvent);

	const FName PropName = PropertyChangedEvent.GetPropertyName();
	if (PropName == GET_MEMBER_NAME_CHECKED(UMetaHumanEmotionDriverComponent, ExpressionPresets))
	{
		RebuildMorphTargetSet();
	}
	else if (PropName == GET_MEMBER_NAME_CHECKED(FEmotionOverlaySettings, FaceMeshComponentName))
	{
		// Invalidate cached mesh so it's re-resolved on next tick.
		CachedFaceMesh.Reset();
	}
}
#endif

// ===========================================================================
// Internal helpers
// ===========================================================================

void UMetaHumanEmotionDriverComponent::EnsurePresetsInitialized()
{
	if (bPresetsInitialized) return;

	if (ExpressionPresets.IsEmpty())
	{
		ExpressionPresets = MakeDefaultPresets();
		UE_LOG(LogEmotionBridge, Log,
			TEXT("MetaHumanEmotionDriver: applied built-in ARKit expression presets."));
	}

	RebuildMorphTargetSet();
	bPresetsInitialized = true;
}

void UMetaHumanEmotionDriverComponent::RebuildMorphTargetSet()
{
	AllDrivenMorphTargets.Empty();
	for (const FEmotionExpressionPreset& Preset : ExpressionPresets)
	{
		for (const FEmotionMorphWeight& MW : Preset.MorphWeights)
		{
			AllDrivenMorphTargets.Add(MW.MorphTargetName);
		}
	}
	UE_LOG(LogEmotionBridge, Verbose,
		TEXT("MetaHumanEmotionDriver: driving %d morph targets across %d presets."),
		AllDrivenMorphTargets.Num(), ExpressionPresets.Num());
}

void UMetaHumanEmotionDriverComponent::TickBlend(float DeltaTime)
{
	if (!BlendState.IsBlending()) return;

	const float Duration = FMath::Max(KINDA_SMALL_NUMBER, OverlaySettings.BlendDurationSec);
	BlendState.BlendAlpha = FMath::Clamp(
		BlendState.BlendAlpha + DeltaTime / Duration, 0.f, 1.f);
}

USkeletalMeshComponent* UMetaHumanEmotionDriverComponent::ResolveFaceMesh()
{
	if (CachedFaceMesh.IsValid())
		return CachedFaceMesh.Get();

	AActor* Owner = GetOwner();
	if (!Owner) return nullptr;

	// 1. Named component override.
	if (!OverlaySettings.FaceMeshComponentName.IsNone())
	{
		for (UActorComponent* Comp : Owner->GetComponents())
		{
			if (Comp->GetFName() == OverlaySettings.FaceMeshComponentName)
			{
				if (USkeletalMeshComponent* SK = Cast<USkeletalMeshComponent>(Comp))
				{
					CachedFaceMesh = SK;
					UE_LOG(LogEmotionBridge, Log,
						TEXT("MetaHumanEmotionDriver: using named component '%s'."),
						*Comp->GetName());
					return SK;
				}
			}
		}
		UE_LOG(LogEmotionBridge, Warning,
			TEXT("MetaHumanEmotionDriver: component '%s' not found or not a SkeletalMeshComponent on '%s'."),
			*OverlaySettings.FaceMeshComponentName.ToString(),
			*Owner->GetActorLabel());
	}

	// 2. Auto-detect: prefer component whose name contains "Face".
	TArray<USkeletalMeshComponent*> SkelComps;
	Owner->GetComponents<USkeletalMeshComponent>(SkelComps);

	for (USkeletalMeshComponent* SK : SkelComps)
	{
		if (SK->GetName().Contains(TEXT("Face"), ESearchCase::IgnoreCase))
		{
			CachedFaceMesh = SK;
			UE_LOG(LogEmotionBridge, Log,
				TEXT("MetaHumanEmotionDriver: auto-detected face mesh '%s' on '%s'."),
				*SK->GetName(), *Owner->GetActorLabel());
			return SK;
		}
	}

	// 3. Fall back to first SkeletalMeshComponent.
	if (SkelComps.Num() > 0)
	{
		CachedFaceMesh = SkelComps[0];
		UE_LOG(LogEmotionBridge, Log,
			TEXT("MetaHumanEmotionDriver: using first SkeletalMeshComponent '%s' on '%s'."),
			*SkelComps[0]->GetName(), *Owner->GetActorLabel());
		return SkelComps[0];
	}

	UE_LOG(LogEmotionBridge, Warning,
		TEXT("MetaHumanEmotionDriver: no SkeletalMeshComponent found on '%s'. "
			 "Add a MetaHuman character to the level and bind via the Emotion Bridge panel."),
		*Owner->GetActorLabel());
	return nullptr;
}

const FEmotionExpressionPreset* UMetaHumanEmotionDriverComponent::FindPreset(
	const FString& EmotionName) const
{
	const FString Lower = EmotionName.ToLower();
	for (const FEmotionExpressionPreset& P : ExpressionPresets)
	{
		if (P.EmotionName.ToLower() == Lower)
			return &P;
	}
	return nullptr;
}

float UMetaHumanEmotionDriverComponent::ComputeBlendedWeight(
	FName MorphTargetName,
	const FEmotionExpressionPreset* FromPreset,
	const FEmotionExpressionPreset* ToPreset,
	float Alpha,
	float EffectiveIntensity) const
{
	const float FromWeight = FromPreset ? FromPreset->FindWeight(MorphTargetName) : 0.f;
	const float ToWeight   = ToPreset   ? ToPreset->FindWeight(MorphTargetName)   : 0.f;

	// Linear blend between from and to.
	const float BlendedRaw = FMath::Lerp(FromWeight, ToWeight, Alpha);

	// Scale by effective intensity (includes confidence, multipliers, BaseIntensity).
	return FMath::Clamp(BlendedRaw * EffectiveIntensity, 0.f, 1.f);
}

void UMetaHumanEmotionDriverComponent::ApplyBlendStateToMesh()
{
	if (!OverlaySettings.bEnabled)
		return;

	USkeletalMeshComponent* FaceMesh = ResolveFaceMesh();
	if (!FaceMesh)
		return;

	if (AllDrivenMorphTargets.IsEmpty())
		return;

	const FEmotionExpressionPreset* FromPreset = FindPreset(BlendState.FromEmotion);
	const FEmotionExpressionPreset* ToPreset   = FindPreset(BlendState.ToEmotion);

	// ── Compute effective intensity ──────────────────────────────────────────
	float EffectiveIntensity = 1.f;

	if (OverlaySettings.bUseConfidenceAsWeight)
	{
		// Direct confidence multiplier: low confidence → subtler expression.
		EffectiveIntensity *= FMath::Clamp(BlendState.ToConfidence, 0.f, 1.f);
	}

	// Per-emotion multiplier from settings.
	if (const float* CustomMult =
		OverlaySettings.EmotionIntensityMultipliers.Find(BlendState.ToEmotion))
	{
		EffectiveIntensity *= FMath::Clamp(*CustomMult, 0.f, 2.f);
	}

	// Preset-level BaseIntensity.
	if (ToPreset)
	{
		EffectiveIntensity *= ToPreset->BaseIntensity;
	}

	EffectiveIntensity = FMath::Clamp(EffectiveIntensity, 0.f, 2.f);

	// ── Write morph targets ──────────────────────────────────────────────────
	for (const FName& TargetName : AllDrivenMorphTargets)
	{
		const float FinalWeight = ComputeBlendedWeight(
			TargetName, FromPreset, ToPreset, BlendState.BlendAlpha, EffectiveIntensity);

		// bRemoveZeroWeight=false: keep the entry in the morph target table even at
		// zero so we don't thrash the table during transitions through neutral.
		FaceMesh->SetMorphTarget(TargetName, FinalWeight, /*bRemoveZeroWeight=*/false);
	}
}
