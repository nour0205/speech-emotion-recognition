// Copyright (c) EmotionDemo Project. All rights reserved.
// Phase 2B — UMetaHumanEmotionDriverComponent implementation.

#include "MetaHumanEmotionDriverComponent.h"
#include "MetaHumanEmotionAnimInstance.h"
#include "EmotionBridgeLog.h"

#include "Components/SkeletalMeshComponent.h"
#include "Engine/SkeletalMesh.h"
#include "Animation/MorphTarget.h"
#include "Animation/AnimClassInterface.h"
#include "GameFramework/Actor.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "ControlRig.h"
#include "AnimNode_ControlRig.h"
#include "AnimNode_ControlRigBase.h"
#include "UObject/UObjectIterator.h"

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
	ResolveFaceMesh();   // warm the cache early
	EnsureAnimInstance(); // configure face mesh tick flags (no AnimInstance swap)
}

void UMetaHumanEmotionDriverComponent::TickComponent(
	float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// ── Heartbeat — confirms the component is actually being ticked by the
	//    editor when not in PIE.  The FIRST 3 heartbeats fire at Log level so
	//    they always show in the Output Log; after that we downgrade to Verbose
	//    to avoid spam.  If you never see a single "Driver TICK" line while the
	//    emotion panel is playing, tick registration failed and no face
	//    animation will ever happen — that is itself the diagnosis.
	{
		static double LastHeartbeat = 0.0;
		static int32  HeartbeatCount = 0;
		const double Now = FPlatformTime::Seconds();
		if (Now - LastHeartbeat > 1.0)
		{
			LastHeartbeat = Now;
			++HeartbeatCount;

			const TCHAR* ActorLbl = GetOwner() ? *GetOwner()->GetActorLabel() : TEXT("<none>");
			const TCHAR* AutonLbl = bIsAutonomousPlaying ? TEXT("YES") : TEXT("no");

			if (HeartbeatCount <= 3)
			{
				UE_LOG(LogEmotionBridge, Log,
					TEXT("Driver TICK #%d  actor='%s' from='%s' to='%s' alpha=%.2f conf=%.2f autonomous=%s elapsed=%.2f TickType=%d dt=%.4f"),
					HeartbeatCount, ActorLbl,
					*BlendState.FromEmotion, *BlendState.ToEmotion,
					BlendState.BlendAlpha, BlendState.ToConfidence,
					AutonLbl, AutonomousElapsedSec, (int32)TickType, DeltaTime);
			}
			else
			{
				UE_LOG(LogEmotionBridge, Verbose,
					TEXT("Driver TICK #%d  actor='%s' from='%s' to='%s' alpha=%.2f conf=%.2f autonomous=%s elapsed=%.2f TickType=%d dt=%.4f"),
					HeartbeatCount, ActorLbl,
					*BlendState.FromEmotion, *BlendState.ToEmotion,
					BlendState.BlendAlpha, BlendState.ToConfidence,
					AutonLbl, AutonomousElapsedSec, (int32)TickType, DeltaTime);
			}
		}
	}

	EnsurePresetsInitialized();
	EnsureAnimInstance(); // no-ops once installed; re-installs if mesh changes

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
	UE_LOG(LogEmotionBridge, Log,
		TEXT("MetaHumanEmotionDriver: blend '%s' -> '%s' (conf=%.2f)"),
		*BlendState.ToEmotion, *NormEmotion, InConfidence);

	BlendState.FromEmotion  = BlendState.ToEmotion;   // snap from = previous target
	BlendState.ToEmotion    = NormEmotion;
	BlendState.BlendAlpha   = 0.f;
	BlendState.ToConfidence = FMath::Clamp(InConfidence, 0.f, 1.f);

	// Enable ticking (safe to call repeatedly).
	SetComponentTickEnabled(true);

	// Post-state diagnostic — confirms that after this call the component is
	// in a state where facial animation CAN happen.  If any of these report
	// false / null, that's exactly the pipeline stage that is broken.
	USkeletalMeshComponent* FaceMeshDbg = ResolveFaceMesh();
	UE_LOG(LogEmotionBridge, Verbose,
		TEXT("ApplyEmotion POST  tickEnabled=%s registered=%s faceMesh=%s animInst=%s postProc=%s "
		     "preset='%s'=%s presetsTotal=%d"),
		IsComponentTickEnabled() ? TEXT("YES") : TEXT("NO"),
		IsRegistered()           ? TEXT("YES") : TEXT("NO"),
		FaceMeshDbg ? *FaceMeshDbg->GetName() : TEXT("null"),
		(FaceMeshDbg && FaceMeshDbg->GetAnimInstance())
			? *FaceMeshDbg->GetAnimInstance()->GetClass()->GetName()
			: TEXT("null"),
		(FaceMeshDbg && FaceMeshDbg->GetPostProcessInstance())
			? *FaceMeshDbg->GetPostProcessInstance()->GetClass()->GetName()
			: TEXT("null"),
		*NormEmotion,
		FindPreset(NormEmotion) ? TEXT("FOUND") : TEXT("MISSING"),
		ExpressionPresets.Num());
}

void UMetaHumanEmotionDriverComponent::ResetToNeutral()
{
	BlendState.FromEmotion  = TEXT("neutral");
	BlendState.ToEmotion    = TEXT("neutral");
	BlendState.BlendAlpha   = 1.f;
	BlendState.ToConfidence = 1.f;

	// Zero every control by calling Face_AnimBP::SetControl(name, 0).
	// Uses the same property-iteration-based parameter binding as
	// ApplyBlendStateToMesh so we match the actual signature
	// (ControlName:FName, Value:double, bControlAdded:bool).
	if (USkeletalMeshComponent* FaceMesh = ResolveFaceMesh())
	{
		UAnimInstance* MainInst = FaceMesh->GetAnimInstance();
		if (MainInst && CachedSetControlFn
			&& CachedSetControlClass.Get() == MainInst->GetClass())
		{
			const int32 ParmsSize = CachedSetControlFn->ParmsSize;
			for (const FName& ControlName : AllDrivenMorphTargets)
			{
				uint8* ParmBuffer = (uint8*)FMemory_Alloca(ParmsSize);
				FMemory::Memzero(ParmBuffer, ParmsSize);

				for (TFieldIterator<FProperty> It(CachedSetControlFn); It; ++It)
				{
					FProperty* Prop = *It;
					if (!Prop->HasAnyPropertyFlags(CPF_Parm)) continue;
					if (Prop->HasAnyPropertyFlags(CPF_ReturnParm)) continue;

					if (FNameProperty* NP = CastField<FNameProperty>(Prop))
					{
						NP->SetPropertyValue_InContainer(ParmBuffer, ControlName);
					}
					else if (FDoubleProperty* DP = CastField<FDoubleProperty>(Prop))
					{
						DP->SetPropertyValue_InContainer(ParmBuffer, 0.0);
					}
					else if (FFloatProperty* FP = CastField<FFloatProperty>(Prop))
					{
						FP->SetPropertyValue_InContainer(ParmBuffer, 0.0f);
					}
				}

				MainInst->ProcessEvent(CachedSetControlFn, ParmBuffer);
			}
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
	CachedFaceMesh    = InMesh;
	CachedControlRig.Reset();         // legacy — no longer used
	CachedSetControlFn = nullptr;     // re-resolve SetControl on new AnimInstance
	CachedSetControlClass.Reset();
	LastInstalledMesh.Reset();        // force AnimInstance re-configure on new mesh
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
	// Presets use MetaHuman Face_ControlBoard_CtrlRig control names directly.
	// Names are verbatim from the rig's Rig Hierarchy on Amelia — pattern is:
	//     CTRL_<L|R>_<feature>_<direction>
	// Values drive UControlRig::SetControlValue<float> each tick, which
	// propagates through RigLogic to bones + corrective morphs — the real
	// MetaHuman facial-animation path.
	//
	// The field struct is still called FEmotionMorphWeight / MorphTargetName
	// for historical reasons, but now stores Control Rig control names.

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

	// ─── CURVE NAMING ────────────────────────────────────────────────────────
	// Names here match the animation-curve / DNA raw-control names that the
	// PostProcess ABP's RigLogic reads — NOT the Control Rig editor's
	// friendly display names.  Convention (verified against Performance-
	// generated AnimSequences):
	//     ctrl_expressions_<feature_lowercase_no_underscores><side>
	// e.g. CTRL_L_brow_down (rig editor) → ctrl_expressions_browdownl (curve).
	//
	// Verified from the Amelia_FaceAnim curves panel:
	//   ctrl_expressions_browdown{l,r}
	//   ctrl_expressions_browlateral{l,r}
	//   ctrl_expressions_browraisein{l,r}
	//   ctrl_expressions_browraiseouter{l,r}  (note "outer", not "out")
	// The remaining names follow the same scheme but have not yet been
	// eyeballed in the curves list.  The baker's diagnostic logs the delta
	// between preset names and existing AnimSequence curves so any mismatch
	// is obvious after the first bake.

	// Each preset is designed around a UNIQUE silhouette so emotions read at
	// a glance:
	//   ANGRY  — brow DOWN + flared nostrils + sneer (no mouth involvement;
	//            lip sync owns the mouth)
	//   HAPPY  — mouth corners UP + cheek raise (only emotion that pulls
	//            mouth corners up)
	//   SAD    — inner brow UP + mouth corners DOWN (only emotion that does
	//            either)
	// Transition smoothness is governed by BlendDurationSec (0.4s default)
	// in the baker, NOT by these values — turning weights up doesn't make
	// crossfades any more abrupt.

	// ─── ANGRY ───────────────────────────────────────────────────────────────
	// Furrowed brow, lateral slant, tight inner squint, sneer with full nose
	// involvement, flared nostrils.  The nostril dilate is the iconic angry
	// tell — no other emotion uses it.
	Presets.Add(MakePreset(TEXT("angry"), {
		{ TEXT("ctrl_expressions_browdownl"),              0.85f },
		{ TEXT("ctrl_expressions_browdownr"),              0.85f },
		{ TEXT("ctrl_expressions_browlaterall"),           0.55f },
		{ TEXT("ctrl_expressions_browlateralr"),           0.55f },
		{ TEXT("ctrl_expressions_eyesquintinnerl"),        0.75f },
		{ TEXT("ctrl_expressions_eyesquintinnerr"),        0.75f },
		{ TEXT("ctrl_expressions_nosewrinkleupperl"),      0.75f },
		{ TEXT("ctrl_expressions_nosewrinkleupperr"),      0.75f },
		{ TEXT("ctrl_expressions_nosewrinklel"),           0.45f }, // base sneer reinforcement
		{ TEXT("ctrl_expressions_nosewrinkler"),           0.45f },
		{ TEXT("ctrl_expressions_nosenostrildilatel"),     0.60f }, // flared nostrils — angry-only
		{ TEXT("ctrl_expressions_nosenostrildilater"),     0.60f },
	}, 1.0f));

	// ─── HAPPY ───────────────────────────────────────────────────────────────
	// Big Duchenne smile: full-width mouth-corner pull, dimples, full cheek
	// raise (genuine smile signal), happy eye-squint, outer brow lift to
	// brighten the whole face.
	Presets.Add(MakePreset(TEXT("happy"), {
		{ TEXT("ctrl_expressions_mouthcornerpulll"),   0.90f }, // three l's: pull + l
		{ TEXT("ctrl_expressions_mouthcornerpullr"),   0.90f },
		{ TEXT("ctrl_expressions_mouthdimplel"),       0.55f },
		{ TEXT("ctrl_expressions_mouthdimpler"),       0.55f },
		{ TEXT("ctrl_expressions_eyecheekraisel"),     0.85f }, // Duchenne marker
		{ TEXT("ctrl_expressions_eyecheekraiser"),     0.85f },
		{ TEXT("ctrl_expressions_eyesquintinnerl"),    0.45f },
		{ TEXT("ctrl_expressions_eyesquintinnerr"),    0.45f },
		{ TEXT("ctrl_expressions_browraiseouterl"),    0.30f },
		{ TEXT("ctrl_expressions_browraiseouterr"),    0.30f },
	}, 1.0f));

	// ─── SAD ─────────────────────────────────────────────────────────────────
	// Classic pouty sadness: very pronounced inner-brow raise (key sad
	// marker — only sad has this), worried pinch from a small brow-down
	// addition, heavy mouth-corner droop, slight wince + light nasolabial
	// pull to suggest the burden.
	Presets.Add(MakePreset(TEXT("sad"), {
		{ TEXT("ctrl_expressions_browraiseinl"),           0.90f }, // signature sad cue
		{ TEXT("ctrl_expressions_browraiseinr"),           0.90f },
		{ TEXT("ctrl_expressions_browdownl"),              0.35f }, // worried pinch
		{ TEXT("ctrl_expressions_browdownr"),              0.35f },
		{ TEXT("ctrl_expressions_mouthcornerdepressl"),    0.85f }, // signature sad cue
		{ TEXT("ctrl_expressions_mouthcornerdepressr"),    0.85f },
		{ TEXT("ctrl_expressions_eyesquintinnerl"),        0.20f }, // subtle wince
		{ TEXT("ctrl_expressions_eyesquintinnerr"),        0.20f },
		{ TEXT("ctrl_expressions_nosewrinklel"),           0.20f }, // slight nasolabial drag
		{ TEXT("ctrl_expressions_nosewrinkler"),           0.20f },
	}, 1.0f));

	// ─── NEUTRAL ─────────────────────────────────────────────────────────────
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
		// Invalidate cached mesh + rig so they're re-resolved on next tick.
		CachedFaceMesh.Reset();
		CachedControlRig.Reset();
		LastInstalledMesh.Reset();
	}
}
#endif

// ===========================================================================
// Internal helpers
// ===========================================================================

void UMetaHumanEmotionDriverComponent::EnsureAnimInstance()
{
	USkeletalMeshComponent* FaceMesh = ResolveFaceMesh();
	if (!FaceMesh) return;

	// Once we've fully configured this mesh, bail out early.
	if (LastInstalledMesh.Get() == FaceMesh)
	{
		return;
	}

	// ── Diagnostics ──────────────────────────────────────────────────────────
	const FString OwnerLabel = GetOwner() ? GetOwner()->GetActorLabel() : TEXT("<no-owner>");
	const FString SkelAssetName = FaceMesh->GetSkeletalMeshAsset()
		? FaceMesh->GetSkeletalMeshAsset()->GetName()
		: TEXT("<NO SKELETAL MESH ASSET>");

	UAnimInstance* CurInst = FaceMesh->GetAnimInstance();
	const FString CurClassName = CurInst ? CurInst->GetClass()->GetName() : TEXT("<none>");

	UE_LOG(LogEmotionBridge, Log,
		TEXT("MetaHumanEmotionDriver: configuring face mesh actor='%s' meshComp='%s' skelAsset='%s' animClass='%s' (NOT swapping — leaving MetaHuman's original AnimBP intact)."),
		*OwnerLabel, *FaceMesh->GetName(), *SkelAssetName, *CurClassName);

	// NO AnimInstance swap.  We drive the mesh purely through SetMorphTarget
	// from the game thread, which bypasses AnimGraph / RigLogic entirely and
	// lets MetaHuman's original face AnimBP keep doing whatever it does.
	LastInstalledMesh = FaceMesh;

	// Force the mesh to tick its pose/animation even when the viewport is
	// evaluating the scene outside of PIE.  Morph target writes only take
	// effect during a pose update, so both flags below are required for the
	// face to visibly deform in the Level Editor viewport.
	FaceMesh->VisibilityBasedAnimTickOption =
		EVisibilityBasedAnimTickOption::AlwaysTickPoseAndRefreshBones;

#if WITH_EDITOR
	FaceMesh->SetUpdateAnimationInEditor(true);
	FaceMesh->SetUpdateClothInEditor(true);
	UE_LOG(LogEmotionBridge, Log,
		TEXT("MetaHumanEmotionDriver: '%s' configured (AlwaysTickPoseAndRefreshBones, UpdateAnimationInEditor=true, UpdateClothInEditor=true). Driving via SetMorphTarget."),
		*FaceMesh->GetName());
#else
	UE_LOG(LogEmotionBridge, Log,
		TEXT("MetaHumanEmotionDriver: '%s' configured (AlwaysTickPoseAndRefreshBones). Driving via SetMorphTarget."),
		*FaceMesh->GetName());
#endif

	// Verify PostProcess ABP is active — MetaHuman's DNA solver lives there.
	UAnimInstance* PostProcessInst = FaceMesh->GetPostProcessInstance();
	if (PostProcessInst)
	{
		UE_LOG(LogEmotionBridge, Log,
			TEXT("MetaHumanEmotionDriver: PostProcess ABP active ('%s')."),
			*PostProcessInst->GetClass()->GetName());
	}
	else
	{
		UE_LOG(LogEmotionBridge, Warning,
			TEXT("MetaHumanEmotionDriver: PostProcess AnimInstance is NULL on '%s'. "
			     "SetMorphTarget will still work, but secondary MetaHuman deformation may not."),
			*FaceMesh->GetName());
	}

	// Log morph target count as a rough sanity check — MetaHuman faces usually
	// report 0 here because deformation is driven by RigLogic + DNA, not stock
	// morph targets.  A non-zero number means classic ARKit blendshapes are
	// present and the fallback SetMorphTarget path would work too.
	if (USkeletalMesh* SkelAsset = FaceMesh->GetSkeletalMeshAsset())
	{
		// Use auto& so we work with whatever container type UE uses
		// (TArray<UMorphTarget*> vs TArray<TObjectPtr<UMorphTarget>>).
		const auto& Morphs = SkelAsset->GetMorphTargets();
		const int32 MorphCount = Morphs.Num();
		UE_LOG(LogEmotionBridge, Log,
			TEXT("MetaHumanEmotionDriver: skeletal mesh '%s' exposes %d morph targets."),
			*SkelAsset->GetName(), MorphCount);

		// ── One-time full morph-target dump ──────────────────────────────────
		// Writes all names to Saved/EmotionBridge/MorphTargetDump_<meshname>.txt
		// and logs the first 30 names + a prefix histogram to the Output Log.
		// This is the ground truth for "what can SetMorphTarget actually drive
		// on this mesh" and drives our next decision (ARKit-style direct drive
		// vs Control Rig vs something else).
		static TSet<FName> AlreadyDumped;
		const FName MeshKey = SkelAsset->GetFName();
		if (MorphCount > 0 && !AlreadyDumped.Contains(MeshKey))
		{
			AlreadyDumped.Add(MeshKey);

			// Collect names up front so the rest of the code doesn't depend on
			// the exact container element type.
			TArray<FString> Names;
			Names.Reserve(MorphCount);
			for (int32 i = 0; i < MorphCount; ++i)
			{
				// Implicit conversion handles both raw UMorphTarget* and
				// TObjectPtr<UMorphTarget> cleanly.
				UMorphTarget* MT = Morphs[i];
				Names.Add(MT ? MT->GetName() : FString(TEXT("<null>")));
			}

			// Build file content + prefix histogram.
			FString DumpContent;
			DumpContent.Reserve(MorphCount * 48);
			DumpContent += FString::Printf(TEXT("# Morph Target Dump\n"));
			DumpContent += FString::Printf(TEXT("# Mesh: %s\n"), *SkelAsset->GetPathName());
			DumpContent += FString::Printf(TEXT("# Count: %d\n\n"), MorphCount);

			TMap<FString, int32> PrefixCounts;
			for (int32 i = 0; i < MorphCount; ++i)
			{
				const FString& Name = Names[i];
				DumpContent += FString::Printf(TEXT("[%d] %s\n"), i, *Name);

				int32 UnderscoreIdx = INDEX_NONE;
				FString Prefix = Name;
				if (Name.FindChar(TEXT('_'), UnderscoreIdx))
				{
					Prefix = Name.Left(UnderscoreIdx);
				}
				PrefixCounts.FindOrAdd(Prefix) += 1;
			}

			DumpContent += TEXT("\n# Prefix histogram (first segment before first '_'):\n");

			// Sort prefixes by frequency descending for readability.
			TArray<TPair<FString, int32>> SortedPrefixes;
			for (const auto& KV : PrefixCounts) { SortedPrefixes.Add(KV); }
			SortedPrefixes.Sort([](const TPair<FString, int32>& A, const TPair<FString, int32>& B)
			{
				return A.Value > B.Value;
			});
			for (const auto& KV : SortedPrefixes)
			{
				DumpContent += FString::Printf(TEXT("  %s : %d\n"), *KV.Key, KV.Value);
			}

			// Save to disk.
			const FString DumpPath = FPaths::ProjectSavedDir()
				/ TEXT("EmotionBridge")
				/ FString::Printf(TEXT("MorphTargetDump_%s.txt"), *SkelAsset->GetName());
			if (FFileHelper::SaveStringToFile(DumpContent, *DumpPath))
			{
				UE_LOG(LogEmotionBridge, Log,
					TEXT("MorphDump: wrote %d morph target names to '%s'."),
					MorphCount, *DumpPath);
			}
			else
			{
				UE_LOG(LogEmotionBridge, Warning,
					TEXT("MorphDump: FAILED to write '%s'."), *DumpPath);
			}

			// Also spam first 30 + full histogram into the Output Log so you
			// can eyeball the naming convention without opening the file.
			const int32 HeadLimit = FMath::Min(30, MorphCount);
			UE_LOG(LogEmotionBridge, Log,
				TEXT("MorphDump: first %d of %d morph target names:"), HeadLimit, MorphCount);
			for (int32 i = 0; i < HeadLimit; ++i)
			{
				UE_LOG(LogEmotionBridge, Log,
					TEXT("  morph[%d] = %s"), i, *Names[i]);
			}
			if (MorphCount > HeadLimit)
			{
				UE_LOG(LogEmotionBridge, Log,
					TEXT("  ... (%d more in %s)"), MorphCount - HeadLimit, *DumpPath);
			}

			UE_LOG(LogEmotionBridge, Log,
				TEXT("MorphDump: prefix histogram (top %d):"), SortedPrefixes.Num());
			const int32 PrefixLimit = FMath::Min(20, SortedPrefixes.Num());
			for (int32 i = 0; i < PrefixLimit; ++i)
			{
				UE_LOG(LogEmotionBridge, Log,
					TEXT("  %s : %d"), *SortedPrefixes[i].Key, SortedPrefixes[i].Value);
			}
		}
	}
}

void UMetaHumanEmotionDriverComponent::EnsurePresetsInitialized()
{
	if (bPresetsInitialized) return;

	// Detect stale presets serialized from an earlier iteration of this code.
	// Current convention: every driven name uses the DNA raw-control / animation
	// curve namespace "ctrl_expressions_<name_lowercase><side>" — this is the
	// name RigLogic actually reads on the baked AnimSequence.  Older iterations
	// used "CTRL_<L|R>_..." (rig-editor display names, not read by RigLogic),
	// "CTRL_expressions..." (dot form), or "head_lod0_mesh__..." (raw morphs).
	// Anything that doesn't start with "ctrl_expressions_" is stale.
	bool bStale = false;
	if (!ExpressionPresets.IsEmpty())
	{
		static const FString kExpectedPrefix = TEXT("ctrl_expressions_");
		for (const FEmotionExpressionPreset& P : ExpressionPresets)
		{
			for (const FEmotionMorphWeight& MW : P.MorphWeights)
			{
				const FString NameStr = MW.MorphTargetName.ToString();
				if (!NameStr.StartsWith(kExpectedPrefix))
				{
					bStale = true;
					UE_LOG(LogEmotionBridge, Warning,
						TEXT("MetaHumanEmotionDriver: detected stale preset name '%s' "
						     "(expected prefix '%s'). Regenerating defaults."),
						*NameStr, *kExpectedPrefix);
					break;
				}
			}
			if (bStale) break;
		}
	}

	if (ExpressionPresets.IsEmpty() || bStale)
	{
		ExpressionPresets = MakeDefaultPresets();
		UE_LOG(LogEmotionBridge, Log,
			TEXT("MetaHumanEmotionDriver: applied built-in Control Rig expression presets (%d presets)."),
			ExpressionPresets.Num());
	}
	else
	{
		UE_LOG(LogEmotionBridge, Log,
			TEXT("MetaHumanEmotionDriver: using existing %d preserved expression presets."),
			ExpressionPresets.Num());
	}

	// Dump each preset's contents so the Output Log shows EXACTLY what morph
	// names are being driven and at what weights.  Mismatches between these
	// names and the MorphDump above are the #1 cause of silent no-op face.
	for (const FEmotionExpressionPreset& P : ExpressionPresets)
	{
		UE_LOG(LogEmotionBridge, Log,
			TEXT("  preset '%s' base=%.2f weights=%d"),
			*P.EmotionName, P.BaseIntensity, P.MorphWeights.Num());
		for (const FEmotionMorphWeight& MW : P.MorphWeights)
		{
			UE_LOG(LogEmotionBridge, Log,
				TEXT("    %s = %.2f"),
				*MW.MorphTargetName.ToString(), MW.Weight);
		}
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

namespace
{
	/**
	 * Walks an AnimInstance's anim-node properties, logs what it finds,
	 * and adds every live UControlRig it discovers to Out.
	 * Matches ANY subclass of FAnimNode_ControlRigBase (not just the concrete
	 * FAnimNode_ControlRig), so variants like FAnimNode_ControlRig_ExternalSource
	 * are also captured.
	 */
	void CollectControlRigs(UAnimInstance* Inst, const TCHAR* Label,
	                        TArray<TPair<FString, UControlRig*>>& Out)
	{
		if (!Inst)
		{
			UE_LOG(LogEmotionBridge, Log,
				TEXT("CollectControlRigs [%s]: AnimInstance is NULL."), Label);
			return;
		}

		IAnimClassInterface* IF = IAnimClassInterface::GetFromClass(Inst->GetClass());
		if (!IF)
		{
			UE_LOG(LogEmotionBridge, Log,
				TEXT("CollectControlRigs [%s]: '%s' has no IAnimClassInterface."),
				Label, *Inst->GetClass()->GetName());
			return;
		}

		int32 TotalNodes = 0;
		int32 BaseHits = 0;
		int32 ConcreteHits = 0;
		TSet<FString> NodeTypeNames;

		for (const FStructProperty* NodeProp : IF->GetAnimNodeProperties())
		{
			if (!NodeProp || !NodeProp->Struct) continue;
			++TotalNodes;
			NodeTypeNames.Add(NodeProp->Struct->GetName());

			// Wider net — FAnimNode_ControlRigBase covers the concrete
			// FAnimNode_ControlRig and its variants.
			if (!NodeProp->Struct->IsChildOf(FAnimNode_ControlRigBase::StaticStruct())) continue;
			++BaseHits;

			// We can still only pull UControlRig* from the concrete node type.
			if (NodeProp->Struct->IsChildOf(FAnimNode_ControlRig::StaticStruct()))
			{
				++ConcreteHits;
				FAnimNode_ControlRig* Node = NodeProp->ContainerPtrToValuePtr<FAnimNode_ControlRig>(Inst);
				if (Node)
				{
					if (UControlRig* CR = Node->GetControlRig())
					{
						Out.Add({ FString::Printf(TEXT("%s/%s"), Label, *Inst->GetClass()->GetName()), CR });
					}
				}
			}
		}

		// Build a printable list of unique node types — bounded so the log stays readable.
		FString TypesBlob;
		int32 Shown = 0;
		for (const FString& T : NodeTypeNames)
		{
			if (Shown >= 12) { TypesBlob += TEXT(", ..."); break; }
			if (!TypesBlob.IsEmpty()) TypesBlob += TEXT(", ");
			TypesBlob += T;
			++Shown;
		}

		UE_LOG(LogEmotionBridge, Log,
			TEXT("CollectControlRigs [%s/%s]: %d total nodes, %d ControlRigBase, %d concrete ControlRig. Unique node types: [%s]"),
			Label, *Inst->GetClass()->GetName(),
			TotalNodes, BaseHits, ConcreteHits, *TypesBlob);
	}

	/**
	 * Last-resort scan: walks EVERY live UControlRig object in the process
	 * and keeps the ones whose outer chain leads to our face mesh, its
	 * AnimInstances, or our owning actor.  This catches rigs spawned outside
	 * the AnimBP graph (e.g. via UMetaHumanComponent or a separate subsystem).
	 */
	void CollectControlRigsByObjectIteration(AActor* Owner,
	                                          USkeletalMeshComponent* FaceMesh,
	                                          TArray<TPair<FString, UControlRig*>>& Out)
	{
		if (!FaceMesh) return;

		UAnimInstance* Main = FaceMesh->GetAnimInstance();
		UAnimInstance* PP   = FaceMesh->GetPostProcessInstance();

		int32 Examined = 0;
		int32 Matched  = 0;
		for (TObjectIterator<UControlRig> It; It; ++It)
		{
			UControlRig* CR = *It;
			if (!CR || !IsValid(CR) || CR->HasAnyFlags(RF_ClassDefaultObject)) continue;
			++Examined;

			// Walk outer chain — does it trace to something related to our actor?
			UObject* Cursor = CR;
			while (Cursor)
			{
				if (Cursor == FaceMesh || Cursor == Main || Cursor == PP || Cursor == Owner)
				{
					++Matched;
					Out.Add({ FString::Printf(TEXT("ObjIter/outer=%s"),
						*CR->GetOuter()->GetName()), CR });
					break;
				}
				Cursor = Cursor->GetOuter();
			}
		}

		UE_LOG(LogEmotionBridge, Log,
			TEXT("CollectControlRigsByObjectIteration: examined %d UControlRig objects, %d matched our actor chain."),
			Examined, Matched);
	}
}

UControlRig* UMetaHumanEmotionDriverComponent::ResolveControlRig()
{
	if (CachedControlRig.IsValid())
	{
		return CachedControlRig.Get();
	}

	USkeletalMeshComponent* FaceMesh = ResolveFaceMesh();
	if (!FaceMesh) return nullptr;

	// Collect candidate UControlRigs from BOTH slots:
	//   1. Main AnimInstance (e.g. Face_AnimBP_C) — where MetaHuman puts the
	//      face Control Rig in UE 5.5+.
	//   2. PostProcess AnimInstance (ABP_*_FaceMesh_PostProcess) — where neck
	//      and head-IK rigs live.
	// The face rig is typically in (1).
	TArray<TPair<FString, UControlRig*>> Candidates;
	CollectControlRigs(FaceMesh->GetAnimInstance(),        TEXT("Main"),        Candidates);
	CollectControlRigs(FaceMesh->GetPostProcessInstance(), TEXT("PostProcess"), Candidates);

	// ALWAYS also do the object-iter scan — the face rig may be spawned
	// outside the AnimBP graph (MetaHumanComponent, subsystem, separately
	// instantiated Control Rig, etc).  Duplicates are fine, probes de-dupe.
	CollectControlRigsByObjectIteration(GetOwner(), FaceMesh, Candidates);

	// Rate-limit the not-found error so a failing resolve doesn't spam the log
	// multiple times per tick.  Only log once every 2 seconds.
	static double LastErrorTime = 0.0;
	const double Now = FPlatformTime::Seconds();
	const bool bVerboseThisCall = (Now - LastErrorTime) > 2.0;

	if (Candidates.IsEmpty())
	{
		if (bVerboseThisCall)
		{
			LastErrorTime = Now;
			UE_LOG(LogEmotionBridge, Warning,
				TEXT("ResolveControlRig: no live UControlRig in main or PostProcess AnimInstance on '%s'."),
				*FaceMesh->GetName());
		}
		return nullptr;
	}

	// Strategy 1: class-name match.
	for (const TPair<FString, UControlRig*>& C : Candidates)
	{
		const FString ClassName = C.Value->GetClass()->GetName();
		if (ClassName.Contains(TEXT("Face"), ESearchCase::IgnoreCase)
			|| ClassName.Contains(TEXT("ControlBoard"), ESearchCase::IgnoreCase))
		{
			CachedControlRig = C.Value;
			UE_LOG(LogEmotionBridge, Log,
				TEXT("ResolveControlRig: picked '%s' by class-name match [%s]."),
				*ClassName, *C.Key);
			return C.Value;
		}
	}

	// Strategy 2: sentinel-control probe (CTRL_L_brow_down exists on face rig).
	static const FName SentinelControl(TEXT("CTRL_L_brow_down"));
	for (const TPair<FString, UControlRig*>& C : Candidates)
	{
		if (C.Value->FindControl(SentinelControl) != nullptr)
		{
			CachedControlRig = C.Value;
			UE_LOG(LogEmotionBridge, Log,
				TEXT("ResolveControlRig: picked '%s' by sentinel control '%s' [%s]."),
				*C.Value->GetClass()->GetName(), *SentinelControl.ToString(), *C.Key);
			return C.Value;
		}
	}

	// Nothing matched — log once in a while.  We also probe each candidate
	// rig for a range of likely face-control name variants so we can see
	// which (if any) has face-shaped controls, even if they don't match the
	// primary sentinel.
	if (bVerboseThisCall)
	{
		LastErrorTime = Now;

		static const TArray<FName> ProbeNames = {
			FName(TEXT("CTRL_L_brow_down")),
			FName(TEXT("CTRL_R_brow_down")),
			FName(TEXT("CTRL_face_brow_down_L")),
			FName(TEXT("CTRL_brow_down_L")),
			FName(TEXT("brow_down_L")),
			FName(TEXT("browDown_L")),
			FName(TEXT("CTRL_expressions.browDownL")),
			FName(TEXT("CTRL_L_mouth_cornerPull")),
			FName(TEXT("mouth_cornerPull_L")),
		};

		FString AllInfo;
		for (const TPair<FString, UControlRig*>& C : Candidates)
		{
			FString HitList;
			for (const FName& Probe : ProbeNames)
			{
				if (C.Value->FindControl(Probe) != nullptr)
				{
					if (!HitList.IsEmpty()) HitList += TEXT(", ");
					HitList += Probe.ToString();
				}
			}
			if (!AllInfo.IsEmpty()) AllInfo += TEXT("\n  ");
			AllInfo += FString::Printf(TEXT("[%s] '%s' -- probe hits: [%s]"),
				*C.Key, *C.Value->GetClass()->GetName(),
				HitList.IsEmpty() ? TEXT("<none>") : *HitList);
		}

		UE_LOG(LogEmotionBridge, Error,
			TEXT("ResolveControlRig: no face rig among %d candidates:\n  %s"),
			Candidates.Num(), *AllInfo);
	}
	return nullptr;
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
		EffectiveIntensity *= FMath::Clamp(BlendState.ToConfidence, 0.f, 1.f);
	}

	if (const float* CustomMult =
		OverlaySettings.EmotionIntensityMultipliers.Find(BlendState.ToEmotion))
	{
		EffectiveIntensity *= FMath::Clamp(*CustomMult, 0.f, 2.f);
	}

	if (ToPreset)
	{
		EffectiveIntensity *= ToPreset->BaseIntensity;
	}

	EffectiveIntensity = FMath::Clamp(EffectiveIntensity, 0.f, 2.f);

	// ── Drive the face via Face_AnimBP::SetControl ──────────────────────────
	// The MetaHuman Face_AnimBP exposes a BlueprintCallable function
	//     SetControl(FName ControlName, float Value)
	// that writes into its CustomControlValues TMap; the AnimGraph reads
	// that map and applies values to RigLogic.  This is the same path
	// MetaHuman Performance uses to inject control values offline — using
	// it at runtime is the supported, documented entry point.

	UAnimInstance* MainInst = FaceMesh->GetAnimInstance();
	if (!MainInst)
	{
		static bool bWarned = false;
		if (!bWarned)
		{
			bWarned = true;
			UE_LOG(LogEmotionBridge, Warning,
				TEXT("ApplyBlendStateToMesh: face mesh has no main AnimInstance — cannot call SetControl."));
		}
		return;
	}

	// Resolve & cache the SetControl UFunction.  Re-resolves if the
	// AnimInstance class changes (e.g. user rebound a different actor).
	UClass* CurrentClass = MainInst->GetClass();
	if (CachedSetControlFn == nullptr || CachedSetControlClass.Get() != CurrentClass)
	{
		CachedSetControlClass = CurrentClass;
		// Try the canonical name first, then a couple of plausible variants.
		static const TArray<FName> CandidateFnNames = {
			FName(TEXT("SetControl")),
			FName(TEXT("Set Control")),
			FName(TEXT("SetControlValue")),
		};
		CachedSetControlFn = nullptr;
		for (const FName& Try : CandidateFnNames)
		{
			if (UFunction* Fn = CurrentClass->FindFunctionByName(Try))
			{
				CachedSetControlFn = Fn;
				break;
			}
		}

		if (CachedSetControlFn)
		{
			// Log the resolved function's parameter signature once so we can
			// see exactly what we're calling.
			FString ParamsBlob;
			for (TFieldIterator<FProperty> It(CachedSetControlFn);
				 It && It->HasAnyPropertyFlags(CPF_Parm); ++It)
			{
				if (!ParamsBlob.IsEmpty()) ParamsBlob += TEXT(", ");
				ParamsBlob += FString::Printf(TEXT("%s:%s"),
					*It->GetName(), *It->GetCPPType());
			}
			UE_LOG(LogEmotionBridge, Log,
				TEXT("Resolved Face_AnimBP::%s on '%s'. Signature: (%s). ParmsSize=%d"),
				*CachedSetControlFn->GetName(), *CurrentClass->GetName(),
				*ParamsBlob, CachedSetControlFn->ParmsSize);
		}
		else
		{
			UE_LOG(LogEmotionBridge, Error,
				TEXT("Face_AnimBP class '%s' has NO SetControl/Set Control/SetControlValue function. "
				     "Open Face_AnimBP, check the Functions panel for a function that takes (FName, float), "
				     "and tell me its exact name."),
				*CurrentClass->GetName());
		}
	}

	if (!CachedSetControlFn)
	{
		return;
	}

	static bool bDiagLogged = false;
	if (!bDiagLogged)
	{
		bDiagLogged = true;
		UE_LOG(LogEmotionBridge, Log,
			TEXT("MetaHumanEmotionDriver DIAGNOSTIC: actor='%s' meshComp='%s' driving %d controls "
			     "via %s::%s, EffIntensity=%.2f"),
			GetOwner() ? *GetOwner()->GetActorLabel() : TEXT("<none>"),
			*FaceMesh->GetName(), AllDrivenMorphTargets.Num(),
			*CurrentClass->GetName(), *CachedSetControlFn->GetName(),
			EffectiveIntensity);
	}

	// Throttled summary (once / sec).
	static double LastDumpTime = 0.0;
	const double NowSec = FPlatformTime::Seconds();
	const bool bDump = (NowSec - LastDumpTime) > 1.0;
	if (bDump) { LastDumpTime = NowSec; }

	int32 NonZeroCount = 0;
	float MaxValueSeen = 0.f;

	// Build the parameter buffer using property iteration — no struct-layout
	// assumption.  This handles SetControl whose actual signature is
	//     (FName ControlName, double Value, [out] bool bControlAdded)
	// and any other variant the AnimBP exposes.
	const int32 ParmsSize = CachedSetControlFn->ParmsSize;

	for (const FName& ControlName : AllDrivenMorphTargets)
	{
		const float FinalValue = ComputeBlendedWeight(
			ControlName, FromPreset, ToPreset, BlendState.BlendAlpha, EffectiveIntensity);

		// Allocate + zero a buffer sized to the function's actual parameter
		// block.  FName / double / float / bool are POD-friendly so memzero
		// is sufficient initialization; no Initialize/Destroy needed.
		uint8* ParmBuffer = (uint8*)FMemory_Alloca(ParmsSize);
		FMemory::Memzero(ParmBuffer, ParmsSize);

		// Walk parameters and set inputs by their actual UE-reflected type.
		for (TFieldIterator<FProperty> It(CachedSetControlFn); It; ++It)
		{
			FProperty* Prop = *It;
			if (!Prop->HasAnyPropertyFlags(CPF_Parm)) continue;
			if (Prop->HasAnyPropertyFlags(CPF_ReturnParm)) continue;

			if (FNameProperty* NP = CastField<FNameProperty>(Prop))
			{
				NP->SetPropertyValue_InContainer(ParmBuffer, ControlName);
			}
			else if (FDoubleProperty* DP = CastField<FDoubleProperty>(Prop))
			{
				DP->SetPropertyValue_InContainer(ParmBuffer, (double)FinalValue);
			}
			else if (FFloatProperty* FP = CastField<FFloatProperty>(Prop))
			{
				FP->SetPropertyValue_InContainer(ParmBuffer, FinalValue);
			}
			// bool / other out-params remain zero-initialized; the function
			// is free to write into them, the buffer is sized for it.
		}

		MainInst->ProcessEvent(CachedSetControlFn, ParmBuffer);

		if (FinalValue > KINDA_SMALL_NUMBER) { ++NonZeroCount; }
		MaxValueSeen = FMath::Max(MaxValueSeen, FinalValue);
	}

	if (bDump)
	{
		UE_LOG(LogEmotionBridge, Log,
			TEXT("Driver SETCTL  %d controls via %s, %d non-zero, max=%.3f, to='%s' conf=%.2f alpha=%.2f eff=%.2f"),
			AllDrivenMorphTargets.Num(), *CachedSetControlFn->GetName(),
			NonZeroCount, MaxValueSeen,
			*BlendState.ToEmotion, BlendState.ToConfidence,
			BlendState.BlendAlpha, EffectiveIntensity);
	}
}
