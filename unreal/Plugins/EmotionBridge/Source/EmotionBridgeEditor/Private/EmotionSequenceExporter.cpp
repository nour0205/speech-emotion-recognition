// Copyright (c) EmotionDemo Project. All rights reserved.
// M2 — Implementation.

#include "EmotionSequenceExporter.h"

#include "EmotionBridgeLog.h"

#include "LevelSequence.h"
#include "MovieScene.h"
#include "MovieSceneSection.h"
#include "MovieScenePossessable.h"
#include "Tracks/MovieSceneAudioTrack.h"
#include "Tracks/MovieSceneSkeletalAnimationTrack.h"
#include "Sections/MovieSceneAudioSection.h"
#include "Sections/MovieSceneSkeletalAnimationSection.h"

#include "Animation/AnimSequence.h"
#include "Components/SkeletalMeshComponent.h"
#include "Engine/Engine.h"
#include "Engine/World.h"
#include "Sound/SoundBase.h"
#include "Sound/SoundWave.h"
#include "GameFramework/Actor.h"

namespace
{
	/**
	 * Finds the Face SkeletalMeshComponent on a MetaHuman actor.  Prefers
	 * a component literally named "Face"; falls back to any SkeletalMesh
	 * whose name contains "Face" (case-insensitive).
	 */
	USkeletalMeshComponent* FindFaceComponent(AActor* Actor)
	{
		if (!Actor) return nullptr;

		USkeletalMeshComponent* Exact = nullptr;
		USkeletalMeshComponent* Fallback = nullptr;

		TArray<USkeletalMeshComponent*> SkelComps;
		Actor->GetComponents<USkeletalMeshComponent>(SkelComps);
		for (USkeletalMeshComponent* SK : SkelComps)
		{
			const FString N = SK->GetName();
			if (N.Equals(TEXT("Face"), ESearchCase::IgnoreCase))
			{
				Exact = SK;
				break;
			}
			if (!Fallback && N.Contains(TEXT("Face"), ESearchCase::IgnoreCase))
			{
				Fallback = SK;
			}
		}
		return Exact ? Exact : Fallback;
	}

	/**
	 * Returns the FGuid binding for an object in the given Level Sequence.
	 * Uses UMovieSceneSequence::FindPossessableObjectId — the non-deprecated
	 * lookup (verified in UE 5.7's MovieSceneSequence.h:187).  Avoids both
	 * the deprecated FindBindingFromObject(Obj, Ctx) and LocateBoundObjects
	 * helpers (deprecated in 5.5).
	 */
	FGuid FindBinding(ULevelSequence* Seq, UObject* Object, UObject* Context)
	{
		if (!Seq || !Object || !Context) return FGuid();
		return Seq->FindPossessableObjectId(*Object, Context);
	}

	/** Removes every track of a given UClass from a Movie Scene binding. */
	void RemoveTracksOfClass(UMovieScene* MS, const FGuid& Binding, UClass* TrackClass)
	{
		if (!MS || !Binding.IsValid() || !TrackClass) return;

		// Snapshot pointers — RemoveTrack invalidates the underlying array.
		TArray<UMovieSceneTrack*> Snapshot;
		for (UMovieSceneTrack* T : MS->FindTracks(TrackClass, Binding))
		{
			Snapshot.Add(T);
		}
		for (UMovieSceneTrack* T : Snapshot)
		{
			MS->RemoveTrack(*T);
		}
	}

	/** Removes every master track of a given UClass. */
	void RemoveMasterTracksOfClass(UMovieScene* MS, UClass* TrackClass)
	{
		if (!MS || !TrackClass) return;

		TArray<UMovieSceneTrack*> Snapshot;
		for (UMovieSceneTrack* T : MS->GetTracks())
		{
			if (T && T->IsA(TrackClass))
			{
				Snapshot.Add(T);
			}
		}
		for (UMovieSceneTrack* T : Snapshot)
		{
			MS->RemoveTrack(*T);
		}
	}
}

bool FEmotionSequenceExporter::PopulateSequence(
	ULevelSequence* TargetSeq,
	AActor* MetaHumanActor,
	USoundBase* AudioAsset,
	UAnimSequence* FaceAnim,
	FString& OutErrorMsg)
{
#if !WITH_EDITOR
	OutErrorMsg = TEXT("PopulateSequence requires editor build.");
	return false;
#else
	OutErrorMsg.Reset();

	if (!TargetSeq)
	{
		OutErrorMsg = TEXT("TargetSeq is null. Pick a Level Sequence in the panel.");
		return false;
	}
	if (!MetaHumanActor)
	{
		OutErrorMsg = TEXT("MetaHumanActor is null. Bind a MetaHuman in the panel.");
		return false;
	}
	if (!AudioAsset && !FaceAnim)
	{
		OutErrorMsg = TEXT("Neither audio nor animation supplied — nothing to export.");
		return false;
	}

	UMovieScene* MS = TargetSeq->GetMovieScene();
	if (!MS)
	{
		OutErrorMsg = TEXT("Level Sequence has no MovieScene.");
		return false;
	}

	UWorld* World = MetaHumanActor->GetWorld();
	if (!World)
	{
		OutErrorMsg = TEXT("MetaHuman actor is not in a world (placed in a level?).");
		return false;
	}

	MS->Modify();

	// ── Compute duration from inputs ─────────────────────────────────────────
	float DurationSec = 0.f;
	if (AudioAsset)
	{
		DurationSec = FMath::Max(DurationSec, AudioAsset->GetDuration());
	}
	if (FaceAnim)
	{
		DurationSec = FMath::Max(DurationSec, FaceAnim->GetPlayLength());
	}
	if (DurationSec <= 0.f) DurationSec = 1.f; // safety

	const FFrameRate TickResolution = MS->GetTickResolution();
	const FFrameNumber EndFrame = (DurationSec * TickResolution).RoundToFrame();
	const TRange<FFrameNumber> SectionRange(FFrameNumber(0), EndFrame);

	// Set the playback range to match.
	MS->SetPlaybackRange(SectionRange, /*bAlwaysMarkDirty=*/true);

	// ── Resolve / create the actor possessable ───────────────────────────────
	FGuid ActorGuid = FindBinding(TargetSeq, MetaHumanActor, World);
	if (!ActorGuid.IsValid())
	{
		ActorGuid = MS->AddPossessable(
			MetaHumanActor->GetActorLabel(),
			MetaHumanActor->GetClass());
		TargetSeq->BindPossessableObject(ActorGuid, *MetaHumanActor, World);
		UE_LOG(LogEmotionBridge, Log,
			TEXT("PopulateSequence: added new possessable for actor '%s' (Guid=%s)."),
			*MetaHumanActor->GetActorLabel(), *ActorGuid.ToString());
	}

	// ── Resolve / create the Face component possessable ──────────────────────
	USkeletalMeshComponent* FaceComp = FindFaceComponent(MetaHumanActor);
	FGuid FaceGuid;
	if (FaceComp)
	{
		FaceGuid = FindBinding(TargetSeq, FaceComp, MetaHumanActor);
		if (!FaceGuid.IsValid())
		{
			FaceGuid = MS->AddPossessable(FaceComp->GetName(), FaceComp->GetClass());
			TargetSeq->BindPossessableObject(FaceGuid, *FaceComp, MetaHumanActor);
			if (FMovieScenePossessable* Possessable = MS->FindPossessable(FaceGuid))
			{
				Possessable->SetParent(ActorGuid, MS);
			}
			UE_LOG(LogEmotionBridge, Log,
				TEXT("PopulateSequence: added Face component possessable (Guid=%s, parent=%s)."),
				*FaceGuid.ToString(), *ActorGuid.ToString());
		}
	}
	else if (FaceAnim)
	{
		OutErrorMsg = FString::Printf(
			TEXT("Could not find a 'Face' SkeletalMeshComponent on '%s' — cannot bind face animation."),
			*MetaHumanActor->GetActorLabel());
		return false;
	}

	// ── Audio: clear any existing master audio tracks, add fresh one ─────────
	if (AudioAsset)
	{
		RemoveMasterTracksOfClass(MS, UMovieSceneAudioTrack::StaticClass());

		UMovieSceneAudioTrack* AudioTrack = MS->AddTrack<UMovieSceneAudioTrack>();
		if (!AudioTrack)
		{
			OutErrorMsg = TEXT("Failed to add UMovieSceneAudioTrack.");
			return false;
		}
		AudioTrack->SetDisplayName(FText::FromString(TEXT("EmotionBridge Audio")));

		UMovieSceneAudioSection* AudioSection = Cast<UMovieSceneAudioSection>(
			AudioTrack->CreateNewSection());
		if (!AudioSection)
		{
			OutErrorMsg = TEXT("Failed to create audio section.");
			return false;
		}
		AudioSection->SetSound(AudioAsset);
		AudioSection->SetRange(SectionRange);
		AudioTrack->AddSection(*AudioSection);
	}

	// ── Face animation: clear any existing animation tracks on Face binding ──
	if (FaceAnim && FaceGuid.IsValid())
	{
		RemoveTracksOfClass(MS, FaceGuid, UMovieSceneSkeletalAnimationTrack::StaticClass());

		UMovieSceneSkeletalAnimationTrack* AnimTrack = Cast<UMovieSceneSkeletalAnimationTrack>(
			MS->AddTrack(UMovieSceneSkeletalAnimationTrack::StaticClass(), FaceGuid));
		if (!AnimTrack)
		{
			OutErrorMsg = TEXT("Failed to add UMovieSceneSkeletalAnimationTrack on Face binding.");
			return false;
		}
		AnimTrack->SetDisplayName(FText::FromString(TEXT("EmotionBridge Face Animation")));

		UMovieSceneSkeletalAnimationSection* AnimSection = Cast<UMovieSceneSkeletalAnimationSection>(
			AnimTrack->AddNewAnimation(FFrameNumber(0), FaceAnim));
		if (!AnimSection)
		{
			OutErrorMsg = TEXT("Failed to add animation section.");
			return false;
		}
		AnimSection->SetRange(SectionRange);
	}

	TargetSeq->MarkPackageDirty();

	UE_LOG(LogEmotionBridge, Log,
		TEXT("PopulateSequence: '%s' updated — actor='%s', audio='%s', faceAnim='%s', duration=%.2fs."),
		*TargetSeq->GetPathName(),
		*MetaHumanActor->GetActorLabel(),
		AudioAsset ? *AudioAsset->GetName() : TEXT("<none>"),
		FaceAnim   ? *FaceAnim->GetName()   : TEXT("<none>"),
		DurationSec);

	return true;
#endif
}
