// Copyright (c) EmotionDemo Project. All rights reserved.

#include "EmotionPlaybackComponent.h"
#include "EmotionLampActor.h"
#include "EmotionBridgeLog.h"

UEmotionPlaybackComponent::UEmotionPlaybackComponent()
{
	// Start with tick disabled; enable only during active playback.
	PrimaryComponentTick.bCanEverTick = true;
	PrimaryComponentTick.bStartWithTickEnabled = false;
}

void UEmotionPlaybackComponent::SetTimeline(const FEmotionTimelineResponse& InTimeline)
{
	Timeline               = InTimeline;
	ElapsedSec             = 0.f;
	LastActiveSegmentIndex = -1;

	UE_LOG(LogEmotionBridge, Log,
		TEXT("PlaybackComponent: timeline set — %d segments, %.2f s total."),
		Timeline.Segments.Num(), Timeline.DurationSec);
}

void UEmotionPlaybackComponent::StartPlayback()
{
	if (!Timeline.bIsValid)
	{
		UE_LOG(LogEmotionBridge, Warning,
			TEXT("PlaybackComponent: StartPlayback called but timeline is invalid. Run Analyze first."));
		return;
	}
	if (Timeline.Segments.Num() == 0)
	{
		UE_LOG(LogEmotionBridge, Warning,
			TEXT("PlaybackComponent: StartPlayback called but segment list is empty."));
		return;
	}

	ElapsedSec             = 0.f;
	LastActiveSegmentIndex = -1;
	bIsPlaying             = true;
	SetComponentTickEnabled(true);

	UE_LOG(LogEmotionBridge, Log, TEXT("PlaybackComponent: playback started."));
}

void UEmotionPlaybackComponent::StopPlayback()
{
	bIsPlaying             = false;
	ElapsedSec             = 0.f;
	LastActiveSegmentIndex = -1;
	SetComponentTickEnabled(false);

	// Reset the lamp to a neutral state on explicit stop.
	if (TargetLampActor.IsValid())
	{
		TargetLampActor->ResetToNeutral();
	}

	UE_LOG(LogEmotionBridge, Log, TEXT("PlaybackComponent: playback stopped, lamp reset to neutral."));
}

void UEmotionPlaybackComponent::TickComponent(
	float DeltaTime, ELevelTick TickType,
	FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	if (!bIsPlaying)
		return;

	ElapsedSec += DeltaTime;

	// Check for end of timeline.
	if (ElapsedSec >= Timeline.DurationSec)
	{
		bIsPlaying = false;
		SetComponentTickEnabled(false);
		OnPlaybackFinished.Broadcast();
		UE_LOG(LogEmotionBridge, Log,
			TEXT("PlaybackComponent: reached end of timeline (%.2f s)."), ElapsedSec);
		// Hold the last applied emotion — do NOT reset to neutral so the
		// user can see the final state before clicking Stop.
		return;
	}

	// Find which segment is active right now.
	int32 ActiveIndex = -1;
	for (int32 i = 0; i < Timeline.Segments.Num(); ++i)
	{
		const FEmotionSegment& Seg = Timeline.Segments[i];
		if (ElapsedSec >= Seg.StartSec && ElapsedSec < Seg.EndSec)
		{
			ActiveIndex = i;
			break;
		}
	}

	// Only update the lamp when the active segment actually changes.
	if (ActiveIndex == LastActiveSegmentIndex)
		return;

	LastActiveSegmentIndex = ActiveIndex;

	if (ActiveIndex >= 0)
	{
		const FEmotionSegment& Seg = Timeline.Segments[ActiveIndex];
		UE_LOG(LogEmotionBridge, Log,
			TEXT("PlaybackComponent: t=%.2f segment[%d] emotion=%s conf=%.2f"),
			ElapsedSec, ActiveIndex, *Seg.Emotion, Seg.Confidence);

		if (TargetLampActor.IsValid())
		{
			TargetLampActor->ApplyEmotion(Seg.Emotion, Seg.Confidence);
		}
		else
		{
			UE_LOG(LogEmotionBridge, Warning,
				TEXT("PlaybackComponent: TargetLampActor is not set or was destroyed."));
		}

		OnEmotionChanged.Broadcast(Seg.Emotion, Seg.Confidence);
	}
	else
	{
		// Time falls outside all segments — apply neutral.
		if (TargetLampActor.IsValid())
		{
			TargetLampActor->ApplyEmotion(TEXT("neutral"), 1.0f);
		}
		OnEmotionChanged.Broadcast(TEXT("neutral"), 1.0f);
	}
}

FString UEmotionPlaybackComponent::GetEmotionAtTime(float TimeSec, float& OutConfidence) const
{
	for (const FEmotionSegment& Seg : Timeline.Segments)
	{
		if (TimeSec >= Seg.StartSec && TimeSec < Seg.EndSec)
		{
			OutConfidence = Seg.Confidence;
			return Seg.Emotion;
		}
	}
	OutConfidence = 1.0f;
	return TEXT("neutral");
}
