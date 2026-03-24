// Copyright (c) EmotionDemo Project. All rights reserved.
#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "EmotionTimelineTypes.h"
#include "EmotionPlaybackComponent.generated.h"

class AEmotionLampActor;

/** Fired when the active emotion segment changes during playback. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnEmotionChanged,
	const FString&, NewEmotion,
	float,          Confidence);

/** Fired when playback reaches the end of the timeline. */
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnPlaybackFinished);

/**
 * UEmotionPlaybackComponent
 *
 * Attach to any actor. Call SetTimeline() after receiving a /timeline response,
 * then StartPlayback() / StopPlayback() to drive a TargetLampActor's colors.
 *
 * Playback is time-simulation only — no audio is played.
 * Tick is disabled when not playing to avoid unnecessary overhead.
 */
UCLASS(ClassGroup=(EmotionBridge), meta=(BlueprintSpawnableComponent))
class EMOTIONBRIDGE_API UEmotionPlaybackComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UEmotionPlaybackComponent();

	/** Store the timeline returned by FEmotionApiClient. Resets playback cursor. */
	UFUNCTION(BlueprintCallable, Category="EmotionBridge|Playback")
	void SetTimeline(const FEmotionTimelineResponse& InTimeline);

	/** Begin timeline simulation. Does nothing if the timeline is invalid or empty. */
	UFUNCTION(BlueprintCallable, Category="EmotionBridge|Playback")
	void StartPlayback();

	/** Stop playback and reset TargetLampActor to neutral. */
	UFUNCTION(BlueprintCallable, Category="EmotionBridge|Playback")
	void StopPlayback();

	/** Returns true while playback is running. */
	UFUNCTION(BlueprintPure, Category="EmotionBridge|Playback")
	bool IsPlaying() const { return bIsPlaying; }

	// -----------------------------------------------------------------------
	// Events
	// -----------------------------------------------------------------------

	UPROPERTY(BlueprintAssignable, Category="EmotionBridge|Playback")
	FOnEmotionChanged OnEmotionChanged;

	UPROPERTY(BlueprintAssignable, Category="EmotionBridge|Playback")
	FOnPlaybackFinished OnPlaybackFinished;

	// -----------------------------------------------------------------------
	// Configuration
	// -----------------------------------------------------------------------

	/**
	 * The lamp actor whose ApplyEmotion() is called during playback.
	 * Set this in the editor or via Blueprint before calling StartPlayback().
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge|Playback")
	TWeakObjectPtr<AEmotionLampActor> TargetLampActor;

protected:
	virtual void TickComponent(float DeltaTime, ELevelTick TickType,
		FActorComponentTickFunction* ThisTickFunction) override;

private:
	FEmotionTimelineResponse Timeline;
	float ElapsedSec = 0.f;
	bool bIsPlaying = false;
	int32 LastActiveSegmentIndex = -1;

	/**
	 * Walk the segment list and return the emotion active at TimeSec.
	 * Returns "neutral" with confidence=1 when no segment covers the time.
	 */
	FString GetEmotionAtTime(float TimeSec, float& OutConfidence) const;
};
