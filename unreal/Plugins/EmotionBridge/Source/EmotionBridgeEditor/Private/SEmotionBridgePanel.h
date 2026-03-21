// Copyright (c) EmotionDemo Project. All rights reserved.
// Internal — not exported from the module.
#pragma once

#include "CoreMinimal.h"
#include "Widgets/SCompoundWidget.h"
#include "EmotionApiClient.h"
#include "EmotionTimelineTypes.h"
#include "HAL/PlatformProcess.h"

// AudioCapture is used for microphone recording.
#include "AudioCapture.h"

class AEmotionLampActor;

// ---------------------------------------------------------------------------
// Row data for the segment list view
// ---------------------------------------------------------------------------
struct FEmotionSegmentRow
{
	FEmotionSegment Segment;
	explicit FEmotionSegmentRow(const FEmotionSegment& S) : Segment(S) {}
};

// ---------------------------------------------------------------------------
// SEmotionBridgePanel
// ---------------------------------------------------------------------------

/**
 * Main Slate panel for the "Emotion Bridge" editor tab.
 *
 * PLAYBACK:
 *   Driven by SWidget::Tick (overriding GetCanTick → true).
 *   Elapsed time is accumulated via InDeltaTime — no wall-clock race conditions.
 *   Audio (the WAV file) is played via a platform subprocess (afplay / PowerShell)
 *   launched simultaneously with the playback timer.
 *
 * RECORDING:
 *   Uses Audio::FAudioCapture. Click "Record" to capture from the default
 *   microphone; click "Stop Recording" to write a 16-bit mono WAV to a temp
 *   file and auto-populate the WAV path field.
 *
 * ACTOR INTERACTION:
 *   FindOrSpawnLampActor() searches the editor world first, then the PIE world.
 *   The lamp's OnConstruction() creates a DynamicMaterial so color changes are
 *   visible without PIE.
 */
class SEmotionBridgePanel : public SCompoundWidget
{
public:
	SLATE_BEGIN_ARGS(SEmotionBridgePanel) {}
	SLATE_END_ARGS()

	void Construct(const FArguments& InArgs);
	virtual ~SEmotionBridgePanel() override;

	// Slate tick — drives playback; called every frame when the tab is visible.
	// bCanTick is set to true in Construct(); GetCanTick() is non-virtual in UE5.7.
	virtual void Tick(const FGeometry& AllottedGeometry,
		const double InCurrentTime, const float InDeltaTime) override;

private:
	// -----------------------------------------------------------------------
	// Core state
	// -----------------------------------------------------------------------
	TUniquePtr<FEmotionApiClient> ApiClient;
	FEmotionTimelineResponse      CurrentTimeline;
	bool                          bIsAnalyzing = false;

	// -----------------------------------------------------------------------
	// Playback state
	// -----------------------------------------------------------------------
	bool         bIsPlaying             = false;
	double       PlaybackElapsedSec     = 0.0;
	int32        LastActiveSegmentIndex = -1;
	FLinearColor CurrentDisplayColor    = FLinearColor::White; // in-panel color swatch
	TWeakObjectPtr<AEmotionLampActor> LampActorRef;
	FProcHandle  AudioPlayerHandle; // platform audio process (afplay / PowerShell)

	// -----------------------------------------------------------------------
	// Recording state
	// -----------------------------------------------------------------------
	bool   bIsRecording      = false;
	double RecordingElapsedSec = 0.0;
	TArray<float> RecordedSamples;  // protected by RecordingLock
	FCriticalSection RecordingLock;
	int32  CapturedSampleRate  = 16000;
	int32  CapturedNumChannels = 1;
	TUniquePtr<Audio::FAudioCapture> AudioCapture;

	// -----------------------------------------------------------------------
	// Parameter state
	// -----------------------------------------------------------------------
	FString WavFilePath;
	float   WindowSec        = 2.0f;
	float   HopSec           = 0.5f;
	FString PadMode          = TEXT("none");
	FString SmoothingMethod  = TEXT("none");
	int32   HysteresisMinRun = 3;
	int32   MajorityWindow   = 5;
	float   EmaAlpha         = 0.6f;

	// Combo-box option lists
	TArray<TSharedPtr<FString>> PadModeOptions;
	TSharedPtr<FString>         SelectedPadMode;
	TArray<TSharedPtr<FString>> SmoothingOptions;
	TSharedPtr<FString>         SelectedSmoothingMethod;

	// Segment list
	TArray<TSharedPtr<FEmotionSegmentRow>> SegmentRows;

	// -----------------------------------------------------------------------
	// Widget references
	// -----------------------------------------------------------------------
	TSharedPtr<SEditableTextBox>   ApiUrlBox;
	TSharedPtr<SEditableTextBox>   WavPathBox;
	TSharedPtr<STextBlock>         StatusText;
	TSharedPtr<STextBlock>         MetadataText;
	TSharedPtr<STextBlock>         EmotionLabelDisplay; // large emotion name in color swatch
	TSharedPtr<SListView<TSharedPtr<FEmotionSegmentRow>>> SegmentListView;

	// -----------------------------------------------------------------------
	// UI builders
	// -----------------------------------------------------------------------
	TSharedRef<SWidget> BuildBackendSection();
	TSharedRef<SWidget> BuildFileSection();
	TSharedRef<SWidget> BuildParametersSection();
	TSharedRef<SWidget> BuildResultsSection();
	TSharedRef<SWidget> BuildPlaybackSection();

	// -----------------------------------------------------------------------
	// Button callbacks
	// -----------------------------------------------------------------------
	FReply OnHealthCheck();
	FReply OnBrowseWav();
	FReply OnRecordStart();
	FReply OnRecordStop();
	FReply OnAnalyze();
	FReply OnPlayDemo();
	FReply OnStopDemo();
	FReply OnFocusViewport();

	// -----------------------------------------------------------------------
	// Internal
	// -----------------------------------------------------------------------
	void OnTimelineReceived(const FEmotionTimelineResponse& Response);
	void OnHealthCheckResult(bool bHealthy);
	void SetStatus(const FString& Message, FLinearColor Color = FLinearColor::White);

	AEmotionLampActor* FindOrSpawnLampActor();

	/**
	 * Apply the given emotion to ALL targets in the editor world:
	 *  1. Every actor that has a UEmotionColorComponent.
	 *  2. The AEmotionLampActor reference (if still valid).
	 * Call this instead of LampActorRef->ApplyEmotion() directly.
	 */
	void BroadcastEmotion(const FString& Emotion, float Confidence);

	/** Launch a platform-native audio player for WavFilePath (fire-and-forget). */
	void LaunchAudioPlayer();

	/** Stop the audio player process if still running. */
	void StopAudioPlayer();

	/** Write the captured float samples to a 16-bit mono WAV file. Returns path on success. */
	FString WriteRecordingToWav();

	void RefreshSegmentList();
	TSharedRef<ITableRow> GenerateSegmentRow(
		TSharedPtr<FEmotionSegmentRow> Item,
		const TSharedRef<STableViewBase>& OwnerTable);

	FSlateColor GetSlateColorForEmotion(const FString& Emotion) const;
};
