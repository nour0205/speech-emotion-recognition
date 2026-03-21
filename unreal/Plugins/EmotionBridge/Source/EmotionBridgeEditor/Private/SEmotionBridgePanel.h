// Copyright (c) EmotionDemo Project. All rights reserved.
// Internal — not exported from the module.
#pragma once

#include "CoreMinimal.h"
#include "Widgets/SCompoundWidget.h"
#include "EmotionApiClient.h"
#include "EmotionTimelineTypes.h"
#include "Containers/Ticker.h"

class AEmotionLampActor;

// ---------------------------------------------------------------------------
// Helper struct: one row in the segment list view.
// ---------------------------------------------------------------------------
struct FEmotionSegmentRow
{
	FEmotionSegment Segment;

	explicit FEmotionSegmentRow(const FEmotionSegment& InSeg)
		: Segment(InSeg) {}
};

// ---------------------------------------------------------------------------
// SEmotionBridgePanel
// ---------------------------------------------------------------------------

/**
 * Main editor panel for the Emotion Bridge tab.
 *
 * Layout (top to bottom):
 *   [Header]
 *   [Backend section]  — API URL + Health Check + status indicator
 *   [File section]     — WAV path + Browse
 *   [Parameters]       — window_sec, hop_sec, pad_mode, smoothing, min_run, flags
 *   [Analyze button]
 *   [Results section]  — metadata + segment list
 *   [Playback section] — Play Demo + Stop Demo
 *
 * Playback is driven by FTSTicker so it continues even when the panel is
 * occluded. The ticker is removed when playback stops or the panel is destroyed.
 */
class SEmotionBridgePanel : public SCompoundWidget
{
public:
	SLATE_BEGIN_ARGS(SEmotionBridgePanel) {}
	SLATE_END_ARGS()

	void Construct(const FArguments& InArgs);
	virtual ~SEmotionBridgePanel() override;

private:
	// -----------------------------------------------------------------------
	// State
	// -----------------------------------------------------------------------

	/** HTTP client. Owned for the lifetime of the panel. */
	TUniquePtr<FEmotionApiClient> ApiClient;

	/** Latest successful timeline response. */
	FEmotionTimelineResponse CurrentTimeline;

	/** True while waiting for the HTTP response. */
	bool bIsAnalyzing = false;

	// --- Playback state ---
	bool   bIsPlaying             = false;
	double PlaybackStartWallTime  = 0.0;  // FPlatformTime::Seconds() at start
	int32  LastActiveSegmentIndex = -1;
	TWeakObjectPtr<AEmotionLampActor> LampActorRef;
	FTSTicker::FDelegateHandle PlaybackTickerHandle;

	// --- Parameter state ---
	FString WavFilePath;
	float   WindowSec         = 2.0f;
	float   HopSec            = 0.5f;
	FString PadMode           = TEXT("zero");
	FString SmoothingMethod   = TEXT("hysteresis");
	int32   HysteresisMinRun  = 3;
	bool    bIncludeWindows   = false;
	bool    bIncludeScores    = false;

	// --- Combo box options ---
	TArray<TSharedPtr<FString>> PadModeOptions;
	TSharedPtr<FString>         SelectedPadMode;
	TArray<TSharedPtr<FString>> SmoothingOptions;
	TSharedPtr<FString>         SelectedSmoothingMethod;

	// --- Segment list ---
	TArray<TSharedPtr<FEmotionSegmentRow>> SegmentRows;

	// -----------------------------------------------------------------------
	// Widget references (for programmatic updates)
	// -----------------------------------------------------------------------
	TSharedPtr<SEditableTextBox> ApiUrlBox;
	TSharedPtr<SEditableTextBox> WavPathBox;
	TSharedPtr<STextBlock>       StatusText;
	TSharedPtr<SBorder>          StatusBorder;
	TSharedPtr<STextBlock>       MetadataText;
	TSharedPtr<SListView<TSharedPtr<FEmotionSegmentRow>>> SegmentListView;

	// -----------------------------------------------------------------------
	// UI builders (called from Construct)
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
	FReply OnAnalyze();
	FReply OnPlayDemo();
	FReply OnStopDemo();

	// -----------------------------------------------------------------------
	// Internal helpers
	// -----------------------------------------------------------------------

	/** Called on the game thread when the HTTP response arrives. */
	void OnTimelineReceived(const FEmotionTimelineResponse& Response);

	/** Called on the game thread when the health check completes. */
	void OnHealthCheckResult(bool bHealthy);

	/** Update the status bar text and color. */
	void SetStatus(const FString& Message, FLinearColor Color);

	/**
	 * Find an existing AEmotionLampActor in the editor world or spawn a new one.
	 * Returns nullptr if GEditor is unavailable.
	 */
	AEmotionLampActor* FindOrSpawnLampActor();

	/** FTSTicker callback; returns true to keep ticking, false to stop. */
	bool OnPlaybackTick(float DeltaTime);

	/** Populate SegmentRows from CurrentTimeline and refresh the list view. */
	void RefreshSegmentList();

	/** Generate a single row widget for the segment list view. */
	TSharedRef<ITableRow> GenerateSegmentRow(
		TSharedPtr<FEmotionSegmentRow> Item,
		const TSharedRef<STableViewBase>& OwnerTable);

	/** Return a Slate color for the given emotion label (reads Settings). */
	FSlateColor GetSlateColorForEmotion(const FString& Emotion) const;

	/** Highlight the currently active row during playback. */
	FSlateColor GetRowColor(TSharedPtr<FEmotionSegmentRow> Row) const;

	/** Column identifiers for the segment list view. */
	static const FName ColStart;
	static const FName ColEnd;
	static const FName ColEmotion;
	static const FName ColConfidence;
};
