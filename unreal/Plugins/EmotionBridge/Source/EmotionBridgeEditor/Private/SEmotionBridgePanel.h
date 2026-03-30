// Copyright (c) EmotionDemo Project. All rights reserved.
// Internal — not exported from the module.
#pragma once

#include "CoreMinimal.h"
#include "Widgets/SCompoundWidget.h"
#include "EmotionApiClient.h"
#include "EmotionTimelineTypes.h"
#include "EmotionTakeTypes.h"   // Phase 2A
#include "HAL/PlatformProcess.h"

// AudioCapture is used for microphone recording.
#include "AudioCapture.h"

class AEmotionLampActor;
class SEmotionTakeLibrary; // Phase 2A — forward declaration
class UMetaHumanEmotionDriverComponent; // Phase 2B — forward declaration

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
 *
 * PHASE 2B — METAHUMAN EMOTION OVERLAY:
 *   The METAHUMAN FACE section lets the user:
 *     1. Bind a MetaHuman actor (auto-adds UMetaHumanEmotionDriverComponent).
 *     2. Import the WAV as a SoundWave content asset.
 *     3. Toggle overlay, set blend duration and per-emotion intensity.
 *   BroadcastEmotion() drives the bound driver each frame during playback.
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
	TSharedRef<SWidget> BuildSaveTakeSection();     // Phase 2A
	TSharedRef<SWidget> BuildTakeLibrarySection();  // Phase 2A
	TSharedRef<SWidget> BuildMetaHumanSection();    // Phase 2B

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
	FReply OnSaveTakeClicked(); // Phase 2A

	// Phase 2B
	FReply OnBindSelectedActor();
	FReply OnClearMetaHumanBinding();
	FReply OnImportSoundWave();

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
	 *  3. The bound UMetaHumanEmotionDriverComponent (Phase 2B).
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

	// -----------------------------------------------------------------------
	// Phase 2A: Take Library callbacks  (wired to SEmotionTakeLibrary)
	// -----------------------------------------------------------------------

	/**
	 * Load a take: restore timeline, params, and WAV path from a saved record.
	 * Does NOT call the backend — the take is fully self-contained.
	 */
	void OnLoadTakeRequested(const FEmotionTakeRecord& Take);

	/**
	 * Load a take and immediately start playback (combines load + Play Demo).
	 */
	void OnPlayTakeRequested(const FEmotionTakeRecord& Take);

	/**
	 * Trigger reanalysis for an existing take.
	 * Sets WavFilePath and params from the stored record, then fires the
	 * /timeline/unreal request.  When the response arrives, the take is
	 * updated in-place (timeline + params replaced; name/notes/tags kept).
	 */
	void OnReanalyzeTakeRequested(const FEmotionTakeRecord& Take);

	// -----------------------------------------------------------------------
	// Phase 2A state
	// -----------------------------------------------------------------------

	/** Becomes true after a successful analysis. Enables the Save Take button. */
	bool bCanSaveTake = false;

	/**
	 * When non-empty, the next OnTimelineReceived() will update this take ID
	 * in-place instead of just populating the panel.
	 * Set by OnReanalyzeTakeRequested(); cleared in OnTimelineReceived().
	 */
	FString PendingReanalyzeTakeId;

	/** Name entry box inside the Save Take section. */
	TSharedPtr<SEditableTextBox> TakeNameBox;

	/** The embedded Take Library widget. */
	TSharedPtr<SEmotionTakeLibrary> TakeLibraryWidget;

	// -----------------------------------------------------------------------
	// Phase 2B state — MetaHuman emotion overlay
	// -----------------------------------------------------------------------

	/** The MetaHuman actor currently bound for emotion overlay. */
	TWeakObjectPtr<AActor> BoundMetaHumanActor;

	/**
	 * The UMetaHumanEmotionDriverComponent on the bound actor.
	 * Auto-added by OnBindSelectedActor() if the actor doesn't already have one.
	 */
	TWeakObjectPtr<UMetaHumanEmotionDriverComponent> BoundDriverComponent;

	/** Display label of the currently bound actor (used for UI and take save). */
	FString BoundActorLabel;

	/**
	 * Content-browser path to the SoundWave asset imported from WavFilePath.
	 * Example: "/Game/EmotionBridge/Audio/MySpeech"
	 * Populated by OnImportSoundWave() and saved in the take Phase2B record.
	 */
	FString SoundWaveAssetPath;

	/** Whether the emotion overlay layer is enabled (written to driver settings on play). */
	bool  bOverlayEnabled = true;

	/** Blend transition duration in seconds (written to driver settings on play). */
	float BlendDuration = 0.4f;

	/** Whether the driver should use API confidence as an intensity multiplier. */
	bool  bUseConfidenceAsWeight = true;

	/** Per-emotion intensity multipliers driven by the slider widgets. */
	TMap<FString, float> EmotionIntensityMultipliers;

	// Phase 2B widget references
	TSharedPtr<STextBlock> MH_TargetStatusText;   // "Bound to: X" or "No actor bound"
	TSharedPtr<STextBlock> MH_FaceMeshStatusText; // "Face mesh: detected / not found"
	TSharedPtr<STextBlock> MH_SoundWaveStatusText;// SoundWave import status
	TSharedPtr<STextBlock> MH_LiveEmotionText;    // current emotion + blend alpha (during play)

	/** Update the MetaHuman target status text widgets after binding/clearing. */
	void UpdateMetaHumanTargetStatusUI();

	/** Update the SoundWave status text after import or take load. */
	void UpdateSoundWaveStatusUI();
};
