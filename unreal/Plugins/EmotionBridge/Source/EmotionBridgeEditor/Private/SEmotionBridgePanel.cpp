// Copyright (c) EmotionDemo Project. All rights reserved.

#include "SEmotionBridgePanel.h"
#include "EmotionBridgeLog.h"
#include "EmotionBridgeSettings.h"
#include "EmotionLampActor.h"
#include "EmotionColorComponent.h"
// Phase 2A
#include "EmotionTakeStore.h"
#include "SEmotionTakeLibrary.h"
#include "Misc/DateTime.h"
// Phase 2B
#include "MetaHumanEmotionDriverComponent.h"
#include "EmotionAudioAssetHelper.h"

// Slate
#include "Widgets/SBoxPanel.h"
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Layout/SScrollBox.h"
#include "Widgets/Layout/SSeparator.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SEditableTextBox.h"
#include "Widgets/Input/SCheckBox.h"
#include "Widgets/Input/SComboBox.h"
#include "Widgets/Views/SListView.h"
#include "Widgets/Views/STableRow.h"
#include "Widgets/Views/SHeaderRow.h"
#include "Widgets/Colors/SColorBlock.h"
#include "Styling/CoreStyle.h"
#include "Styling/AppStyle.h"

// Editor
#include "Editor.h"
#include "EngineUtils.h"
#include "Selection.h"          // USelection (GEditor->GetSelectedActors)
#include "IDesktopPlatform.h"
#include "DesktopPlatformModule.h"
#include "Framework/Application/SlateApplication.h"
#include "EditorViewportClient.h"
#include "LevelEditorViewport.h"

// Platform
#include "HAL/PlatformProcess.h"
#include "HAL/FileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

#define LOCTEXT_NAMESPACE "SEmotionBridgePanel"

// ============================================================================
// Construct / Destroy
// ============================================================================

void SEmotionBridgePanel::Construct(const FArguments& InArgs)
{
	const UEmotionBridgeSettings* S = UEmotionBridgeSettings::Get();
	const FString DefaultUrl = S ? S->ApiBaseUrl          : TEXT("http://localhost:8000");
	WindowSec        = S ? S->DefaultWindowSec        : 2.0f;
	HopSec           = S ? S->DefaultHopSec           : 0.5f;
	PadMode          = S ? S->DefaultPadMode          : TEXT("none");
	SmoothingMethod  = S ? S->DefaultSmoothingMethod  : TEXT("none");
	HysteresisMinRun = S ? S->DefaultHysteresisMinRun : 3;
	MajorityWindow   = S ? S->DefaultMajorityWindow   : 5;
	EmaAlpha         = S ? S->DefaultEmaAlpha         : 0.6f;

	// Combo options — "none" first so it is the fallback selection
	PadModeOptions.Add(MakeShared<FString>(TEXT("none")));
	PadModeOptions.Add(MakeShared<FString>(TEXT("zero")));
	PadModeOptions.Add(MakeShared<FString>(TEXT("reflect")));
	for (auto& O : PadModeOptions) if (*O == PadMode) SelectedPadMode = O;
	if (!SelectedPadMode.IsValid()) SelectedPadMode = PadModeOptions[0];

	SmoothingOptions.Add(MakeShared<FString>(TEXT("none")));
	SmoothingOptions.Add(MakeShared<FString>(TEXT("hysteresis")));
	SmoothingOptions.Add(MakeShared<FString>(TEXT("majority")));
	SmoothingOptions.Add(MakeShared<FString>(TEXT("ema")));
	for (auto& O : SmoothingOptions) if (*O == SmoothingMethod) SelectedSmoothingMethod = O;
	if (!SelectedSmoothingMethod.IsValid()) SelectedSmoothingMethod = SmoothingOptions[0];

	// Enable Slate tick so the Tick() override is called every rendered frame.
	SetCanTick(true);

	ApiClient = MakeUnique<FEmotionApiClient>(DefaultUrl);

	// Phase 2B: seed per-emotion intensity multipliers at 1.0
	EmotionIntensityMultipliers.Add(TEXT("angry"),   1.0f);
	EmotionIntensityMultipliers.Add(TEXT("happy"),   1.0f);
	EmotionIntensityMultipliers.Add(TEXT("sad"),     1.0f);
	EmotionIntensityMultipliers.Add(TEXT("neutral"), 1.0f);

	ChildSlot
	[
		SNew(SScrollBox)
		+ SScrollBox::Slot().Padding(10.f)
		[
			SNew(SVerticalBox)

			// Header
			+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,4)
			[
				SNew(STextBlock)
				.Text(LOCTEXT("Header", "Emotion Bridge"))
				.Font(FCoreStyle::GetDefaultFontStyle("Bold", 16))
			]
			+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,10)
			[
				SNew(STextBlock)
				.Text(LOCTEXT("SubHeader",
					"Upload a WAV file (or record from mic) \u2192 analyze the emotion timeline "
					"via the local backend \u2192 watch a lamp actor change color in real time."))
				.AutoWrapText(true)
				.ColorAndOpacity(FSlateColor::UseSubduedForeground())
			]

			// Status bar
			+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,8)
			[
				SNew(SBorder)
				.BorderImage(FAppStyle::GetBrush("ToolPanel.GroupBorder"))
				.Padding(FMargin(8.f, 5.f))
				[
					SAssignNew(StatusText, STextBlock)
					.Text(LOCTEXT("Ready", "Ready. Select or record a WAV file, then click Analyze."))
					.AutoWrapText(true)
				]
			]

			+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,6) [ BuildBackendSection() ]
			+ SVerticalBox::Slot().AutoHeight().Padding(0,2)     [ SNew(SSeparator) ]
			+ SVerticalBox::Slot().AutoHeight().Padding(0,6,0,6) [ BuildFileSection() ]
			+ SVerticalBox::Slot().AutoHeight().Padding(0,2)     [ SNew(SSeparator) ]
			+ SVerticalBox::Slot().AutoHeight().Padding(0,6,0,6) [ BuildParametersSection() ]
			+ SVerticalBox::Slot().AutoHeight().Padding(0,2)     [ SNew(SSeparator) ]

			// Analyze button
			+ SVerticalBox::Slot().AutoHeight().Padding(0,8)
			[
				SNew(SButton)
				.HAlign(HAlign_Center)
				.Text(LOCTEXT("Analyze", "Analyze"))
				.ToolTipText(LOCTEXT("AnalyzeTip",
					"POST the WAV file to /timeline/unreal and parse the returned emotion timeline."))
				.IsEnabled_Lambda([this]{ return !bIsAnalyzing && !bIsRecording; })
				.OnClicked(this, &SEmotionBridgePanel::OnAnalyze)
			]

			+ SVerticalBox::Slot().AutoHeight().Padding(0,2)     [ SNew(SSeparator) ]
			+ SVerticalBox::Slot().AutoHeight().Padding(0,6,0,6) [ BuildResultsSection() ]
			+ SVerticalBox::Slot().AutoHeight().Padding(0,2)     [ SNew(SSeparator) ]
			+ SVerticalBox::Slot().AutoHeight().Padding(0,6,0,6) [ BuildPlaybackSection() ]
			+ SVerticalBox::Slot().AutoHeight().Padding(0,2)     [ SNew(SSeparator) ]
			+ SVerticalBox::Slot().AutoHeight().Padding(0,6,0,6) [ BuildSaveTakeSection() ]
			+ SVerticalBox::Slot().AutoHeight().Padding(0,2)     [ SNew(SSeparator) ]
			+ SVerticalBox::Slot().AutoHeight().Padding(0,6,0,8) [ BuildTakeLibrarySection() ]
			+ SVerticalBox::Slot().AutoHeight().Padding(0,2)     [ SNew(SSeparator) ]
			+ SVerticalBox::Slot().AutoHeight().Padding(0,6,0,8) [ BuildMetaHumanSection() ] // Phase 2B
		]
	];
}

SEmotionBridgePanel::~SEmotionBridgePanel()
{
	StopAudioPlayer();

	if (bIsRecording && AudioCapture.IsValid())
	{
		AudioCapture->StopStream();
		AudioCapture->CloseStream();
		AudioCapture.Reset();
	}
}

// ============================================================================
// Slate Tick — drives playback.
// GetCanTick() returns true so Slate calls this every rendered frame.
// Using InDeltaTime accumulation avoids wall-clock capture/race conditions.
// ============================================================================

void SEmotionBridgePanel::Tick(
	const FGeometry& AllottedGeometry, const double InCurrentTime, const float InDeltaTime)
{
	SCompoundWidget::Tick(AllottedGeometry, InCurrentTime, InDeltaTime);

	// Recording elapsed counter (display only, actual samples are captured on audio thread).
	if (bIsRecording)
	{
		RecordingElapsedSec += InDeltaTime;
		SetStatus(
			FString::Printf(TEXT("Recording... %.1f s  (click Stop Recording when done)"),
				RecordingElapsedSec),
			FLinearColor(1.f, 0.4f, 0.4f));
	}

	if (!bIsPlaying)
		return;

	PlaybackElapsedSec += InDeltaTime;

	// End of timeline?
	if (PlaybackElapsedSec >= static_cast<double>(CurrentTimeline.DurationSec))
	{
		bIsPlaying = false;
		StopAudioPlayer();
		// Keep the final color visible; reset label.
		if (EmotionLabelDisplay.IsValid())
			EmotionLabelDisplay->SetText(LOCTEXT("Done", "DONE"));
		SetStatus(TEXT("Playback finished."), FLinearColor::Green);
		UE_LOG(LogEmotionBridge, Log, TEXT("Playback finished at %.2f s."), PlaybackElapsedSec);
		return;
	}

	// Find the active segment at the current elapsed time.
	const float ElapsedF  = static_cast<float>(PlaybackElapsedSec);
	int32 ActiveIndex = -1;
	for (int32 i = 0; i < CurrentTimeline.Segments.Num(); ++i)
	{
		const FEmotionSegment& Seg = CurrentTimeline.Segments[i];
		if (ElapsedF >= Seg.StartSec && ElapsedF < Seg.EndSec)
		{
			ActiveIndex = i;
			break;
		}
	}

	// Only update the lamp when the segment changes (saves redundant GPU work).
	if (ActiveIndex == LastActiveSegmentIndex)
		return;

	LastActiveSegmentIndex = ActiveIndex;

	if (ActiveIndex >= 0)
	{
		const FEmotionSegment& Seg = CurrentTimeline.Segments[ActiveIndex];
		// BroadcastEmotion updates the panel swatch AND all actors in the world.
		BroadcastEmotion(Seg.Emotion, Seg.Confidence);
		SetStatus(
			FString::Printf(TEXT("Playing  t=%.1fs  [ %s ]  conf=%.0f%%"),
				ElapsedF, *Seg.Emotion.ToUpper(), Seg.Confidence * 100.f),
			GetSlateColorForEmotion(Seg.Emotion).GetSpecifiedColor());
	}
	else
	{
		// Gap between segments — neutral.
		BroadcastEmotion(TEXT("neutral"), 1.0f);
	}

	// Phase 2B: update live emotion readout in the MetaHuman section.
	if (MH_LiveEmotionText.IsValid() && BoundDriverComponent.IsValid())
	{
		const FString LiveLabel = FString::Printf(
			TEXT("Active: %s  |  blend: %.0f%%"),
			*BoundDriverComponent->GetCurrentEmotion().ToUpper(),
			BoundDriverComponent->GetBlendAlpha() * 100.f);
		MH_LiveEmotionText->SetText(FText::FromString(LiveLabel));
	}
}

// ============================================================================
// Section builders
// ============================================================================

TSharedRef<SWidget> SEmotionBridgePanel::BuildBackendSection()
{
	const UEmotionBridgeSettings* S = UEmotionBridgeSettings::Get();
	const FString DefaultUrl = S ? S->ApiBaseUrl : TEXT("http://localhost:8000");

	return SNew(SVerticalBox)
		+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,4)
		[
			SNew(STextBlock).Text(LOCTEXT("BackendHdr", "BACKEND"))
				.Font(FCoreStyle::GetDefaultFontStyle("Bold", 10))
		]
		+ SVerticalBox::Slot().AutoHeight()
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0,0,6,0)
			[ SNew(STextBlock).Text(LOCTEXT("URLLabel", "API URL:")) ]
			+ SHorizontalBox::Slot().FillWidth(1.f)
			[
				SAssignNew(ApiUrlBox, SEditableTextBox)
				.Text(FText::FromString(DefaultUrl))
				.HintText(LOCTEXT("URLHint", "http://localhost:8000"))
				.OnTextCommitted_Lambda([this](const FText& T, ETextCommit::Type)
				{ if (ApiClient.IsValid()) ApiClient->SetBaseUrl(T.ToString()); })
			]
			+ SHorizontalBox::Slot().AutoWidth().Padding(6,0,0,0)
			[
				SNew(SButton).Text(LOCTEXT("HealthBtn", "Health Check"))
				.ToolTipText(LOCTEXT("HealthTip", "GET /health to verify the backend is running."))
				.OnClicked(this, &SEmotionBridgePanel::OnHealthCheck)
			]
		];
}

TSharedRef<SWidget> SEmotionBridgePanel::BuildFileSection()
{
	return SNew(SVerticalBox)
		+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,4)
		[
			SNew(STextBlock).Text(LOCTEXT("FileHdr", "AUDIO FILE"))
				.Font(FCoreStyle::GetDefaultFontStyle("Bold", 10))
		]
		// WAV picker
		+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,4)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0,0,6,0)
			[ SNew(STextBlock).Text(LOCTEXT("WavLabel", "WAV File:")) ]
			+ SHorizontalBox::Slot().FillWidth(1.f)
			[
				SAssignNew(WavPathBox, SEditableTextBox)
				.HintText(LOCTEXT("WavHint", "/absolute/path/to/audio.wav"))
				.OnTextCommitted_Lambda([this](const FText& T, ETextCommit::Type)
				{ WavFilePath = T.ToString(); })
			]
			+ SHorizontalBox::Slot().AutoWidth().Padding(6,0,0,0)
			[
				SNew(SButton).Text(LOCTEXT("BrowseBtn", "Browse..."))
				.OnClicked(this, &SEmotionBridgePanel::OnBrowseWav)
			]
		]
		// Microphone recording
		+ SVerticalBox::Slot().AutoHeight()
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot().AutoWidth().Padding(0,0,8,0)
			[
				SNew(SButton)
				.Text(LOCTEXT("RecordBtn", "Record from Mic"))
				.ToolTipText(LOCTEXT("RecordTip",
					"Capture from the default microphone. "
					"Click Stop Recording to save the WAV and auto-populate the path."))
				.IsEnabled_Lambda([this]{ return !bIsRecording && !bIsAnalyzing && !bIsPlaying; })
				.OnClicked(this, &SEmotionBridgePanel::OnRecordStart)
			]
			+ SHorizontalBox::Slot().AutoWidth()
			[
				SNew(SButton)
				.Text(LOCTEXT("StopRecordBtn", "Stop Recording"))
				.IsEnabled_Lambda([this]{ return bIsRecording; })
				.OnClicked(this, &SEmotionBridgePanel::OnRecordStop)
			]
		];
}

TSharedRef<SWidget> SEmotionBridgePanel::BuildParametersSection()
{
	return SNew(SVerticalBox)
		+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,4)
		[
			SNew(STextBlock)
			.Text(LOCTEXT("ParamsHdr", "TIMELINE PARAMETERS  (sent to /timeline/unreal)"))
			.Font(FCoreStyle::GetDefaultFontStyle("Bold", 10))
		]
		// Row 1: window + hop
		+ SVerticalBox::Slot().AutoHeight().Padding(0,2)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0,0,4,0)
			[ SNew(STextBlock).Text(LOCTEXT("WinLbl", "Window (s):")) ]
			+ SHorizontalBox::Slot().MaxWidth(70.f).Padding(0,0,16,0)
			[
				SNew(SEditableTextBox)
				.Text_Lambda([this]{ return FText::FromString(FString::Printf(TEXT("%.2f"),WindowSec)); })
				.OnTextCommitted_Lambda([this](const FText& T, ETextCommit::Type)
				{ WindowSec = FMath::Max(0.1f, FCString::Atof(*T.ToString())); })
			]
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0,0,4,0)
			[ SNew(STextBlock).Text(LOCTEXT("HopLbl", "Hop (s):")) ]
			+ SHorizontalBox::Slot().MaxWidth(70.f)
			[
				SNew(SEditableTextBox)
				.Text_Lambda([this]{ return FText::FromString(FString::Printf(TEXT("%.2f"),HopSec)); })
				.OnTextCommitted_Lambda([this](const FText& T, ETextCommit::Type)
				{ HopSec = FMath::Max(0.05f, FCString::Atof(*T.ToString())); })
			]
		]
		// Row 2: pad_mode
		+ SVerticalBox::Slot().AutoHeight().Padding(0,2)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0,0,4,0)
			[ SNew(STextBlock).Text(LOCTEXT("PadLbl", "Pad Mode:")) ]
			+ SHorizontalBox::Slot().MaxWidth(120.f)
			[
				SNew(SComboBox<TSharedPtr<FString>>)
				.OptionsSource(&PadModeOptions)
				.InitiallySelectedItem(SelectedPadMode)
				.OnGenerateWidget_Lambda([](TSharedPtr<FString> O) -> TSharedRef<SWidget>
				{ return SNew(STextBlock).Text(FText::FromString(*O)); })
				.OnSelectionChanged_Lambda([this](TSharedPtr<FString> Sel, ESelectInfo::Type)
				{ if (Sel.IsValid()) { SelectedPadMode = Sel; PadMode = *Sel; } })
				[
					SNew(STextBlock).Text_Lambda([this]
					{ return FText::FromString(SelectedPadMode.IsValid() ? *SelectedPadMode : PadMode); })
				]
			]
		]
		// Row 3: smoothing method + per-method params
		+ SVerticalBox::Slot().AutoHeight().Padding(0,2)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0,0,4,0)
			[ SNew(STextBlock).Text(LOCTEXT("SmoothLbl", "Smoothing:")) ]
			+ SHorizontalBox::Slot().MaxWidth(140.f).Padding(0,0,12,0)
			[
				SNew(SComboBox<TSharedPtr<FString>>)
				.OptionsSource(&SmoothingOptions)
				.InitiallySelectedItem(SelectedSmoothingMethod)
				.OnGenerateWidget_Lambda([](TSharedPtr<FString> O) -> TSharedRef<SWidget>
				{ return SNew(STextBlock).Text(FText::FromString(*O)); })
				.OnSelectionChanged_Lambda([this](TSharedPtr<FString> Sel, ESelectInfo::Type)
				{ if (Sel.IsValid()) { SelectedSmoothingMethod = Sel; SmoothingMethod = *Sel; } })
				[
					SNew(STextBlock).Text_Lambda([this]
					{ return FText::FromString(SelectedSmoothingMethod.IsValid()
						? *SelectedSmoothingMethod : SmoothingMethod); })
				]
			]
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0,0,4,0)
			[ SNew(STextBlock).Text(LOCTEXT("MinRunLbl","Min Run:"))
				.ToolTipText(LOCTEXT("MinRunTip","hysteresis_min_run (used when Smoothing=hysteresis)")) ]
			+ SHorizontalBox::Slot().MaxWidth(50.f).Padding(0,0,12,0)
			[
				SNew(SEditableTextBox)
				.Text_Lambda([this]{ return FText::FromString(FString::FromInt(HysteresisMinRun)); })
				.OnTextCommitted_Lambda([this](const FText& T, ETextCommit::Type)
				{ HysteresisMinRun = FMath::Max(1, FCString::Atoi(*T.ToString())); })
			]
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0,0,4,0)
			[ SNew(STextBlock).Text(LOCTEXT("MajLbl","Maj Win:"))
				.ToolTipText(LOCTEXT("MajTip","majority_window (used when Smoothing=majority, must be odd)")) ]
			+ SHorizontalBox::Slot().MaxWidth(50.f).Padding(0,0,12,0)
			[
				SNew(SEditableTextBox)
				.Text_Lambda([this]{ return FText::FromString(FString::FromInt(MajorityWindow)); })
				.OnTextCommitted_Lambda([this](const FText& T, ETextCommit::Type)
				{ MajorityWindow = FMath::Max(1, FCString::Atoi(*T.ToString())); })
			]
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0,0,4,0)
			[ SNew(STextBlock).Text(LOCTEXT("EmaLbl","\u03B1:"))
				.ToolTipText(LOCTEXT("EmaTip","ema_alpha  0.01–1.0 (used when Smoothing=ema)")) ]
			+ SHorizontalBox::Slot().MaxWidth(60.f)
			[
				SNew(SEditableTextBox)
				.Text_Lambda([this]{ return FText::FromString(FString::Printf(TEXT("%.2f"),EmaAlpha)); })
				.OnTextCommitted_Lambda([this](const FText& T, ETextCommit::Type)
				{ EmaAlpha = FMath::Clamp(FCString::Atof(*T.ToString()), 0.01f, 1.f); })
			]
		];
}

TSharedRef<SWidget> SEmotionBridgePanel::BuildResultsSection()
{
	TSharedRef<SHeaderRow> Header = SNew(SHeaderRow)
		+ SHeaderRow::Column(TEXT("Start"))  .DefaultLabel(LOCTEXT("ColS","Start (s)")).FillWidth(1.f)
		+ SHeaderRow::Column(TEXT("End"))    .DefaultLabel(LOCTEXT("ColE","End (s)"))  .FillWidth(1.f)
		+ SHeaderRow::Column(TEXT("Emotion")).DefaultLabel(LOCTEXT("ColM","Emotion"))  .FillWidth(2.f)
		+ SHeaderRow::Column(TEXT("Conf"))   .DefaultLabel(LOCTEXT("ColC","Conf %"))   .FillWidth(1.5f);

	return SNew(SVerticalBox)
		+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,4)
		[
			SNew(STextBlock).Text(LOCTEXT("ResultsHdr","RESULTS"))
				.Font(FCoreStyle::GetDefaultFontStyle("Bold", 10))
		]
		+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,4)
		[
			SAssignNew(MetadataText, STextBlock)
			.Text(LOCTEXT("NoResults","No results yet. Run Analyze."))
			.AutoWrapText(true)
			.ColorAndOpacity(FSlateColor::UseSubduedForeground())
		]
		+ SVerticalBox::Slot().MaxHeight(200.f)
		[
			SAssignNew(SegmentListView, SListView<TSharedPtr<FEmotionSegmentRow>>)
			.ListItemsSource(&SegmentRows)
			.OnGenerateRow(this, &SEmotionBridgePanel::GenerateSegmentRow)
			.HeaderRow(Header)
			.SelectionMode(ESelectionMode::Single)
		];
}

TSharedRef<SWidget> SEmotionBridgePanel::BuildPlaybackSection()
{
	return SNew(SVerticalBox)
		+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,4)
		[
			SNew(STextBlock).Text(LOCTEXT("PlayHdr","PLAYBACK DEMO"))
				.Font(FCoreStyle::GetDefaultFontStyle("Bold", 10))
		]
		+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,4)
		[
			SNew(STextBlock)
			.Text(LOCTEXT("PlayDesc",
				"Click Play Demo after Analyze. The color block below changes with each "
				"emotion segment and the WAV plays simultaneously. No level or actor needed."))
			.AutoWrapText(true)
			.ColorAndOpacity(FSlateColor::UseSubduedForeground())
		]

		// ---- Emotion color swatch ----------------------------------------
		// Works immediately — no level, no actor, no asset setup required.
		// The swatch shows the active emotion color; the label shows the name.
		+ SVerticalBox::Slot().AutoHeight().Padding(0,6,0,8)
		[
			SNew(SHorizontalBox)
			// Color block
			+ SHorizontalBox::Slot().AutoWidth().Padding(0,0,12,0)
			[
				SNew(SBox).WidthOverride(100.f).HeightOverride(100.f)
				[
					SNew(SColorBlock)
					.Color(TAttribute<FLinearColor>::CreateLambda(
						[this]{ return CurrentDisplayColor; }))
					.ShowBackgroundForAlpha(false)
					.AlphaDisplayMode(EColorBlockAlphaDisplayMode::Ignore)
				]
			]
			// Emotion label next to the swatch
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center)
			[
				SNew(SVerticalBox)
				+ SVerticalBox::Slot().AutoHeight()
				[
					SAssignNew(EmotionLabelDisplay, STextBlock)
					.Text(LOCTEXT("SwatchDefault", "\u2014"))
					.Font(FCoreStyle::GetDefaultFontStyle("Bold", 28))
				]
				+ SVerticalBox::Slot().AutoHeight()
				[
					SNew(STextBlock)
					.Text(LOCTEXT("SwatchSub", "current emotion"))
					.ColorAndOpacity(FSlateColor::UseSubduedForeground())
				]
			]
		]

		// ---- Play / Stop / Focus buttons --------------------------------
		+ SVerticalBox::Slot().AutoHeight()
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot().AutoWidth().Padding(0,0,8,0)
			[
				SNew(SButton).Text(LOCTEXT("PlayBtn","Play Demo"))
				.ToolTipText(LOCTEXT("PlayTip","Start playback. Run Analyze first."))
				.IsEnabled_Lambda([this]{ return !bIsPlaying && CurrentTimeline.bIsValid; })
				.OnClicked(this, &SEmotionBridgePanel::OnPlayDemo)
			]
			+ SHorizontalBox::Slot().AutoWidth().Padding(0,0,8,0)
			[
				SNew(SButton).Text(LOCTEXT("StopBtn","Stop Demo"))
				.IsEnabled_Lambda([this]{ return bIsPlaying; })
				.OnClicked(this, &SEmotionBridgePanel::OnStopDemo)
			]
			+ SHorizontalBox::Slot().AutoWidth()
			[
				SNew(SButton)
				.Text(LOCTEXT("FocusBtn", "Focus / Spawn in Viewport"))
				.ToolTipText(LOCTEXT("FocusTip",
					"Select and focus the viewport on the emotion target actor. "
					"If none exists, spawns an EmotionLampActor at the world origin. "
					"Add an Emotion Color component to any actor to use your own character."))
				.OnClicked(this, &SEmotionBridgePanel::OnFocusViewport)
			]
		]
		// ---- How to link your own actor ----------------------------------
		+ SVerticalBox::Slot().AutoHeight().Padding(0,6,0,0)
		[
			SNew(STextBlock)
			.Text(LOCTEXT("LinkHint",
				"To link any actor: select it in the viewport \u2192 Details \u2192 "
				"\u201c+ Add Component\u201d \u2192 search \u201cEmotion Color\u201d. "
				"Done \u2014 it will respond to playback automatically."))
			.AutoWrapText(true)
			.ColorAndOpacity(FSlateColor::UseSubduedForeground())
		];
}

// ============================================================================
// Button callbacks
// ============================================================================

FReply SEmotionBridgePanel::OnHealthCheck()
{
	if (ApiUrlBox.IsValid() && ApiClient.IsValid())
		ApiClient->SetBaseUrl(ApiUrlBox->GetText().ToString());
	SetStatus(TEXT("Checking /health ..."), FLinearColor(1.f, 0.85f, 0.f));
	ApiClient->CheckHealth(
		FOnHealthCheckComplete::CreateSP(this, &SEmotionBridgePanel::OnHealthCheckResult));
	return FReply::Handled();
}

FReply SEmotionBridgePanel::OnBrowseWav()
{
	IDesktopPlatform* DP = FDesktopPlatformModule::Get();
	if (!DP) return FReply::Handled();

	TArray<FString> Files;
	const void* ParentWin = FSlateApplication::Get().GetActiveTopLevelWindow().IsValid()
		? FSlateApplication::Get().GetActiveTopLevelWindow()->GetNativeWindow()->GetOSWindowHandle()
		: nullptr;

	if (DP->OpenFileDialog(ParentWin, TEXT("Select WAV File"),
		FPaths::ProjectDir(), TEXT(""),
		TEXT("WAV Audio (*.wav)|*.wav|All Files (*.*)|*.*"),
		EFileDialogFlags::None, Files) && Files.Num() > 0)
	{
		WavFilePath = Files[0];
		if (WavPathBox.IsValid()) WavPathBox->SetText(FText::FromString(WavFilePath));
	}
	return FReply::Handled();
}

FReply SEmotionBridgePanel::OnRecordStart()
{
	if (bIsRecording) return FReply::Handled();

	AudioCapture       = MakeUnique<Audio::FAudioCapture>();
	RecordedSamples.Empty();
	RecordingElapsedSec = 0.0;
	CapturedSampleRate  = 16000;
	CapturedNumChannels = 1;

	Audio::FAudioCaptureDeviceParams Params;
	const bool bOpened = AudioCapture->OpenAudioCaptureStream(
		Params,
		[this](const void* InAudio, int32 NumFrames, int32 NumCh,
			int32 SR, double, bool)
		{
			// Audio thread callback — use a lock.
			FScopeLock Lock(&RecordingLock);
			CapturedSampleRate  = SR;
			CapturedNumChannels = NumCh;

			const float* Data = static_cast<const float*>(InAudio);
			const int32 NumSamples = NumFrames * NumCh;
			RecordedSamples.Append(Data, NumSamples);
		},
		1024);

	if (!bOpened)
	{
		SetStatus(
			TEXT("Could not open audio capture device. "
				 "Check microphone permissions in System Preferences > Privacy."),
			FLinearColor::Red);
		AudioCapture.Reset();
		return FReply::Handled();
	}

	AudioCapture->StartStream();
	bIsRecording = true;
	SetStatus(TEXT("Recording... click Stop Recording when done."), FLinearColor(1.f,0.4f,0.4f));
	UE_LOG(LogEmotionBridge, Log, TEXT("Microphone recording started."));
	return FReply::Handled();
}

FReply SEmotionBridgePanel::OnRecordStop()
{
	if (!bIsRecording) return FReply::Handled();

	AudioCapture->StopStream();
	AudioCapture->CloseStream();
	bIsRecording        = false;
	RecordingElapsedSec = 0.0;

	const FString Path = WriteRecordingToWav();
	if (!Path.IsEmpty())
	{
		WavFilePath = Path;
		if (WavPathBox.IsValid()) WavPathBox->SetText(FText::FromString(WavFilePath));
		SetStatus(
			FString::Printf(TEXT("Saved recording (%d samples @ %d Hz). Click Analyze."),
				RecordedSamples.Num(), CapturedSampleRate),
			FLinearColor::Green);
	}
	else
	{
		SetStatus(TEXT("Recording stopped but WAV write failed."), FLinearColor::Red);
	}
	AudioCapture.Reset();
	UE_LOG(LogEmotionBridge, Log, TEXT("Microphone recording stopped."));
	return FReply::Handled();
}

FReply SEmotionBridgePanel::OnAnalyze()
{
	if (ApiUrlBox.IsValid() && ApiClient.IsValid())
		ApiClient->SetBaseUrl(ApiUrlBox->GetText().ToString());
	if (WavPathBox.IsValid())
		WavFilePath = WavPathBox->GetText().ToString();

	if (WavFilePath.IsEmpty())
	{
		SetStatus(TEXT("Select or record a WAV file first."), FLinearColor(1,0.5f,0));
		return FReply::Handled();
	}
	if (!FPaths::FileExists(WavFilePath))
	{
		SetStatus(FString::Printf(TEXT("File not found: %s"), *WavFilePath), FLinearColor::Red);
		return FReply::Handled();
	}
	if (FPaths::GetExtension(WavFilePath).ToLower() != TEXT("wav"))
	{
		SetStatus(TEXT("Only .wav files are supported."), FLinearColor::Red);
		return FReply::Handled();
	}

	bIsAnalyzing = true;
	CurrentTimeline = FEmotionTimelineResponse{};
	RefreshSegmentList();
	SetStatus(
		TEXT("Analyzing via /timeline/unreal... "
			 "(first run: 30-120 s while the model downloads from HuggingFace)"),
		FLinearColor(1.f, 0.85f, 0.f));

	ApiClient->RequestTimeline(
		WavFilePath,
		WindowSec, HopSec,
		PadMode, SmoothingMethod, HysteresisMinRun,
		static_cast<float>(MajorityWindow), EmaAlpha,
		FOnTimelineComplete::CreateSP(this, &SEmotionBridgePanel::OnTimelineReceived));

	return FReply::Handled();
}

FReply SEmotionBridgePanel::OnPlayDemo()
{
	if (!CurrentTimeline.bIsValid || CurrentTimeline.Segments.Num() == 0)
	{
		SetStatus(TEXT("No valid timeline. Run Analyze first."), FLinearColor(1,0.5f,0));
		return FReply::Handled();
	}

	// Lamp is optional — the color swatch in the panel always works.
	// If a level is open with an AEmotionLampActor, it will also change color.
	AEmotionLampActor* Lamp = FindOrSpawnLampActor(); // may be nullptr
	LampActorRef           = Lamp;
	bIsPlaying             = true;
	PlaybackElapsedSec     = 0.0;
	LastActiveSegmentIndex = -1;

	// Apply the first segment immediately — no blank first frame.
	if (CurrentTimeline.Segments.Num() > 0)
	{
		const FEmotionSegment& First = CurrentTimeline.Segments[0];
		BroadcastEmotion(First.Emotion, First.Confidence);
		LastActiveSegmentIndex = 0;
	}

	// Phase 2B: configure emotion driver if a MetaHuman actor is bound.
	if (BoundDriverComponent.IsValid())
	{
		FEmotionOverlaySettings NewSettings  = BoundDriverComponent->OverlaySettings;
		NewSettings.bEnabled                 = bOverlayEnabled;
		NewSettings.BlendDurationSec         = BlendDuration;
		NewSettings.bUseConfidenceAsWeight   = bUseConfidenceAsWeight;
		NewSettings.EmotionIntensityMultipliers = EmotionIntensityMultipliers;
		BoundDriverComponent->OverlaySettings = NewSettings;
		BoundDriverComponent->ResetToNeutral();
		UE_LOG(LogEmotionBridge, Log,
			TEXT("Phase2B: configured emotion driver on '%s' (blend=%.2fs, overlay=%s)."),
			*BoundActorLabel, BlendDuration, bOverlayEnabled ? TEXT("on") : TEXT("off"));
	}

	LaunchAudioPlayer();

	SetStatus(
		FString::Printf(TEXT("Playing — %.2f s, %d segments. Watch the lamp in the viewport."),
			CurrentTimeline.DurationSec, CurrentTimeline.Segments.Num()),
		FLinearColor::Green);

	UE_LOG(LogEmotionBridge, Log,
		TEXT("Demo playback started on actor '%s'."), *Lamp->GetName());
	return FReply::Handled();
}

FReply SEmotionBridgePanel::OnStopDemo()
{
	bIsPlaying             = false;
	PlaybackElapsedSec     = 0.0;
	LastActiveSegmentIndex = -1;
	CurrentDisplayColor    = FLinearColor::White;
	StopAudioPlayer();

	BroadcastEmotion(TEXT("neutral"), 1.0f);
	if (EmotionLabelDisplay.IsValid())
		EmotionLabelDisplay->SetText(LOCTEXT("SwatchDefault", "\u2014"));

	// Phase 2B: reset MetaHuman face to neutral.
	if (BoundDriverComponent.IsValid())
	{
		BoundDriverComponent->ResetToNeutral();
	}
	if (MH_LiveEmotionText.IsValid())
		MH_LiveEmotionText->SetText(LOCTEXT("MHLiveDefault", "\u2014"));

	SetStatus(TEXT("Playback stopped."), FLinearColor::White);
	UE_LOG(LogEmotionBridge, Log, TEXT("Demo playback stopped by user."));
	return FReply::Handled();
}

// ============================================================================
// HTTP callbacks
// ============================================================================

void SEmotionBridgePanel::OnTimelineReceived(const FEmotionTimelineResponse& Response)
{
	bIsAnalyzing = false;

	if (!Response.bIsValid)
	{
		SetStatus(FString::Printf(TEXT("Error: %s"), *Response.ErrorMessage), FLinearColor::Red);
		PendingReanalyzeTakeId.Empty();
		return;
	}

	CurrentTimeline = Response;
	RefreshSegmentList();

	const FString Meta = FString::Printf(
		TEXT("type=%s  source=%s  v=%s  duration=%.2f s  segments=%d"),
		*Response.Type, *Response.Source, *Response.Version,
		Response.DurationSec, Response.Segments.Num());
	if (MetadataText.IsValid())
		MetadataText->SetText(FText::FromString(Meta));

	// ---- Phase 2A: reanalysis in-place ----------------------------------------
	if (!PendingReanalyzeTakeId.IsEmpty())
	{
		FEmotionTakeRecord Existing;
		if (FEmotionTakeStore::LoadTake(PendingReanalyzeTakeId, Existing))
		{
			// Replace analysis data; keep identity/annotations.
			Existing.Timeline   = Response;
			Existing.DurationSec = Response.DurationSec;
			Existing.SampleRate  = Response.SampleRate;
			Existing.Params.WindowSec        = WindowSec;
			Existing.Params.HopSec           = HopSec;
			Existing.Params.PadMode          = PadMode;
			Existing.Params.SmoothingMethod  = SmoothingMethod;
			Existing.Params.HysteresisMinRun = HysteresisMinRun;
			Existing.Params.MajorityWindow   = MajorityWindow;
			Existing.Params.EmaAlpha         = EmaAlpha;
			Existing.UpdatedAt = FDateTime::UtcNow().ToIso8601();

			if (FEmotionTakeStore::SaveTake(Existing, /*bCopyAudio=*/false))
			{
				SetStatus(
					FString::Printf(
						TEXT("Reanalysis complete \u2014 take \u201c%s\u201d updated (%d segments)."),
						*Existing.DisplayName, Response.Segments.Num()),
					FLinearColor::Green);
			}
			else
			{
				SetStatus(
					FString::Printf(
						TEXT("Reanalysis complete but save failed for take \u201c%s\u201d."),
						*Existing.DisplayName),
					FLinearColor(1.f, 0.5f, 0.f));
			}

			if (TakeLibraryWidget.IsValid()) TakeLibraryWidget->RefreshLibrary();
		}
		else
		{
			UE_LOG(LogEmotionBridge, Warning,
				TEXT("OnTimelineReceived: could not load take '%s' for in-place update."),
				*PendingReanalyzeTakeId);
			SetStatus(
				FString::Printf(TEXT("Analysis complete \u2014 %d segments (take update failed, check log)."),
					Response.Segments.Num()),
				FLinearColor(1.f, 0.5f, 0.f));
		}

		PendingReanalyzeTakeId.Empty();
		bCanSaveTake = false; // already saved — do not prompt again
		return;
	}
	// ---------------------------------------------------------------------------

	// Normal (new) analysis complete — let the user save as a new take.
	bCanSaveTake = true;
	SetStatus(
		FString::Printf(
			TEXT("Analysis complete \u2014 %d segments detected. "
			     "Click Play Demo or Save Take below."),
			Response.Segments.Num()),
		FLinearColor::Green);
}

void SEmotionBridgePanel::OnHealthCheckResult(bool bHealthy)
{
	if (bHealthy)
		SetStatus(TEXT("Backend healthy (HTTP 200 /health)."), FLinearColor::Green);
	else
		SetStatus(
			TEXT("Backend not reachable. Run:  docker compose up api"),
			FLinearColor::Red);
}

// ============================================================================
// Helpers
// ============================================================================

// ---------------------------------------------------------------------------
// BroadcastEmotion — apply to AEmotionLampActor + all UEmotionColorComponents
// ---------------------------------------------------------------------------

void SEmotionBridgePanel::BroadcastEmotion(const FString& Emotion, float Confidence)
{
	// 1. Update the in-panel swatch.
	CurrentDisplayColor = GetSlateColorForEmotion(Emotion).GetSpecifiedColor();
	if (EmotionLabelDisplay.IsValid())
		EmotionLabelDisplay->SetText(FText::FromString(Emotion.ToUpper()));

	// Helper: broadcast to one world.
	auto BroadcastToWorld = [&](UWorld* World)
	{
		if (!World) return;
		for (TActorIterator<AActor> It(World); It; ++It)
		{
			if (UEmotionColorComponent* Comp =
				It->FindComponentByClass<UEmotionColorComponent>())
			{
				Comp->ApplyEmotion(Emotion, Confidence);
			}
		}
	};

	if (GEditor)
	{
		// 2. Editor world (visible in viewport without PIE).
		BroadcastToWorld(GEditor->GetEditorWorldContext().World());

		// 3. PIE worlds (if the game is running).
		if (GEngine)
		{
			for (const FWorldContext& Ctx : GEngine->GetWorldContexts())
				if (Ctx.WorldType == EWorldType::PIE)
					BroadcastToWorld(Ctx.World());
		}
	}

	// 4. AEmotionLampActor (legacy — kept for backwards compatibility).
	if (LampActorRef.IsValid())
		LampActorRef->ApplyEmotion(Emotion, Confidence);

	// 5. Phase 2B: drive MetaHuman face via emotion driver component.
	if (BoundDriverComponent.IsValid())
		BoundDriverComponent->ApplyEmotion(Emotion, Confidence);
}

// ---------------------------------------------------------------------------
// OnFocusViewport — select target actor and move camera to it
// ---------------------------------------------------------------------------

FReply SEmotionBridgePanel::OnFocusViewport()
{
	if (!GEditor) return FReply::Handled();

	UWorld* World = GEditor->GetEditorWorldContext().World();
	if (!World) return FReply::Handled();

	// Find the first actor that has UEmotionColorComponent or is an EmotionLampActor.
	AActor* Target = nullptr;
	for (TActorIterator<AActor> It(World); It; ++It)
	{
		if (It->FindComponentByClass<UEmotionColorComponent>()
			|| Cast<AEmotionLampActor>(*It))
		{
			Target = *It;
			break;
		}
	}

	if (!Target)
	{
		// Nothing found — spawn a lamp so the user has something to look at.
		FActorSpawnParameters SP;
		SP.Name = FName("EmotionLamp_AutoSpawned");
		SP.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
		Target = World->SpawnActor<AEmotionLampActor>(
			AEmotionLampActor::StaticClass(),
			FVector(0.f, 0.f, 150.f), FRotator::ZeroRotator, SP);

		if (Target)
		{
			LampActorRef = Cast<AEmotionLampActor>(Target);
			SetStatus(TEXT("Spawned EmotionLampActor. See it selected in the viewport."),
				FLinearColor::Green);
		}
		else
		{
			SetStatus(TEXT("Could not spawn lamp — open a level first (File > New Level)."),
				FLinearColor::Red);
			return FReply::Handled();
		}
	}

	// Select the actor in the editor.
	GEditor->SelectNone(false, true);
	GEditor->SelectActor(Target, true, true);
	GEditor->NoteSelectionChange();

	// Move the viewport camera to frame the actor.
	for (FLevelEditorViewportClient* Client : GEditor->GetLevelViewportClients())
	{
		if (Client && Client->IsPerspective())
		{
			Client->FocusViewportOnBox(Target->GetComponentsBoundingBox(true));
			break;
		}
	}

	SetStatus(
		FString::Printf(TEXT("Focused on: %s"), *Target->GetName()),
		FLinearColor::White);
	return FReply::Handled();
}

// ---------------------------------------------------------------------------

void SEmotionBridgePanel::SetStatus(const FString& Message, FLinearColor Color)
{
	if (StatusText.IsValid())
	{
		StatusText->SetText(FText::FromString(Message));
		StatusText->SetColorAndOpacity(FSlateColor(Color));
	}
}

AEmotionLampActor* SEmotionBridgePanel::FindOrSpawnLampActor()
{
	if (!GEditor) return nullptr;

	// Helper: search a world for the first AEmotionLampActor.
	auto Search = [](UWorld* W) -> AEmotionLampActor*
	{
		if (!W) return nullptr;
		for (TActorIterator<AEmotionLampActor> It(W); It; ++It)
			return *It;
		return nullptr;
	};

	// 1. Check editor world.
	UWorld* EditorWorld = GEditor->GetEditorWorldContext().World();
	if (AEmotionLampActor* Found = Search(EditorWorld)) return Found;

	// 2. Check any active PIE worlds.
	if (GEngine)
	{
		for (const FWorldContext& Ctx : GEngine->GetWorldContexts())
		{
			if (Ctx.WorldType == EWorldType::PIE)
				if (AEmotionLampActor* Found = Search(Ctx.World())) return Found;
		}
	}

	// 3. Spawn in the editor world (auto-spawn).
	if (!EditorWorld)
	{
		UE_LOG(LogEmotionBridge, Warning,
			TEXT("FindOrSpawnLampActor: no editor world available."));
		return nullptr;
	}

	FActorSpawnParameters SP;
	SP.Name = FName("EmotionLamp_AutoSpawned");
	SP.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

	AEmotionLampActor* Lamp = EditorWorld->SpawnActor<AEmotionLampActor>(
		AEmotionLampActor::StaticClass(),
		FVector(0.f, 0.f, 100.f), FRotator::ZeroRotator, SP);

	// UE_LOG expands to a compound statement { ... }, so it must be inside braces
	// when used in a braceless if/else — otherwise the trailing ; orphans the else.
	if (Lamp)
	{
		UE_LOG(LogEmotionBridge, Log, TEXT("Auto-spawned AEmotionLampActor."));
	}
	else
	{
		UE_LOG(LogEmotionBridge, Error, TEXT("SpawnActor<AEmotionLampActor> failed."));
	}

	return Lamp;
}

void SEmotionBridgePanel::LaunchAudioPlayer()
{
	if (WavFilePath.IsEmpty() || !FPaths::FileExists(WavFilePath))
		return;

	StopAudioPlayer();

#if PLATFORM_MAC
	// afplay is shipped with macOS — no install required.
	const FString Args = FString::Printf(TEXT("\"%s\""), *WavFilePath);
	AudioPlayerHandle  = FPlatformProcess::CreateProc(
		TEXT("/usr/bin/afplay"), *Args,
		true, true, true, nullptr, 0, nullptr, nullptr);

#elif PLATFORM_WINDOWS
	// Use PowerShell's built-in SoundPlayer for a no-dependency approach.
	// Escape single-quotes in the path for PowerShell.
	const FString SafePath = WavFilePath.Replace(TEXT("'"), TEXT("''"));
	const FString Cmd = FString::Printf(
		TEXT("-NoProfile -NonInteractive -Command "
			 "\"$p=New-Object System.Media.SoundPlayer '%s'; $p.PlaySync()\""),
		*SafePath);
	AudioPlayerHandle = FPlatformProcess::CreateProc(
		TEXT("powershell.exe"), *Cmd,
		false, true, true, nullptr, 0, nullptr, nullptr);

#else
	UE_LOG(LogEmotionBridge, Warning,
		TEXT("LaunchAudioPlayer: not implemented on this platform."));
#endif

	UE_LOG(LogEmotionBridge, Log, TEXT("Audio player launched: %s"), *WavFilePath);
}

void SEmotionBridgePanel::StopAudioPlayer()
{
	if (FPlatformProcess::IsProcRunning(AudioPlayerHandle))
	{
		FPlatformProcess::TerminateProc(AudioPlayerHandle);
		FPlatformProcess::CloseProc(AudioPlayerHandle);
		UE_LOG(LogEmotionBridge, Log, TEXT("Audio player process terminated."));
	}
	AudioPlayerHandle = FProcHandle();
}

FString SEmotionBridgePanel::WriteRecordingToWav()
{
	FScopeLock Lock(&RecordingLock);
	if (RecordedSamples.IsEmpty()) return {};

	const int32 SR  = CapturedSampleRate  > 0 ? CapturedSampleRate  : 16000;
	const int32 NCh = CapturedNumChannels > 0 ? CapturedNumChannels : 1;

	// Convert float → signed 16-bit PCM.
	TArray<int16> PCM;
	PCM.SetNumUninitialized(RecordedSamples.Num());
	for (int32 i = 0; i < RecordedSamples.Num(); ++i)
	{
		PCM[i] = static_cast<int16>(
			FMath::Clamp(RecordedSamples[i], -1.f, 1.f) * 32767.f);
	}

	const int32 DataBytes = PCM.Num() * 2;

	// Minimal WAV header (44 bytes, PCM format).
	struct FWAVHeader
	{
		char     RIFF[4]       = {'R','I','F','F'};
		uint32   ChunkSize     = 0;
		char     WAVE[4]       = {'W','A','V','E'};
		char     fmt[4]        = {'f','m','t',' '};
		uint32   SubChunk1Size = 16;
		uint16   AudioFormat   = 1; // PCM
		uint16   NumChannels   = 1;
		uint32   SampleRate    = 16000;
		uint32   ByteRate      = 0;
		uint16   BlockAlign    = 0;
		uint16   BitsPerSample = 16;
		char     data[4]       = {'d','a','t','a'};
		uint32   SubChunk2Size = 0;
	};

	FWAVHeader H;
	H.NumChannels   = static_cast<uint16>(NCh);
	H.SampleRate    = static_cast<uint32>(SR);
	H.ByteRate      = static_cast<uint32>(SR * NCh * 2);
	H.BlockAlign    = static_cast<uint16>(NCh * 2);
	H.SubChunk2Size = static_cast<uint32>(DataBytes);
	H.ChunkSize     = 36 + H.SubChunk2Size;

	TArray<uint8> FileData;
	FileData.Append(reinterpret_cast<const uint8*>(&H), sizeof(FWAVHeader));
	FileData.Append(reinterpret_cast<const uint8*>(PCM.GetData()), DataBytes);

	const FString TempPath = FPaths::Combine(
		FPlatformProcess::UserTempDir(),
		TEXT("EmotionBridge_Recording.wav"));

	if (!FFileHelper::SaveArrayToFile(FileData, *TempPath))
	{
		UE_LOG(LogEmotionBridge, Error,
			TEXT("WriteRecordingToWav: failed to write '%s'"), *TempPath);
		return {};
	}

	UE_LOG(LogEmotionBridge, Log,
		TEXT("Recording saved: %s  (%d samples, %d ch, %d Hz)"),
		*TempPath, PCM.Num(), NCh, SR);
	return TempPath;
}

void SEmotionBridgePanel::RefreshSegmentList()
{
	SegmentRows.Empty();
	for (const FEmotionSegment& Seg : CurrentTimeline.Segments)
		SegmentRows.Add(MakeShared<FEmotionSegmentRow>(Seg));
	if (SegmentListView.IsValid())
		SegmentListView->RequestListRefresh();
}

TSharedRef<ITableRow> SEmotionBridgePanel::GenerateSegmentRow(
	TSharedPtr<FEmotionSegmentRow> Item,
	const TSharedRef<STableViewBase>& OwnerTable)
{
	const FLinearColor C = GetSlateColorForEmotion(Item->Segment.Emotion).GetSpecifiedColor();

	return SNew(STableRow<TSharedPtr<FEmotionSegmentRow>>, OwnerTable)
		.Padding(FMargin(4.f, 2.f))
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot().FillWidth(1.f)
			[ SNew(STextBlock).Text(FText::FromString(
				FString::Printf(TEXT("%.2f"), Item->Segment.StartSec))) ]
			+ SHorizontalBox::Slot().FillWidth(1.f)
			[ SNew(STextBlock).Text(FText::FromString(
				FString::Printf(TEXT("%.2f"), Item->Segment.EndSec))) ]
			+ SHorizontalBox::Slot().FillWidth(2.f)
			[
				SNew(STextBlock)
				.Text(FText::FromString(Item->Segment.Emotion))
				.ColorAndOpacity(FSlateColor(C))
				.Font(FCoreStyle::GetDefaultFontStyle("Bold", 10))
			]
			+ SHorizontalBox::Slot().FillWidth(1.5f)
			[ SNew(STextBlock).Text(FText::FromString(
				FString::Printf(TEXT("%.1f%%"), Item->Segment.Confidence * 100.f))) ]
		];
}

FSlateColor SEmotionBridgePanel::GetSlateColorForEmotion(const FString& Emotion) const
{
	const UEmotionBridgeSettings* S = UEmotionBridgeSettings::Get();
	if (S) return FSlateColor(S->GetColorForEmotion(Emotion));
	return FSlateColor::UseForeground();
}

// ============================================================================
// Phase 2A — Save Take section builder
// ============================================================================

TSharedRef<SWidget> SEmotionBridgePanel::BuildSaveTakeSection()
{
	return SNew(SVerticalBox)
		+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,4)
		[
			SNew(STextBlock)
			.Text(LOCTEXT("SaveTakeHdr", "SAVE TAKE"))
			.Font(FCoreStyle::GetDefaultFontStyle("Bold", 10))
		]
		+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,4)
		[
			SNew(STextBlock)
			.Text(LOCTEXT("SaveTakeDesc",
				"After a successful analysis, save it as a named Take. "
				"Takes are stored in Saved/EmotionBridge/Takes/ and can be "
				"reloaded later without re-running the backend."))
			.AutoWrapText(true)
			.ColorAndOpacity(FSlateColor::UseSubduedForeground())
		]
		+ SVerticalBox::Slot().AutoHeight()
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0,0,6,0)
			[ SNew(STextBlock).Text(LOCTEXT("TakeNameLbl", "Take Name:")) ]
			+ SHorizontalBox::Slot().FillWidth(1.f)
			[
				SAssignNew(TakeNameBox, SEditableTextBox)
				.HintText(LOCTEXT("TakeNameHint",
					"e.g. Actor02_angry_take3  (leave blank for auto name)"))
				.IsEnabled_Lambda([this]{ return bCanSaveTake; })
			]
			+ SHorizontalBox::Slot().AutoWidth().Padding(6,0,0,0)
			[
				SNew(SButton)
				.Text(LOCTEXT("SaveTakeBtn", "Save Take"))
				.ToolTipText(LOCTEXT("SaveTakeTip",
					"Persist the current analysis as a named Take.\n"
					"Saves timeline.json, params.json, metadata.json, and a copy of "
					"the source WAV to Saved/EmotionBridge/Takes/<TakeId>/"))
				.IsEnabled_Lambda([this]{ return bCanSaveTake && CurrentTimeline.bIsValid; })
				.OnClicked(this, &SEmotionBridgePanel::OnSaveTakeClicked)
			]
		];
}

FReply SEmotionBridgePanel::OnSaveTakeClicked()
{
	if (!CurrentTimeline.bIsValid)
	{
		SetStatus(TEXT("Nothing to save — run Analyze first."), FLinearColor(1,0.5f,0));
		return FReply::Handled();
	}

	FString TakeName = TakeNameBox.IsValid() ? TakeNameBox->GetText().ToString().TrimStartAndEnd() : FString{};
	if (TakeName.IsEmpty())
		TakeName = FEmotionTakeStore::GenerateDefaultDisplayName();

	FEmotionTakeRecord Record;
	Record.TakeId        = FEmotionTakeStore::GenerateTakeId();
	Record.DisplayName   = TakeName;
	Record.CreatedAt     = FDateTime::UtcNow().ToIso8601();
	Record.UpdatedAt     = Record.CreatedAt;
	Record.SourceAudioPath = WavFilePath;
	Record.DurationSec   = CurrentTimeline.DurationSec;
	Record.SampleRate    = CurrentTimeline.SampleRate;
	Record.Timeline      = CurrentTimeline;

	Record.Params.WindowSec        = WindowSec;
	Record.Params.HopSec           = HopSec;
	Record.Params.PadMode          = PadMode;
	Record.Params.SmoothingMethod  = SmoothingMethod;
	Record.Params.HysteresisMinRun = HysteresisMinRun;
	Record.Params.MajorityWindow   = MajorityWindow;
	Record.Params.EmaAlpha         = EmaAlpha;

	// Phase 2B: persist MetaHuman binding state.
	Record.Phase2B.SoundWaveAssetPath     = SoundWaveAssetPath;
	Record.Phase2B.BoundActorLabel        = BoundActorLabel;
	Record.Phase2B.OverlayBlendDurationSec = BlendDuration;
	Record.Phase2B.bOverlayEnabled        = bOverlayEnabled;

	if (FEmotionTakeStore::SaveTake(Record, /*bCopyAudio=*/true))
	{
		SetStatus(
			FString::Printf(TEXT("Take \u201c%s\u201d saved  (%s)."),
				*TakeName, *Record.TakeId.Left(8)),
			FLinearColor::Green);

		// Reset the name box so typing a fresh name is easy next time.
		if (TakeNameBox.IsValid()) TakeNameBox->SetText(FText::GetEmpty());
		bCanSaveTake = false;

		if (TakeLibraryWidget.IsValid())
		{
			TakeLibraryWidget->RefreshLibrary();
			TakeLibraryWidget->SelectTakeById(Record.TakeId);
		}
	}
	else
	{
		SetStatus(TEXT("Take save failed — check Output Log for details."), FLinearColor::Red);
	}

	return FReply::Handled();
}

// ============================================================================
// Phase 2A — Take Library section builder
// ============================================================================

TSharedRef<SWidget> SEmotionBridgePanel::BuildTakeLibrarySection()
{
	SAssignNew(TakeLibraryWidget, SEmotionTakeLibrary)
		.OnLoadRequested_Lambda([this](const FEmotionTakeRecord& T){ OnLoadTakeRequested(T); })
		.OnPlayRequested_Lambda([this](const FEmotionTakeRecord& T){ OnPlayTakeRequested(T); })
		.OnReanalyzeRequested_Lambda([this](const FEmotionTakeRecord& T){ OnReanalyzeTakeRequested(T); });

	return TakeLibraryWidget.ToSharedRef();
}

// ============================================================================
// Phase 2A — Take Library action handlers
// ============================================================================

void SEmotionBridgePanel::OnLoadTakeRequested(const FEmotionTakeRecord& Take)
{
	// Restore timeline.
	CurrentTimeline = Take.Timeline;

	// Restore the best available audio path.
	const FString BestAudio = Take.GetBestAudioPath();
	if (!BestAudio.IsEmpty())
	{
		WavFilePath = BestAudio;
		if (WavPathBox.IsValid())
			WavPathBox->SetText(FText::FromString(WavFilePath));
	}

	// Restore analysis parameters into panel state.
	WindowSec        = Take.Params.WindowSec;
	HopSec           = Take.Params.HopSec;
	PadMode          = Take.Params.PadMode;
	SmoothingMethod  = Take.Params.SmoothingMethod;
	HysteresisMinRun = Take.Params.HysteresisMinRun;
	MajorityWindow   = Take.Params.MajorityWindow;
	EmaAlpha         = Take.Params.EmaAlpha;

	// Sync combo selections to the restored parameter values.
	for (auto& O : PadModeOptions)
		if (*O == PadMode) { SelectedPadMode = O; break; }
	for (auto& O : SmoothingOptions)
		if (*O == SmoothingMethod) { SelectedSmoothingMethod = O; break; }

	// Update results area.
	RefreshSegmentList();

	const FString Meta = FString::Printf(
		TEXT("Loaded take \u201c%s\u201d  |  duration=%.2f s  segments=%d  model=%s"),
		*Take.DisplayName, Take.DurationSec, Take.Timeline.Segments.Num(),
		*Take.Timeline.ModelName);
	if (MetadataText.IsValid())
		MetadataText->SetText(FText::FromString(Meta));

	// A loaded take is already saved — don't prompt to save again.
	bCanSaveTake = false;

	// Phase 2B: restore SoundWave path and overlay settings.
	if (!Take.Phase2B.SoundWaveAssetPath.IsEmpty())
	{
		SoundWaveAssetPath = Take.Phase2B.SoundWaveAssetPath;
		UpdateSoundWaveStatusUI();
	}
	if (!Take.Phase2B.BoundActorLabel.IsEmpty())
	{
		// Inform the user which actor was bound — they may need to rebind manually.
		UE_LOG(LogEmotionBridge, Log,
			TEXT("Phase2B: take was saved with actor '%s'. Rebind via 'Bind Selected Actor' if needed."),
			*Take.Phase2B.BoundActorLabel);
	}
	BlendDuration    = Take.Phase2B.OverlayBlendDurationSec > 0.f
		? Take.Phase2B.OverlayBlendDurationSec : BlendDuration;
	bOverlayEnabled  = Take.Phase2B.bOverlayEnabled;

	const FString BoundNote = Take.Phase2B.BoundActorLabel.IsEmpty()
		? FString{}
		: FString::Printf(TEXT(" (was bound to '%s')"), *Take.Phase2B.BoundActorLabel);

	SetStatus(
		FString::Printf(TEXT("Take \u201c%s\u201d loaded. Click Play Demo to replay.%s"),
			*Take.DisplayName, *BoundNote),
		FLinearColor::Green);
}

void SEmotionBridgePanel::OnPlayTakeRequested(const FEmotionTakeRecord& Take)
{
	OnLoadTakeRequested(Take);
	// OnPlayDemo() is a FReply callback — call it and discard the returned value.
	(void)OnPlayDemo();
}

void SEmotionBridgePanel::OnReanalyzeTakeRequested(const FEmotionTakeRecord& Take)
{
	const FString BestAudio = Take.GetBestAudioPath();
	if (BestAudio.IsEmpty())
	{
		SetStatus(
			FString::Printf(
				TEXT("Reanalyze failed: no audio available for take \u201c%s\u201d."),
				*Take.DisplayName),
			FLinearColor::Red);
		return;
	}

	// Restore params from the take so the same settings are used.
	WavFilePath      = BestAudio;
	WindowSec        = Take.Params.WindowSec;
	HopSec           = Take.Params.HopSec;
	PadMode          = Take.Params.PadMode;
	SmoothingMethod  = Take.Params.SmoothingMethod;
	HysteresisMinRun = Take.Params.HysteresisMinRun;
	MajorityWindow   = Take.Params.MajorityWindow;
	EmaAlpha         = Take.Params.EmaAlpha;

	if (WavPathBox.IsValid())
		WavPathBox->SetText(FText::FromString(WavFilePath));

	// Tag this run so OnTimelineReceived() knows to do an in-place update.
	PendingReanalyzeTakeId = Take.TakeId;

	bIsAnalyzing = true;
	CurrentTimeline = FEmotionTimelineResponse{};
	RefreshSegmentList();
	SetStatus(
		FString::Printf(TEXT("Reanalysing take \u201c%s\u201d via /timeline/unreal..."),
			*Take.DisplayName),
		FLinearColor(1.f, 0.85f, 0.f));

	ApiClient->RequestTimeline(
		WavFilePath,
		WindowSec, HopSec,
		PadMode, SmoothingMethod, HysteresisMinRun,
		static_cast<float>(MajorityWindow), EmaAlpha,
		FOnTimelineComplete::CreateSP(this, &SEmotionBridgePanel::OnTimelineReceived));
}

// ============================================================================
// Phase 2B — MetaHuman section builder
// ============================================================================

TSharedRef<SWidget> SEmotionBridgePanel::BuildMetaHumanSection()
{
	// Helpers for row labels with fixed width
	auto RowLabel = [](FText Label) -> TSharedRef<SWidget>
	{
		return SNew(SBox).WidthOverride(130.f).VAlign(VAlign_Center)
		[
			SNew(STextBlock).Text(Label)
		];
	};

	return SNew(SVerticalBox)

	// ── Header ──────────────────────────────────────────────────────────────
	+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,4)
	[
		SNew(STextBlock).Text(LOCTEXT("MHHdr", "METAHUMAN FACE  (Phase 2B)"))
			.Font(FCoreStyle::GetDefaultFontStyle("Bold", 10))
	]
	+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,6)
	[
		SNew(STextBlock)
		.Text(LOCTEXT("MHDesc",
			"Bind a MetaHuman Actor to layer API-driven emotion onto its face during "
			"playback.  The Emotion Overlay adds expression to the upper face (brow, "
			"cheeks, eyes) without interfering with the base speech animation.  "
			"See docs/METAHUMAN_PHASE2B.md for full setup instructions."))
		.AutoWrapText(true)
		.ColorAndOpacity(FSlateColor::UseSubduedForeground())
	]

	// ── TARGET BINDING ──────────────────────────────────────────────────────
	+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,2)
	[
		SNew(STextBlock).Text(LOCTEXT("MHTargetSubHdr", "— TARGET —"))
			.Font(FCoreStyle::GetDefaultFontStyle("Bold", 9))
			.ColorAndOpacity(FSlateColor::UseSubduedForeground())
	]
	+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,4)
	[
		SNew(SHorizontalBox)
		+ SHorizontalBox::Slot().AutoWidth().Padding(0,0,6,0)
		[
			SNew(SButton)
			.Text(LOCTEXT("MHBindBtn", "Bind Selected Actor"))
			.ToolTipText(LOCTEXT("MHBindTip",
				"Select a MetaHuman Actor in the viewport, then click here.\n"
				"A UMetaHumanEmotionDriverComponent is auto-added if not present.\n"
				"The component drives morph targets during Play Demo."))
			.OnClicked(this, &SEmotionBridgePanel::OnBindSelectedActor)
		]
		+ SHorizontalBox::Slot().AutoWidth().Padding(0,0,6,0)
		[
			SNew(SButton)
			.Text(LOCTEXT("MHClearBtn", "Clear"))
			.ToolTipText(LOCTEXT("MHClearTip",
				"Unbind the current actor and reset its face to neutral."))
			.IsEnabled_Lambda([this]{ return BoundDriverComponent.IsValid(); })
			.OnClicked(this, &SEmotionBridgePanel::OnClearMetaHumanBinding)
		]
	]
	+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,2)
	[
		SNew(SHorizontalBox)
		+ SHorizontalBox::Slot().AutoWidth()[ RowLabel(LOCTEXT("MHBoundLbl", "Bound actor:")) ]
		+ SHorizontalBox::Slot().FillWidth(1.f).VAlign(VAlign_Center)
		[
			SAssignNew(MH_TargetStatusText, STextBlock)
			.Text(LOCTEXT("MHNoActor", "No actor bound — select a MetaHuman and click Bind."))
			.AutoWrapText(true)
			.ColorAndOpacity(FSlateColor::UseSubduedForeground())
		]
	]
	+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,6)
	[
		SNew(SHorizontalBox)
		+ SHorizontalBox::Slot().AutoWidth()[ RowLabel(LOCTEXT("MHFaceLbl", "Face mesh:")) ]
		+ SHorizontalBox::Slot().FillWidth(1.f).VAlign(VAlign_Center)
		[
			SAssignNew(MH_FaceMeshStatusText, STextBlock)
			.Text(LOCTEXT("MHNoFace", "\u2014"))
			.ColorAndOpacity(FSlateColor::UseSubduedForeground())
		]
	]

	// ── AUDIO ASSET ─────────────────────────────────────────────────────────
	+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,2)
	[
		SNew(STextBlock).Text(LOCTEXT("MHAudioSubHdr", "— AUDIO ASSET —"))
			.Font(FCoreStyle::GetDefaultFontStyle("Bold", 9))
			.ColorAndOpacity(FSlateColor::UseSubduedForeground())
	]
	+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,2)
	[
		SNew(SHorizontalBox)
		+ SHorizontalBox::Slot().AutoWidth()
		[ RowLabel(LOCTEXT("MHSwLbl", "SoundWave:")) ]
		+ SHorizontalBox::Slot().FillWidth(1.f).VAlign(VAlign_Center)
		[
			SAssignNew(MH_SoundWaveStatusText, STextBlock)
			.Text(LOCTEXT("MHNoSW", "Not imported yet."))
			.AutoWrapText(true)
			.ColorAndOpacity(FSlateColor::UseSubduedForeground())
		]
		+ SHorizontalBox::Slot().AutoWidth().Padding(6,0,0,0)
		[
			SNew(SButton)
			.Text(LOCTEXT("MHImportBtn", "Import WAV as SoundWave"))
			.ToolTipText(LOCTEXT("MHImportTip",
				"Import the current WAV file into the content browser as a SoundWave asset\n"
				"at /Game/EmotionBridge/Audio/.\n"
				"Required for MetaHuman audio-driven facial animation.\n"
				"Select a WAV file first (Audio File section above)."))
			.IsEnabled_Lambda([this]
			{
				return !WavFilePath.IsEmpty() && FPaths::FileExists(WavFilePath);
			})
			.OnClicked(this, &SEmotionBridgePanel::OnImportSoundWave)
		]
	]

	// ── EMOTION OVERLAY ──────────────────────────────────────────────────────
	+ SVerticalBox::Slot().AutoHeight().Padding(0,4,0,2)
	[
		SNew(STextBlock).Text(LOCTEXT("MHOverlaySubHdr", "— EMOTION OVERLAY —"))
			.Font(FCoreStyle::GetDefaultFontStyle("Bold", 9))
			.ColorAndOpacity(FSlateColor::UseSubduedForeground())
	]
	// Overlay enabled toggle
	+ SVerticalBox::Slot().AutoHeight().Padding(0,2)
	[
		SNew(SHorizontalBox)
		+ SHorizontalBox::Slot().AutoWidth().Padding(0,0,8,0)
		[
			SNew(SCheckBox)
			.IsChecked_Lambda([this]{ return bOverlayEnabled
				? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
			.OnCheckStateChanged_Lambda([this](ECheckBoxState S)
			{ bOverlayEnabled = (S == ECheckBoxState::Checked); })
		]
		+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center)
		[ SNew(STextBlock).Text(LOCTEXT("MHEnableOverlay", "Enable emotion overlay layer")) ]
	]
	// Blend duration
	+ SVerticalBox::Slot().AutoHeight().Padding(0,2)
	[
		SNew(SHorizontalBox)
		+ SHorizontalBox::Slot().AutoWidth()[ RowLabel(LOCTEXT("MHBlendLbl", "Blend duration (s):")) ]
		+ SHorizontalBox::Slot().MaxWidth(80.f)
		[
			SNew(SEditableTextBox)
			.Text_Lambda([this]
			{ return FText::FromString(FString::Printf(TEXT("%.2f"), BlendDuration)); })
			.ToolTipText(LOCTEXT("MHBlendTip",
				"Crossfade duration between emotion segments.  0.4 s is a good default."))
			.OnTextCommitted_Lambda([this](const FText& T, ETextCommit::Type)
			{ BlendDuration = FMath::Clamp(FCString::Atof(*T.ToString()), 0.f, 5.f); })
		]
	]
	// Confidence weighting toggle
	+ SVerticalBox::Slot().AutoHeight().Padding(0,2)
	[
		SNew(SHorizontalBox)
		+ SHorizontalBox::Slot().AutoWidth().Padding(0,0,8,0)
		[
			SNew(SCheckBox)
			.IsChecked_Lambda([this]{ return bUseConfidenceAsWeight
				? ECheckBoxState::Checked : ECheckBoxState::Unchecked; })
			.OnCheckStateChanged_Lambda([this](ECheckBoxState S)
			{ bUseConfidenceAsWeight = (S == ECheckBoxState::Checked); })
		]
		+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center)
		[ SNew(STextBlock).Text(LOCTEXT("MHConfWeight",
			"Use API confidence as intensity weight")) ]
	]

	// ── EMOTION INTENSITY ────────────────────────────────────────────────────
	+ SVerticalBox::Slot().AutoHeight().Padding(0,4,0,2)
	[
		SNew(STextBlock).Text(LOCTEXT("MHIntensitySubHdr", "— EMOTION INTENSITY MULTIPLIERS —"))
			.Font(FCoreStyle::GetDefaultFontStyle("Bold", 9))
			.ColorAndOpacity(FSlateColor::UseSubduedForeground())
	]
	+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,2)
	[
		SNew(STextBlock)
		.Text(LOCTEXT("MHIntensityDesc",
			"Scale each emotion's expression strength [0 = off, 1 = normal, 2 = exaggerated]."))
		.ColorAndOpacity(FSlateColor::UseSubduedForeground())
		.AutoWrapText(true)
	]

	// Four emotion sliders — one per canonical emotion.
	+ SVerticalBox::Slot().AutoHeight().Padding(0,2)
	[
		SNew(SHorizontalBox)
		+ SHorizontalBox::Slot().AutoWidth()
		[ SNew(SBox).WidthOverride(60.f).VAlign(VAlign_Center)
		  [ SNew(STextBlock).Text(LOCTEXT("MHAngryLbl","angry:"))
			.ColorAndOpacity(FSlateColor(FLinearColor(1.f,0.15f,0.15f))) ] ]
		+ SHorizontalBox::Slot().MaxWidth(80.f)
		[
			SNew(SEditableTextBox)
			.Text_Lambda([this]{ const float* V = EmotionIntensityMultipliers.Find(TEXT("angry"));
				return FText::FromString(FString::Printf(TEXT("%.2f"), V ? *V : 1.f)); })
			.OnTextCommitted_Lambda([this](const FText& T, ETextCommit::Type)
			{ EmotionIntensityMultipliers.Add(TEXT("angry"),
				FMath::Clamp(FCString::Atof(*T.ToString()),0.f,2.f)); })
		]
		+ SHorizontalBox::Slot().AutoWidth().Padding(12,0,0,0)
		[ SNew(SBox).WidthOverride(60.f).VAlign(VAlign_Center)
		  [ SNew(STextBlock).Text(LOCTEXT("MHHappyLbl","happy:"))
			.ColorAndOpacity(FSlateColor(FLinearColor(1.f,0.85f,0.1f))) ] ]
		+ SHorizontalBox::Slot().MaxWidth(80.f)
		[
			SNew(SEditableTextBox)
			.Text_Lambda([this]{ const float* V = EmotionIntensityMultipliers.Find(TEXT("happy"));
				return FText::FromString(FString::Printf(TEXT("%.2f"), V ? *V : 1.f)); })
			.OnTextCommitted_Lambda([this](const FText& T, ETextCommit::Type)
			{ EmotionIntensityMultipliers.Add(TEXT("happy"),
				FMath::Clamp(FCString::Atof(*T.ToString()),0.f,2.f)); })
		]
		+ SHorizontalBox::Slot().AutoWidth().Padding(12,0,0,0)
		[ SNew(SBox).WidthOverride(40.f).VAlign(VAlign_Center)
		  [ SNew(STextBlock).Text(LOCTEXT("MHSadLbl","sad:"))
			.ColorAndOpacity(FSlateColor(FLinearColor(0.25f,0.45f,1.f))) ] ]
		+ SHorizontalBox::Slot().MaxWidth(80.f)
		[
			SNew(SEditableTextBox)
			.Text_Lambda([this]{ const float* V = EmotionIntensityMultipliers.Find(TEXT("sad"));
				return FText::FromString(FString::Printf(TEXT("%.2f"), V ? *V : 1.f)); })
			.OnTextCommitted_Lambda([this](const FText& T, ETextCommit::Type)
			{ EmotionIntensityMultipliers.Add(TEXT("sad"),
				FMath::Clamp(FCString::Atof(*T.ToString()),0.f,2.f)); })
		]
		+ SHorizontalBox::Slot().AutoWidth().Padding(12,0,0,0)
		[ SNew(SBox).WidthOverride(65.f).VAlign(VAlign_Center)
		  [ SNew(STextBlock).Text(LOCTEXT("MHNeutralLbl","neutral:")) ] ]
		+ SHorizontalBox::Slot().MaxWidth(80.f)
		[
			SNew(SEditableTextBox)
			.Text_Lambda([this]{ const float* V = EmotionIntensityMultipliers.Find(TEXT("neutral"));
				return FText::FromString(FString::Printf(TEXT("%.2f"), V ? *V : 1.f)); })
			.OnTextCommitted_Lambda([this](const FText& T, ETextCommit::Type)
			{ EmotionIntensityMultipliers.Add(TEXT("neutral"),
				FMath::Clamp(FCString::Atof(*T.ToString()),0.f,2.f)); })
		]
	]

	// ── LIVE PLAYBACK STATUS ─────────────────────────────────────────────────
	+ SVerticalBox::Slot().AutoHeight().Padding(0,6,0,2)
	[
		SNew(STextBlock).Text(LOCTEXT("MHLiveSubHdr", "— CURRENT STATE —"))
			.Font(FCoreStyle::GetDefaultFontStyle("Bold", 9))
			.ColorAndOpacity(FSlateColor::UseSubduedForeground())
	]
	+ SVerticalBox::Slot().AutoHeight()
	[
		SNew(SHorizontalBox)
		+ SHorizontalBox::Slot().AutoWidth()[ RowLabel(LOCTEXT("MHLiveLbl", "Emotion driver:")) ]
		+ SHorizontalBox::Slot().FillWidth(1.f).VAlign(VAlign_Center)
		[
			SAssignNew(MH_LiveEmotionText, STextBlock)
			.Text(LOCTEXT("MHLiveDefault", "\u2014"))
			.ColorAndOpacity(FSlateColor::UseSubduedForeground())
		]
	];
}

// ============================================================================
// Phase 2B — MetaHuman action callbacks
// ============================================================================

FReply SEmotionBridgePanel::OnBindSelectedActor()
{
	if (!GEditor)
	{
		SetStatus(TEXT("GEditor not available."), FLinearColor::Red);
		return FReply::Handled();
	}

	// Get the first selected actor.
	AActor* TargetActor = nullptr;
	{
		USelection* Sel = GEditor->GetSelectedActors();
		if (!Sel || Sel->Num() == 0)
		{
			SetStatus(
				TEXT("No actor selected. Select a MetaHuman actor in the viewport first."),
				FLinearColor(1.f, 0.5f, 0.f));
			return FReply::Handled();
		}
		TargetActor = Cast<AActor>(Sel->GetSelectedObject(0));
	}

	if (!TargetActor)
	{
		SetStatus(TEXT("Selected object is not an Actor."), FLinearColor(1.f, 0.5f, 0.f));
		return FReply::Handled();
	}

	// Validate — must have at least one SkeletalMeshComponent.
	TArray<USkeletalMeshComponent*> SkelComps;
	TargetActor->GetComponents<USkeletalMeshComponent>(SkelComps);
	if (SkelComps.IsEmpty())
	{
		SetStatus(
			FString::Printf(TEXT("'%s' has no SkeletalMeshComponent. "
				"Please select a MetaHuman actor."),
				*TargetActor->GetActorLabel()),
			FLinearColor::Red);
		return FReply::Handled();
	}

	// Get or add UMetaHumanEmotionDriverComponent.
	UMetaHumanEmotionDriverComponent* DriverComp =
		TargetActor->FindComponentByClass<UMetaHumanEmotionDriverComponent>();

	if (!DriverComp)
	{
		// Add as a transient instance component (session-only, not saved with the level).
		DriverComp = NewObject<UMetaHumanEmotionDriverComponent>(
			TargetActor,
			UMetaHumanEmotionDriverComponent::StaticClass(),
			NAME_None,
			RF_Transient);
		TargetActor->AddInstanceComponent(DriverComp);
		DriverComp->RegisterComponent();
		UE_LOG(LogEmotionBridge, Log,
			TEXT("Phase2B: added UMetaHumanEmotionDriverComponent (transient) to '%s'."),
			*TargetActor->GetActorLabel());
	}

	BoundMetaHumanActor = TargetActor;
	BoundDriverComponent = DriverComp;
	BoundActorLabel      = TargetActor->GetActorLabel();

	UpdateMetaHumanTargetStatusUI();

	const bool bHasFace = DriverComp->HasValidFaceMesh();
	SetStatus(
		FString::Printf(TEXT("Bound to '%s'. Face mesh: %s"),
			*BoundActorLabel,
			bHasFace
				? TEXT("detected.")
				: TEXT("not yet detected — will auto-detect on first Play Demo tick.")),
		FLinearColor::Green);

	return FReply::Handled();
}

FReply SEmotionBridgePanel::OnClearMetaHumanBinding()
{
	if (BoundDriverComponent.IsValid())
	{
		BoundDriverComponent->ResetToNeutral();
	}
	BoundMetaHumanActor.Reset();
	BoundDriverComponent.Reset();
	BoundActorLabel.Empty();

	UpdateMetaHumanTargetStatusUI();
	if (MH_LiveEmotionText.IsValid())
		MH_LiveEmotionText->SetText(LOCTEXT("MHLiveDefault", "\u2014"));

	SetStatus(TEXT("MetaHuman binding cleared."), FLinearColor::White);
	return FReply::Handled();
}

FReply SEmotionBridgePanel::OnImportSoundWave()
{
	if (WavFilePath.IsEmpty() || !FPaths::FileExists(WavFilePath))
	{
		SetStatus(
			TEXT("Select a WAV file first (Audio File section above)."),
			FLinearColor(1.f, 0.5f, 0.f));
		return FReply::Handled();
	}

	SetStatus(TEXT("Importing WAV as SoundWave..."), FLinearColor(1.f, 0.85f, 0.f));

	FString ErrorMessage;
	const FString ImportedPath = FEmotionAudioAssetHelper::ImportWavAsSoundWave(
		WavFilePath,
		FEmotionAudioAssetHelper::GetDefaultAudioContentPath(),
		ErrorMessage);

	if (ImportedPath.IsEmpty())
	{
		SetStatus(
			FString::Printf(TEXT("SoundWave import failed: %s"), *ErrorMessage),
			FLinearColor::Red);
		return FReply::Handled();
	}

	SoundWaveAssetPath = ImportedPath;
	UpdateSoundWaveStatusUI();

	SetStatus(
		FString::Printf(TEXT("SoundWave imported: %s"), *ImportedPath),
		FLinearColor::Green);

	UE_LOG(LogEmotionBridge, Log,
		TEXT("Phase2B: SoundWave asset → '%s'"), *ImportedPath);
	return FReply::Handled();
}

// ============================================================================
// Phase 2B — UI update helpers
// ============================================================================

void SEmotionBridgePanel::UpdateMetaHumanTargetStatusUI()
{
	if (MH_TargetStatusText.IsValid())
	{
		MH_TargetStatusText->SetText(FText::FromString(
			BoundActorLabel.IsEmpty()
				? TEXT("No actor bound — select a MetaHuman and click Bind.")
				: FString::Printf(TEXT("Bound: %s"), *BoundActorLabel)));
		MH_TargetStatusText->SetColorAndOpacity(BoundActorLabel.IsEmpty()
			? FSlateColor::UseSubduedForeground()
			: FSlateColor(FLinearColor::Green));
	}

	if (MH_FaceMeshStatusText.IsValid())
	{
		FString FaceStatus = TEXT("\u2014");
		if (BoundDriverComponent.IsValid())
		{
			FaceStatus = BoundDriverComponent->HasValidFaceMesh()
				? TEXT("Detected")
				: TEXT("Will auto-detect on first tick");
		}
		MH_FaceMeshStatusText->SetText(FText::FromString(FaceStatus));
	}
}

void SEmotionBridgePanel::UpdateSoundWaveStatusUI()
{
	if (!MH_SoundWaveStatusText.IsValid()) return;

	if (SoundWaveAssetPath.IsEmpty())
	{
		MH_SoundWaveStatusText->SetText(
			LOCTEXT("MHNoSW", "Not imported yet."));
		MH_SoundWaveStatusText->SetColorAndOpacity(FSlateColor::UseSubduedForeground());
	}
	else
	{
		MH_SoundWaveStatusText->SetText(FText::FromString(SoundWaveAssetPath));
		MH_SoundWaveStatusText->SetColorAndOpacity(FSlateColor(FLinearColor::Green));
	}
}

#undef LOCTEXT_NAMESPACE
