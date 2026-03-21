// Copyright (c) EmotionDemo Project. All rights reserved.

#include "SEmotionBridgePanel.h"
#include "EmotionBridgeLog.h"
#include "EmotionBridgeSettings.h"
#include "EmotionLampActor.h"
#include "EmotionColorComponent.h"

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

	SetStatus(
		FString::Printf(TEXT("Analysis complete \u2014 %d segments detected. Click Play Demo."),
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

#undef LOCTEXT_NAMESPACE
