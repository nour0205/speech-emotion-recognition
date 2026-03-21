// Copyright (c) EmotionDemo Project. All rights reserved.

#include "SEmotionBridgePanel.h"
#include "EmotionBridgeLog.h"
#include "EmotionBridgeSettings.h"
#include "EmotionLampActor.h"

// Slate
#include "Widgets/SBoxPanel.h"
#include "Widgets/SOverlay.h"
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
#include "Styling/CoreStyle.h"
#include "Styling/AppStyle.h"

// Editor
#include "Editor.h"
#include "EngineUtils.h"
#include "IDesktopPlatform.h"
#include "DesktopPlatformModule.h"
#include "Framework/Application/SlateApplication.h"

// Platform
#include "HAL/PlatformTime.h"

#define LOCTEXT_NAMESPACE "SEmotionBridgePanel"

// ---------------------------------------------------------------------------
// Column IDs
// ---------------------------------------------------------------------------
const FName SEmotionBridgePanel::ColStart      = TEXT("Start");
const FName SEmotionBridgePanel::ColEnd        = TEXT("End");
const FName SEmotionBridgePanel::ColEmotion    = TEXT("Emotion");
const FName SEmotionBridgePanel::ColConfidence = TEXT("Confidence");

// ---------------------------------------------------------------------------
// Construct
// ---------------------------------------------------------------------------

void SEmotionBridgePanel::Construct(const FArguments& InArgs)
{
	// Initialize from project settings defaults.
	const UEmotionBridgeSettings* Settings = UEmotionBridgeSettings::Get();
	const FString DefaultUrl = Settings ? Settings->ApiBaseUrl               : TEXT("http://localhost:8000");
	WindowSec        = Settings ? Settings->DefaultWindowSec        : 2.0f;
	HopSec           = Settings ? Settings->DefaultHopSec           : 0.5f;
	PadMode          = Settings ? Settings->DefaultPadMode          : TEXT("zero");
	SmoothingMethod  = Settings ? Settings->DefaultSmoothingMethod  : TEXT("hysteresis");
	HysteresisMinRun = Settings ? Settings->DefaultHysteresisMinRun : 3;

	// Build combo-box option lists.
	PadModeOptions.Add(MakeShared<FString>(TEXT("zero")));
	PadModeOptions.Add(MakeShared<FString>(TEXT("reflect")));
	PadModeOptions.Add(MakeShared<FString>(TEXT("edge")));
	for (TSharedPtr<FString>& Opt : PadModeOptions)
		if (*Opt == PadMode) SelectedPadMode = Opt;
	if (!SelectedPadMode.IsValid()) SelectedPadMode = PadModeOptions[0];

	SmoothingOptions.Add(MakeShared<FString>(TEXT("hysteresis")));
	SmoothingOptions.Add(MakeShared<FString>(TEXT("none")));
	SmoothingOptions.Add(MakeShared<FString>(TEXT("majority_vote")));
	for (TSharedPtr<FString>& Opt : SmoothingOptions)
		if (*Opt == SmoothingMethod) SelectedSmoothingMethod = Opt;
	if (!SelectedSmoothingMethod.IsValid()) SelectedSmoothingMethod = SmoothingOptions[0];

	// Create the HTTP client.
	ApiClient = MakeUnique<FEmotionApiClient>(DefaultUrl);

	// -----------------------------------------------------------------------
	// Full widget tree
	// -----------------------------------------------------------------------
	ChildSlot
	[
		SNew(SScrollBox)
		+ SScrollBox::Slot().Padding(8.f)
		[
			SNew(SVerticalBox)

			// ---- Header ----
			+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 0.f, 0.f, 6.f)
			[
				SNew(STextBlock)
				.Text(LOCTEXT("Header", "Emotion Bridge"))
				.Font(FCoreStyle::GetDefaultFontStyle("Bold", 16))
			]
			+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 0.f, 0.f, 8.f)
			[
				SNew(STextBlock)
				.Text(LOCTEXT("SubHeader", "Connect to the local speech-emotion-recognition backend, analyze a WAV file, and drive actor color changes from the returned emotion timeline."))
				.AutoWrapText(true)
				.ColorAndOpacity(FSlateColor::UseSubduedForeground())
			]

			// ---- Status bar ----
			+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 0.f, 0.f, 8.f)
			[
				SAssignNew(StatusBorder, SBorder)
				.BorderImage(FAppStyle::GetBrush("ToolPanel.GroupBorder"))
				.Padding(FMargin(8.f, 4.f))
				[
					SAssignNew(StatusText, STextBlock)
					.Text(LOCTEXT("StatusReady", "Ready. Select a WAV file and click Analyze."))
					.AutoWrapText(true)
				]
			]

			// ---- Backend section ----
			+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 0.f, 0.f, 6.f)
			[
				BuildBackendSection()
			]

			+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 2.f)
			[
				SNew(SSeparator)
			]

			// ---- File section ----
			+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 6.f, 0.f, 6.f)
			[
				BuildFileSection()
			]

			+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 2.f)
			[
				SNew(SSeparator)
			]

			// ---- Parameters section ----
			+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 6.f, 0.f, 6.f)
			[
				BuildParametersSection()
			]

			+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 2.f)
			[
				SNew(SSeparator)
			]

			// ---- Analyze button ----
			+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 8.f)
			[
				SNew(SHorizontalBox)
				+ SHorizontalBox::Slot().FillWidth(1.f)
				[
					SNew(SButton)
					.HAlign(HAlign_Center)
					.Text(LOCTEXT("AnalyzeBtn", "Analyze"))
					.ToolTipText(LOCTEXT("AnalyzeTip", "Upload the WAV file to the /timeline endpoint and parse the response."))
					.IsEnabled_Lambda([this]() { return !bIsAnalyzing; })
					.OnClicked(this, &SEmotionBridgePanel::OnAnalyze)
				]
			]

			+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 2.f)
			[
				SNew(SSeparator)
			]

			// ---- Results section ----
			+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 6.f, 0.f, 6.f)
			[
				BuildResultsSection()
			]

			+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 2.f)
			[
				SNew(SSeparator)
			]

			// ---- Playback section ----
			+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 6.f, 0.f, 6.f)
			[
				BuildPlaybackSection()
			]
		]
	];
}

SEmotionBridgePanel::~SEmotionBridgePanel()
{
	// Remove the playback ticker if it is still running.
	if (PlaybackTickerHandle.IsValid())
	{
		FTSTicker::GetCoreTicker().RemoveTicker(PlaybackTickerHandle);
		PlaybackTickerHandle = {};
	}
}

// ---------------------------------------------------------------------------
// Section builders
// ---------------------------------------------------------------------------

TSharedRef<SWidget> SEmotionBridgePanel::BuildBackendSection()
{
	const UEmotionBridgeSettings* Settings = UEmotionBridgeSettings::Get();
	const FString DefaultUrl = Settings ? Settings->ApiBaseUrl : TEXT("http://localhost:8000");

	return SNew(SVerticalBox)
		+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 0.f, 0.f, 4.f)
		[
			SNew(STextBlock)
			.Text(LOCTEXT("BackendLabel", "BACKEND"))
			.Font(FCoreStyle::GetDefaultFontStyle("Bold", 10))
		]
		+ SVerticalBox::Slot().AutoHeight()
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0.f, 0.f, 6.f, 0.f)
			[
				SNew(STextBlock).Text(LOCTEXT("ApiUrlLabel", "API URL:"))
			]
			+ SHorizontalBox::Slot().FillWidth(1.f)
			[
				SAssignNew(ApiUrlBox, SEditableTextBox)
				.Text(FText::FromString(DefaultUrl))
				.HintText(LOCTEXT("ApiUrlHint", "http://localhost:8000"))
				.ToolTipText(LOCTEXT("ApiUrlTip",
					"Base URL of the speech-emotion-recognition backend. "
					"Change this if you run the backend on a different port."))
				.OnTextCommitted_Lambda([this](const FText& NewText, ETextCommit::Type)
				{
					if (ApiClient.IsValid())
						ApiClient->SetBaseUrl(NewText.ToString());
				})
			]
			+ SHorizontalBox::Slot().AutoWidth().Padding(6.f, 0.f, 0.f, 0.f)
			[
				SNew(SButton)
				.Text(LOCTEXT("HealthBtn", "Health Check"))
				.ToolTipText(LOCTEXT("HealthTip", "Send GET /health to verify the backend is reachable."))
				.OnClicked(this, &SEmotionBridgePanel::OnHealthCheck)
			]
		];
}

TSharedRef<SWidget> SEmotionBridgePanel::BuildFileSection()
{
	return SNew(SVerticalBox)
		+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 0.f, 0.f, 4.f)
		[
			SNew(STextBlock)
			.Text(LOCTEXT("FileLabel", "AUDIO FILE"))
			.Font(FCoreStyle::GetDefaultFontStyle("Bold", 10))
		]
		+ SVerticalBox::Slot().AutoHeight()
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0.f, 0.f, 6.f, 0.f)
			[
				SNew(STextBlock).Text(LOCTEXT("WavLabel", "WAV File:"))
			]
			+ SHorizontalBox::Slot().FillWidth(1.f)
			[
				SAssignNew(WavPathBox, SEditableTextBox)
				.HintText(LOCTEXT("WavHint", "/path/to/audio.wav"))
				.ToolTipText(LOCTEXT("WavTip",
					"Absolute path to the WAV file to analyze. "
					"Use Browse or paste a path directly."))
				.OnTextCommitted_Lambda([this](const FText& NewText, ETextCommit::Type)
				{
					WavFilePath = NewText.ToString();
				})
			]
			+ SHorizontalBox::Slot().AutoWidth().Padding(6.f, 0.f, 0.f, 0.f)
			[
				SNew(SButton)
				.Text(LOCTEXT("BrowseBtn", "Browse..."))
				.OnClicked(this, &SEmotionBridgePanel::OnBrowseWav)
			]
		];
}

TSharedRef<SWidget> SEmotionBridgePanel::BuildParametersSection()
{
	return SNew(SVerticalBox)
		+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 0.f, 0.f, 4.f)
		[
			SNew(STextBlock)
			.Text(LOCTEXT("ParamsLabel", "TIMELINE PARAMETERS"))
			.Font(FCoreStyle::GetDefaultFontStyle("Bold", 10))
		]

		// Row 1: window_sec + hop_sec
		+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 2.f)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0.f, 0.f, 4.f, 0.f)
			[
				SNew(STextBlock).Text(LOCTEXT("WindowLabel", "Window (s):"))
			]
			+ SHorizontalBox::Slot().MaxWidth(70.f).Padding(0.f, 0.f, 16.f, 0.f)
			[
				SNew(SEditableTextBox)
				.Text_Lambda([this]() { return FText::FromString(FString::Printf(TEXT("%.2f"), WindowSec)); })
				.OnTextCommitted_Lambda([this](const FText& T, ETextCommit::Type)
				{
					WindowSec = FMath::Max(0.1f, FCString::Atof(*T.ToString()));
				})
			]
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0.f, 0.f, 4.f, 0.f)
			[
				SNew(STextBlock).Text(LOCTEXT("HopLabel", "Hop (s):"))
			]
			+ SHorizontalBox::Slot().MaxWidth(70.f)
			[
				SNew(SEditableTextBox)
				.Text_Lambda([this]() { return FText::FromString(FString::Printf(TEXT("%.2f"), HopSec)); })
				.OnTextCommitted_Lambda([this](const FText& T, ETextCommit::Type)
				{
					HopSec = FMath::Max(0.05f, FCString::Atof(*T.ToString()));
				})
			]
		]

		// Row 2: pad_mode combo
		+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 2.f)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0.f, 0.f, 4.f, 0.f)
			[
				SNew(STextBlock).Text(LOCTEXT("PadModeLabel", "Pad Mode:"))
			]
			+ SHorizontalBox::Slot().MaxWidth(140.f)
			[
				SNew(SComboBox<TSharedPtr<FString>>)
				.OptionsSource(&PadModeOptions)
				.InitiallySelectedItem(SelectedPadMode)
				.OnGenerateWidget_Lambda([](TSharedPtr<FString> Opt) -> TSharedRef<SWidget>
				{
					return SNew(STextBlock).Text(FText::FromString(*Opt));
				})
				.OnSelectionChanged_Lambda([this](TSharedPtr<FString> Sel, ESelectInfo::Type)
				{
					if (Sel.IsValid()) { SelectedPadMode = Sel; PadMode = *Sel; }
				})
				[
					SNew(STextBlock).Text_Lambda([this]()
					{
						return FText::FromString(SelectedPadMode.IsValid() ? *SelectedPadMode : PadMode);
					})
				]
			]
		]

		// Row 3: smoothing_method + hysteresis_min_run
		+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 2.f)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0.f, 0.f, 4.f, 0.f)
			[
				SNew(STextBlock).Text(LOCTEXT("SmoothingLabel", "Smoothing:"))
			]
			+ SHorizontalBox::Slot().MaxWidth(160.f).Padding(0.f, 0.f, 16.f, 0.f)
			[
				SNew(SComboBox<TSharedPtr<FString>>)
				.OptionsSource(&SmoothingOptions)
				.InitiallySelectedItem(SelectedSmoothingMethod)
				.OnGenerateWidget_Lambda([](TSharedPtr<FString> Opt) -> TSharedRef<SWidget>
				{
					return SNew(STextBlock).Text(FText::FromString(*Opt));
				})
				.OnSelectionChanged_Lambda([this](TSharedPtr<FString> Sel, ESelectInfo::Type)
				{
					if (Sel.IsValid()) { SelectedSmoothingMethod = Sel; SmoothingMethod = *Sel; }
				})
				[
					SNew(STextBlock).Text_Lambda([this]()
					{
						return FText::FromString(
							SelectedSmoothingMethod.IsValid() ? *SelectedSmoothingMethod : SmoothingMethod);
					})
				]
			]
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(0.f, 0.f, 4.f, 0.f)
			[
				SNew(STextBlock).Text(LOCTEXT("MinRunLabel", "Min Run:"))
			]
			+ SHorizontalBox::Slot().MaxWidth(50.f)
			[
				SNew(SEditableTextBox)
				.Text_Lambda([this]() { return FText::FromString(FString::FromInt(HysteresisMinRun)); })
				.OnTextCommitted_Lambda([this](const FText& T, ETextCommit::Type)
				{
					HysteresisMinRun = FMath::Max(1, FCString::Atoi(*T.ToString()));
				})
			]
		]

		// Row 4: include_windows + include_scores checkboxes
		+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 2.f)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot().AutoWidth().Padding(0.f, 0.f, 16.f, 0.f)
			[
				SNew(SCheckBox)
				.IsChecked_Lambda([this]()
				{
					return bIncludeWindows ? ECheckBoxState::Checked : ECheckBoxState::Unchecked;
				})
				.OnCheckStateChanged_Lambda([this](ECheckBoxState S)
				{
					bIncludeWindows = (S == ECheckBoxState::Checked);
				})
				[
					SNew(STextBlock).Text(LOCTEXT("IncludeWindows", "Include Windows"))
				]
			]
			+ SHorizontalBox::Slot().AutoWidth()
			[
				SNew(SCheckBox)
				.IsChecked_Lambda([this]()
				{
					return bIncludeScores ? ECheckBoxState::Checked : ECheckBoxState::Unchecked;
				})
				.OnCheckStateChanged_Lambda([this](ECheckBoxState S)
				{
					bIncludeScores = (S == ECheckBoxState::Checked);
				})
				[
					SNew(STextBlock).Text(LOCTEXT("IncludeScores", "Include Scores"))
				]
			]
		];
}

TSharedRef<SWidget> SEmotionBridgePanel::BuildResultsSection()
{
	// Column header row displayed above the list.
	TSharedRef<SHeaderRow> HeaderRow = SNew(SHeaderRow)
		+ SHeaderRow::Column(ColStart)
			.DefaultLabel(LOCTEXT("ColStart", "Start (s)"))
			.FillWidth(1.f)
		+ SHeaderRow::Column(ColEnd)
			.DefaultLabel(LOCTEXT("ColEnd", "End (s)"))
			.FillWidth(1.f)
		+ SHeaderRow::Column(ColEmotion)
			.DefaultLabel(LOCTEXT("ColEmotion", "Emotion"))
			.FillWidth(2.f)
		+ SHeaderRow::Column(ColConfidence)
			.DefaultLabel(LOCTEXT("ColConf", "Confidence"))
			.FillWidth(1.5f);

	return SNew(SVerticalBox)
		+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 0.f, 0.f, 4.f)
		[
			SNew(STextBlock)
			.Text(LOCTEXT("ResultsLabel", "RESULTS"))
			.Font(FCoreStyle::GetDefaultFontStyle("Bold", 10))
		]
		+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 0.f, 0.f, 4.f)
		[
			SAssignNew(MetadataText, STextBlock)
			.Text(LOCTEXT("NoResults", "No results yet."))
			.AutoWrapText(true)
			.ColorAndOpacity(FSlateColor::UseSubduedForeground())
		]
		+ SVerticalBox::Slot().MaxHeight(220.f)
		[
			SAssignNew(SegmentListView,
				SListView<TSharedPtr<FEmotionSegmentRow>>)
			.ListItemsSource(&SegmentRows)
			.OnGenerateRow(this, &SEmotionBridgePanel::GenerateSegmentRow)
			.HeaderRow(HeaderRow)
			.SelectionMode(ESelectionMode::Single)
		];
}

TSharedRef<SWidget> SEmotionBridgePanel::BuildPlaybackSection()
{
	return SNew(SVerticalBox)
		+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 0.f, 0.f, 4.f)
		[
			SNew(STextBlock)
			.Text(LOCTEXT("PlaybackLabel", "PLAYBACK DEMO"))
			.Font(FCoreStyle::GetDefaultFontStyle("Bold", 10))
		]
		+ SVerticalBox::Slot().AutoHeight().Padding(0.f, 0.f, 0.f, 4.f)
		[
			SNew(STextBlock)
			.Text(LOCTEXT("PlaybackDesc",
				"Play Demo will find or spawn an AEmotionLampActor in the editor world and "
				"change its point-light color as each emotion segment becomes active. "
				"Stop Demo halts playback and resets the lamp to neutral."))
			.AutoWrapText(true)
			.ColorAndOpacity(FSlateColor::UseSubduedForeground())
		]
		+ SVerticalBox::Slot().AutoHeight()
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot().AutoWidth().Padding(0.f, 0.f, 8.f, 0.f)
			[
				SNew(SButton)
				.Text(LOCTEXT("PlayBtn", "Play Demo"))
				.ToolTipText(LOCTEXT("PlayTip",
					"Start timeline simulation. Run Analyze first."))
				.IsEnabled_Lambda([this]() { return !bIsPlaying && CurrentTimeline.bIsValid; })
				.OnClicked(this, &SEmotionBridgePanel::OnPlayDemo)
			]
			+ SHorizontalBox::Slot().AutoWidth()
			[
				SNew(SButton)
				.Text(LOCTEXT("StopBtn", "Stop Demo"))
				.ToolTipText(LOCTEXT("StopTip", "Stop playback and reset the lamp to neutral."))
				.IsEnabled_Lambda([this]() { return bIsPlaying; })
				.OnClicked(this, &SEmotionBridgePanel::OnStopDemo)
			]
		];
}

// ---------------------------------------------------------------------------
// Button callbacks
// ---------------------------------------------------------------------------

FReply SEmotionBridgePanel::OnHealthCheck()
{
	if (!ApiClient.IsValid())
		return FReply::Handled();

	// Update the base URL from the text box first.
	if (ApiUrlBox.IsValid())
		ApiClient->SetBaseUrl(ApiUrlBox->GetText().ToString());

	SetStatus(TEXT("Checking backend health..."), FLinearColor(1.f, 0.85f, 0.f));

	ApiClient->CheckHealth(
		FOnHealthCheckComplete::CreateSP(this, &SEmotionBridgePanel::OnHealthCheckResult));

	return FReply::Handled();
}

FReply SEmotionBridgePanel::OnBrowseWav()
{
	IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
	if (!DesktopPlatform)
	{
		SetStatus(TEXT("Desktop platform not available for file dialog."), FLinearColor::Red);
		return FReply::Handled();
	}

	// Use the project directory as the initial path.
	const FString DefaultPath = FPaths::ProjectDir();

	TArray<FString> OutFiles;
	const void* ParentWindow = FSlateApplication::Get().GetActiveTopLevelWindow().IsValid()
		? FSlateApplication::Get().GetActiveTopLevelWindow()->GetNativeWindow()->GetOSWindowHandle()
		: nullptr;

	const bool bOpened = DesktopPlatform->OpenFileDialog(
		ParentWindow,
		TEXT("Select WAV File"),
		DefaultPath,
		TEXT(""),
		TEXT("WAV Audio Files (*.wav)|*.wav|All Files (*.*)|*.*"),
		EFileDialogFlags::None,
		OutFiles);

	if (bOpened && OutFiles.Num() > 0)
	{
		WavFilePath = OutFiles[0];
		if (WavPathBox.IsValid())
			WavPathBox->SetText(FText::FromString(WavFilePath));
	}

	return FReply::Handled();
}

FReply SEmotionBridgePanel::OnAnalyze()
{
	// Sync URL from the text box every time.
	if (ApiUrlBox.IsValid() && ApiClient.IsValid())
		ApiClient->SetBaseUrl(ApiUrlBox->GetText().ToString());

	// Sync WAV path from the text box in case the user typed it.
	if (WavPathBox.IsValid())
		WavFilePath = WavPathBox->GetText().ToString();

	// --- Validate inputs ---
	if (WavFilePath.IsEmpty())
	{
		SetStatus(TEXT("Please select or type a WAV file path first."), FLinearColor(1.f, 0.5f, 0.f));
		return FReply::Handled();
	}
	if (!FPaths::FileExists(WavFilePath))
	{
		SetStatus(
			FString::Printf(TEXT("File not found: %s"), *WavFilePath),
			FLinearColor::Red);
		return FReply::Handled();
	}
	const FString Ext = FPaths::GetExtension(WavFilePath).ToLower();
	if (Ext != TEXT("wav"))
	{
		SetStatus(
			FString::Printf(TEXT("Expected a .wav file, got .%s"), *Ext),
			FLinearColor::Red);
		return FReply::Handled();
	}

	// --- Fire the request ---
	bIsAnalyzing = true;
	CurrentTimeline = FEmotionTimelineResponse{}; // clear previous results
	RefreshSegmentList();

	SetStatus(
		TEXT("Analyzing... The model may take 30–120 s to download on the first run. Please wait."),
		FLinearColor(1.f, 0.85f, 0.f));

	ApiClient->RequestTimeline(
		WavFilePath,
		WindowSec, HopSec,
		PadMode, SmoothingMethod, HysteresisMinRun,
		bIncludeWindows, bIncludeScores,
		FOnTimelineComplete::CreateSP(this, &SEmotionBridgePanel::OnTimelineReceived));

	return FReply::Handled();
}

FReply SEmotionBridgePanel::OnPlayDemo()
{
	if (!CurrentTimeline.bIsValid || CurrentTimeline.Segments.Num() == 0)
	{
		SetStatus(TEXT("No valid timeline. Run Analyze first."), FLinearColor(1.f, 0.5f, 0.f));
		return FReply::Handled();
	}

	AEmotionLampActor* Lamp = FindOrSpawnLampActor();
	if (!Lamp)
	{
		SetStatus(TEXT("Could not find or spawn an AEmotionLampActor. Is a level open?"),
			FLinearColor::Red);
		return FReply::Handled();
	}

	LampActorRef          = Lamp;
	bIsPlaying            = true;
	LastActiveSegmentIndex = -1;
	PlaybackStartWallTime  = FPlatformTime::Seconds();

	// Register the FTSTicker callback — runs on the game thread each frame.
	PlaybackTickerHandle = FTSTicker::GetCoreTicker().AddTicker(
		FTickerDelegate::CreateSP(this, &SEmotionBridgePanel::OnPlaybackTick));

	SetStatus(
		FString::Printf(TEXT("Playing demo — %.2f s total, %d segments."),
			CurrentTimeline.DurationSec, CurrentTimeline.Segments.Num()),
		FLinearColor::Green);

	UE_LOG(LogEmotionBridge, Log,
		TEXT("Demo playback started for actor: %s"), *Lamp->GetName());

	return FReply::Handled();
}

FReply SEmotionBridgePanel::OnStopDemo()
{
	bIsPlaying = false;

	if (PlaybackTickerHandle.IsValid())
	{
		FTSTicker::GetCoreTicker().RemoveTicker(PlaybackTickerHandle);
		PlaybackTickerHandle = {};
	}

	if (LampActorRef.IsValid())
	{
		LampActorRef->ResetToNeutral();
	}

	LastActiveSegmentIndex = -1;

	SetStatus(TEXT("Playback stopped. Lamp reset to neutral."), FLinearColor::White);
	UE_LOG(LogEmotionBridge, Log, TEXT("Demo playback stopped by user."));

	return FReply::Handled();
}

// ---------------------------------------------------------------------------
// HTTP response handlers
// ---------------------------------------------------------------------------

void SEmotionBridgePanel::OnTimelineReceived(const FEmotionTimelineResponse& Response)
{
	bIsAnalyzing = false;

	if (!Response.bIsValid)
	{
		SetStatus(
			FString::Printf(TEXT("Error: %s"), *Response.ErrorMessage),
			FLinearColor::Red);
		return;
	}

	CurrentTimeline = Response;
	RefreshSegmentList();

	// Build a one-line metadata summary.
	const FString Meta = FString::Printf(
		TEXT("Model: %s  |  Sample Rate: %d Hz  |  Duration: %.2f s  |  Segments: %d  |  Window: %.2f s  |  Hop: %.2f s"),
		*Response.ModelName,
		Response.SampleRate,
		Response.DurationSec,
		Response.Segments.Num(),
		Response.WindowSec,
		Response.HopSec);

	if (MetadataText.IsValid())
		MetadataText->SetText(FText::FromString(Meta));

	SetStatus(
		FString::Printf(TEXT("Analysis complete — %d segments detected. Click Play Demo to run the visualization."),
			Response.Segments.Num()),
		FLinearColor::Green);
}

void SEmotionBridgePanel::OnHealthCheckResult(bool bHealthy)
{
	if (bHealthy)
	{
		SetStatus(TEXT("Backend is healthy and reachable."), FLinearColor::Green);
	}
	else
	{
		SetStatus(
			TEXT("Backend not reachable. Start it with: docker compose up api"),
			FLinearColor::Red);
	}
}

// ---------------------------------------------------------------------------
// Playback ticker
// ---------------------------------------------------------------------------

bool SEmotionBridgePanel::OnPlaybackTick(float /*DeltaTime*/)
{
	if (!bIsPlaying)
		return false; // Unregister

	const double ElapsedSec = FPlatformTime::Seconds() - PlaybackStartWallTime;

	// End of timeline.
	if (ElapsedSec >= static_cast<double>(CurrentTimeline.DurationSec))
	{
		bIsPlaying = false;
		PlaybackTickerHandle = {};
		SetStatus(TEXT("Playback finished. Lamp holds the last emotion color."), FLinearColor::Green);
		UE_LOG(LogEmotionBridge, Log, TEXT("Demo playback reached end of timeline."));
		return false;
	}

	// Find the active segment.
	int32 ActiveIndex = -1;
	const float ElapsedF = static_cast<float>(ElapsedSec);
	for (int32 i = 0; i < CurrentTimeline.Segments.Num(); ++i)
	{
		const FEmotionSegment& Seg = CurrentTimeline.Segments[i];
		if (ElapsedF >= Seg.StartSec && ElapsedF < Seg.EndSec)
		{
			ActiveIndex = i;
			break;
		}
	}

	// Only update when the segment changes.
	if (ActiveIndex == LastActiveSegmentIndex)
		return true;

	LastActiveSegmentIndex = ActiveIndex;

	if (LampActorRef.IsValid())
	{
		if (ActiveIndex >= 0)
		{
			const FEmotionSegment& Seg = CurrentTimeline.Segments[ActiveIndex];
			LampActorRef->ApplyEmotion(Seg.Emotion, Seg.Confidence);

			const FString SegMsg = FString::Printf(
				TEXT("Playing: t=%.1f s  emotion=%s  conf=%.0f%%"),
				ElapsedF, *Seg.Emotion, Seg.Confidence * 100.f);
			SetStatus(SegMsg, GetSlateColorForEmotion(Seg.Emotion).GetSpecifiedColor());
		}
		else
		{
			// Between segments — go neutral.
			LampActorRef->ApplyEmotion(TEXT("neutral"), 1.0f);
		}
	}
	else
	{
		// Lamp was destroyed during playback.
		UE_LOG(LogEmotionBridge, Warning,
			TEXT("Playback ticker: LampActor is no longer valid."));
		bIsPlaying = false;
		SetStatus(TEXT("Lamp actor was destroyed during playback."), FLinearColor::Red);
		return false;
	}

	return true; // Keep ticking.
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void SEmotionBridgePanel::SetStatus(const FString& Message, FLinearColor Color)
{
	if (StatusText.IsValid())
		StatusText->SetText(FText::FromString(Message));
	if (StatusText.IsValid())
		StatusText->SetColorAndOpacity(FSlateColor(Color));
}

AEmotionLampActor* SEmotionBridgePanel::FindOrSpawnLampActor()
{
	if (!GEditor)
		return nullptr;

	UWorld* World = GEditor->GetEditorWorldContext().World();
	if (!World)
	{
		UE_LOG(LogEmotionBridge, Warning, TEXT("FindOrSpawnLampActor: no editor world available."));
		return nullptr;
	}

	// Search for an existing lamp actor first.
	for (TActorIterator<AEmotionLampActor> It(World); It; ++It)
	{
		UE_LOG(LogEmotionBridge, Log,
			TEXT("FindOrSpawnLampActor: found existing actor '%s'."), *It->GetName());
		return *It;
	}

	// None found — spawn one at the origin, slightly above ground.
	FActorSpawnParameters SpawnParams;
	SpawnParams.Name = FName("EmotionLamp_AutoSpawned");
	SpawnParams.SpawnCollisionHandlingOverride =
		ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

	AEmotionLampActor* Lamp = World->SpawnActor<AEmotionLampActor>(
		AEmotionLampActor::StaticClass(),
		FVector(0.f, 0.f, 50.f),
		FRotator::ZeroRotator,
		SpawnParams);

	if (Lamp)
	{
		UE_LOG(LogEmotionBridge, Log,
			TEXT("FindOrSpawnLampActor: spawned new AEmotionLampActor at origin."));
	}
	else
	{
		UE_LOG(LogEmotionBridge, Error,
			TEXT("FindOrSpawnLampActor: SpawnActor failed."));
	}

	return Lamp;
}

void SEmotionBridgePanel::RefreshSegmentList()
{
	SegmentRows.Empty();
	for (const FEmotionSegment& Seg : CurrentTimeline.Segments)
	{
		SegmentRows.Add(MakeShared<FEmotionSegmentRow>(Seg));
	}
	if (SegmentListView.IsValid())
		SegmentListView->RequestListRefresh();
}

TSharedRef<ITableRow> SEmotionBridgePanel::GenerateSegmentRow(
	TSharedPtr<FEmotionSegmentRow> Item,
	const TSharedRef<STableViewBase>& OwnerTable)
{
	const FLinearColor EmotionColor =
		GetSlateColorForEmotion(Item->Segment.Emotion).GetSpecifiedColor();

	return SNew(STableRow<TSharedPtr<FEmotionSegmentRow>>, OwnerTable)
		.Padding(FMargin(4.f, 2.f))
		[
			SNew(SHorizontalBox)

			// Start (s)
			+ SHorizontalBox::Slot().FillWidth(1.f)
			[
				SNew(STextBlock)
				.Text(FText::FromString(FString::Printf(TEXT("%.2f"), Item->Segment.StartSec)))
			]

			// End (s)
			+ SHorizontalBox::Slot().FillWidth(1.f)
			[
				SNew(STextBlock)
				.Text(FText::FromString(FString::Printf(TEXT("%.2f"), Item->Segment.EndSec)))
			]

			// Emotion (colored)
			+ SHorizontalBox::Slot().FillWidth(2.f)
			[
				SNew(STextBlock)
				.Text(FText::FromString(Item->Segment.Emotion))
				.ColorAndOpacity(FSlateColor(EmotionColor))
				.Font(FCoreStyle::GetDefaultFontStyle("Bold", 10))
			]

			// Confidence %
			+ SHorizontalBox::Slot().FillWidth(1.5f)
			[
				SNew(STextBlock)
				.Text(FText::FromString(
					FString::Printf(TEXT("%.1f%%"), Item->Segment.Confidence * 100.f)))
			]
		];
}

FSlateColor SEmotionBridgePanel::GetSlateColorForEmotion(const FString& Emotion) const
{
	const UEmotionBridgeSettings* Settings = UEmotionBridgeSettings::Get();
	if (Settings)
		return FSlateColor(Settings->GetColorForEmotion(Emotion));
	return FSlateColor::UseForeground();
}

#undef LOCTEXT_NAMESPACE
