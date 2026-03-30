// Copyright (c) EmotionDemo Project. All rights reserved.
// Phase 2A — Take Library Slate widget implementation.

#include "SEmotionTakeLibrary.h"
#include "EmotionBridgeLog.h"
#include "EmotionBridgeSettings.h"

// Slate
#include "Widgets/SBoxPanel.h"
#include "Widgets/Layout/SBorder.h"
#include "Widgets/Layout/SBox.h"
#include "Widgets/Layout/SSeparator.h"
#include "Widgets/Layout/SScrollBox.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SEditableTextBox.h"
#include "Widgets/Input/SComboBox.h"
#include "Widgets/Views/SListView.h"
#include "Widgets/Views/STableRow.h"
#include "Widgets/Views/SHeaderRow.h"
#include "Styling/CoreStyle.h"
#include "Styling/AppStyle.h"

// Framework
#include "Framework/Application/SlateApplication.h"
#include "Framework/MultiBox/MultiBoxBuilder.h"
#include "Misc/MessageDialog.h"

#define LOCTEXT_NAMESPACE "SEmotionTakeLibrary"

// ===========================================================================
// Construct
// ===========================================================================

void SEmotionTakeLibrary::Construct(const FArguments& InArgs)
{
	OnLoadRequested      = InArgs._OnLoadRequested;
	OnPlayRequested      = InArgs._OnPlayRequested;
	OnReanalyzeRequested = InArgs._OnReanalyzeRequested;

	// Build sort and filter combo options.
	SortOptions.Add(MakeShared<FString>(TEXT("Name (A\u2192Z)")));
	SortOptions.Add(MakeShared<FString>(TEXT("Created (newest)")));
	SortOptions.Add(MakeShared<FString>(TEXT("Created (oldest)")));
	SortOptions.Add(MakeShared<FString>(TEXT("Duration (longest)")));
	SelectedSortOption = SortOptions[1]; // default: newest first

	FilterOptions.Add(MakeShared<FString>(TEXT("All emotions")));
	FilterOptions.Add(MakeShared<FString>(TEXT("angry")));
	FilterOptions.Add(MakeShared<FString>(TEXT("happy")));
	FilterOptions.Add(MakeShared<FString>(TEXT("sad")));
	FilterOptions.Add(MakeShared<FString>(TEXT("neutral")));
	SelectedFilterOption = FilterOptions[0];

	ChildSlot
	[
		SNew(SVerticalBox)

		// ---- Header -------------------------------------------------------
		+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,4)
		[
			SNew(SHorizontalBox)
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center)
			[
				SNew(STextBlock)
				.Text(LOCTEXT("LibraryHeader", "TAKE LIBRARY"))
				.Font(FCoreStyle::GetDefaultFontStyle("Bold", 10))
			]
			+ SHorizontalBox::Slot().AutoWidth().VAlign(VAlign_Center).Padding(8,0,0,0)
			[
				SAssignNew(HeaderCountText, STextBlock)
				.Text(LOCTEXT("CountZero", "(0 takes)"))
				.ColorAndOpacity(FSlateColor::UseSubduedForeground())
			]
		]

		+ SVerticalBox::Slot().AutoHeight().Padding(0,0,0,4) [ BuildToolbarRow() ]
		+ SVerticalBox::Slot().MaxHeight(220.f)              [ BuildListView()   ]
		+ SVerticalBox::Slot().AutoHeight().Padding(0,4,0,4) [ BuildActionRow()  ]
		+ SVerticalBox::Slot().AutoHeight()                  [ BuildDetailPanel()]
	];

	// Load from disk immediately.
	RefreshLibrary();
}

// ===========================================================================
// UI builders
// ===========================================================================

TSharedRef<SWidget> SEmotionTakeLibrary::BuildToolbarRow()
{
	return SNew(SHorizontalBox)
		// Search
		+ SHorizontalBox::Slot().FillWidth(1.f).Padding(0,0,6,0)
		[
			SAssignNew(SearchBox, SEditableTextBox)
			.HintText(LOCTEXT("SearchHint", "Search by name..."))
			.OnTextChanged_Lambda([this](const FText& T)
			{
				SearchText = T.ToString();
				ApplyFilterAndSort();
			})
		]
		// Sort
		+ SHorizontalBox::Slot().AutoWidth().Padding(0,0,4,0).VAlign(VAlign_Center)
		[ SNew(STextBlock).Text(LOCTEXT("SortLbl", "Sort:")) ]
		+ SHorizontalBox::Slot().MaxWidth(140.f).Padding(0,0,6,0)
		[
			SNew(SComboBox<TSharedPtr<FString>>)
			.OptionsSource(&SortOptions)
			.InitiallySelectedItem(SelectedSortOption)
			.OnGenerateWidget_Lambda([](TSharedPtr<FString> O) -> TSharedRef<SWidget>
				{ return SNew(STextBlock).Text(FText::FromString(*O)); })
			.OnSelectionChanged_Lambda([this](TSharedPtr<FString> Sel, ESelectInfo::Type)
			{
				if (!Sel.IsValid()) return;
				SelectedSortOption = Sel;
				const int32 Idx = SortOptions.IndexOfByKey(Sel);
				SortMode = static_cast<EEmotionTakeSortMode>(FMath::Clamp(Idx, 0, 3));
				ApplyFilterAndSort();
			})
			[ SNew(STextBlock).Text_Lambda([this]
				{ return FText::FromString(SelectedSortOption.IsValid() ? *SelectedSortOption : TEXT("Sort")); }) ]
		]
		// Emotion filter
		+ SHorizontalBox::Slot().AutoWidth().Padding(0,0,4,0).VAlign(VAlign_Center)
		[ SNew(STextBlock).Text(LOCTEXT("FilterLbl", "Filter:")) ]
		+ SHorizontalBox::Slot().MaxWidth(120.f).Padding(0,0,6,0)
		[
			SNew(SComboBox<TSharedPtr<FString>>)
			.OptionsSource(&FilterOptions)
			.InitiallySelectedItem(SelectedFilterOption)
			.OnGenerateWidget_Lambda([](TSharedPtr<FString> O) -> TSharedRef<SWidget>
				{ return SNew(STextBlock).Text(FText::FromString(*O)); })
			.OnSelectionChanged_Lambda([this](TSharedPtr<FString> Sel, ESelectInfo::Type)
			{
				if (!Sel.IsValid()) return;
				SelectedFilterOption = Sel;
				// "All emotions" → empty filter string; others → the emotion label
				FilterEmotion = (*Sel == TEXT("All emotions")) ? FString{} : *Sel;
				ApplyFilterAndSort();
			})
			[ SNew(STextBlock).Text_Lambda([this]
				{ return FText::FromString(SelectedFilterOption.IsValid() ? *SelectedFilterOption : TEXT("All")); }) ]
		]
		// Refresh
		+ SHorizontalBox::Slot().AutoWidth()
		[
			SNew(SButton)
			.Text(LOCTEXT("RefreshBtn", "Refresh"))
			.ToolTipText(LOCTEXT("RefreshTip", "Reload all takes from disk."))
			.OnClicked(this, &SEmotionTakeLibrary::OnRefreshClicked)
		];
}

TSharedRef<SWidget> SEmotionTakeLibrary::BuildListView()
{
	TSharedRef<SHeaderRow> Header = SNew(SHeaderRow)
		+ SHeaderRow::Column(TEXT("Name"))
			.DefaultLabel(LOCTEXT("ColName",    "Name"))
			.FillWidth(2.5f)
		+ SHeaderRow::Column(TEXT("Created"))
			.DefaultLabel(LOCTEXT("ColCreated", "Created"))
			.FillWidth(1.5f)
		+ SHeaderRow::Column(TEXT("Dur"))
			.DefaultLabel(LOCTEXT("ColDur",     "Duration"))
			.FillWidth(1.f)
		+ SHeaderRow::Column(TEXT("Emotion"))
			.DefaultLabel(LOCTEXT("ColEmo",     "Emotion"))
			.FillWidth(1.5f)
		+ SHeaderRow::Column(TEXT("Source"))
			.DefaultLabel(LOCTEXT("ColSrc",     "Source File"))
			.FillWidth(2.f);

	return SAssignNew(ListView, SListView<TSharedPtr<FEmotionTakeRecord>>)
		.ListItemsSource(&DisplayedTakes)
		.OnGenerateRow(this, &SEmotionTakeLibrary::GenerateTakeRow)
		.OnSelectionChanged(this, &SEmotionTakeLibrary::OnSelectionChanged)
		.HeaderRow(Header)
		.SelectionMode(ESelectionMode::Single);
}

TSharedRef<SWidget> SEmotionTakeLibrary::BuildActionRow()
{
	return SNew(SHorizontalBox)
		+ SHorizontalBox::Slot().AutoWidth().Padding(0,0,4,0)
		[
			SNew(SButton)
			.Text(LOCTEXT("LoadBtn", "Load"))
			.ToolTipText(LOCTEXT("LoadTip",
				"Load the selected take into the main panel. "
				"Restores timeline, segments, and parameters without hitting the backend."))
			.IsEnabled_Lambda([this]{ return SelectedTake.IsValid(); })
			.OnClicked(this, &SEmotionTakeLibrary::OnLoadClicked)
		]
		+ SHorizontalBox::Slot().AutoWidth().Padding(0,0,4,0)
		[
			SNew(SButton)
			.Text(LOCTEXT("PlayBtn", "Play"))
			.ToolTipText(LOCTEXT("PlayTip",
				"Load the selected take and immediately start playback."))
			.IsEnabled_Lambda([this]{ return SelectedTake.IsValid(); })
			.OnClicked(this, &SEmotionTakeLibrary::OnPlayClicked)
		]
		+ SHorizontalBox::Slot().AutoWidth().Padding(0,0,4,0)
		[
			SNew(SButton)
			.Text(LOCTEXT("DupBtn", "Duplicate"))
			.ToolTipText(LOCTEXT("DupTip",
				"Create a copy of the selected take with a new ID. "
				"Useful for preserving the current result before reanalysing."))
			.IsEnabled_Lambda([this]{ return SelectedTake.IsValid(); })
			.OnClicked(this, &SEmotionTakeLibrary::OnDuplicateClicked)
		]
		+ SHorizontalBox::Slot().AutoWidth().Padding(0,0,4,0)
		[
			SNew(SButton)
			.Text(LOCTEXT("ReanalyzeBtn", "Reanalyze"))
			.ToolTipText(LOCTEXT("ReanalyzeTip",
				"Re-run the backend analysis for this take using the stored parameters.\n"
				"The take is updated in-place (timeline + params replaced, name/notes/tags kept).\n"
				"Duplicate first if you want to preserve the current result."))
			.IsEnabled_Lambda([this]{ return SelectedTake.IsValid(); })
			.OnClicked(this, &SEmotionTakeLibrary::OnReanalyzeClicked)
		]
		+ SHorizontalBox::Slot().AutoWidth().Padding(0,0,0,0)
		[
			SNew(SButton)
			.Text(LOCTEXT("DeleteBtn", "Delete"))
			.ToolTipText(LOCTEXT("DeleteTip",
				"Permanently delete the selected take folder from disk. This cannot be undone."))
			.IsEnabled_Lambda([this]{ return SelectedTake.IsValid(); })
			.OnClicked(this, &SEmotionTakeLibrary::OnDeleteClicked)
		];
}

TSharedRef<SWidget> SEmotionTakeLibrary::BuildDetailPanel()
{
	return SNew(SBorder)
		.BorderImage(FAppStyle::GetBrush("ToolPanel.GroupBorder"))
		.Padding(FMargin(8.f, 5.f))
		[
			SAssignNew(DetailText, STextBlock)
			.Text(LOCTEXT("NoSelection",
				"Select a take above to see its details here."))
			.AutoWrapText(true)
			.ColorAndOpacity(FSlateColor::UseSubduedForeground())
		];
}

// ===========================================================================
// Row generator
// ===========================================================================

TSharedRef<ITableRow> SEmotionTakeLibrary::GenerateTakeRow(
	TSharedPtr<FEmotionTakeRecord>  Item,
	const TSharedRef<STableViewBase>& OwnerTable)
{
	const FLinearColor EC = SlateColorForEmotion(
		Item->Summary.IsValid() ? Item->Summary.DominantEmotion : TEXT("neutral"))
		.GetSpecifiedColor();

	// Show only the date portion of the ISO timestamp (first 10 chars = YYYY-MM-DD).
	const FString DateStr = Item->CreatedAt.Left(10);

	const FString DurStr = Item->Summary.TotalDurationSec > 0.f
		? FString::Printf(TEXT("%.1fs"), Item->Summary.TotalDurationSec)
		: (Item->DurationSec > 0.f ? FString::Printf(TEXT("%.1fs"), Item->DurationSec) : TEXT("—"));

	const FString EmotionStr = Item->Summary.IsValid()
		? Item->Summary.DominantEmotion
		: TEXT("—");

	return SNew(STableRow<TSharedPtr<FEmotionTakeRecord>>, OwnerTable)
		.Padding(FMargin(4.f, 2.f))
		[
			SNew(SHorizontalBox)
			// Name
			+ SHorizontalBox::Slot().FillWidth(2.5f).VAlign(VAlign_Center)
			[
				SNew(STextBlock)
				.Text(FText::FromString(Item->DisplayName))
				.Font(FCoreStyle::GetDefaultFontStyle("Regular", 9))
			]
			// Created date
			+ SHorizontalBox::Slot().FillWidth(1.5f).VAlign(VAlign_Center)
			[
				SNew(STextBlock)
				.Text(FText::FromString(DateStr))
				.ColorAndOpacity(FSlateColor::UseSubduedForeground())
			]
			// Duration
			+ SHorizontalBox::Slot().FillWidth(1.f).VAlign(VAlign_Center)
			[
				SNew(STextBlock)
				.Text(FText::FromString(DurStr))
				.ColorAndOpacity(FSlateColor::UseSubduedForeground())
			]
			// Dominant emotion (color-coded)
			+ SHorizontalBox::Slot().FillWidth(1.5f).VAlign(VAlign_Center)
			[
				SNew(STextBlock)
				.Text(FText::FromString(EmotionStr))
				.ColorAndOpacity(FSlateColor(EC))
				.Font(FCoreStyle::GetDefaultFontStyle("Bold", 9))
			]
			// Source filename
			+ SHorizontalBox::Slot().FillWidth(2.f).VAlign(VAlign_Center)
			[
				SNew(STextBlock)
				.Text(FText::FromString(Item->Summary.SourceFilename))
				.ColorAndOpacity(FSlateColor::UseSubduedForeground())
				.Font(FCoreStyle::GetDefaultFontStyle("Regular", 8))
			]
		];
}

// ===========================================================================
// Filter / sort
// ===========================================================================

bool SEmotionTakeLibrary::PassesFilter(const FEmotionTakeRecord& Take) const
{
	// Text search (case-insensitive, against display name).
	if (!SearchText.IsEmpty())
	{
		if (!Take.DisplayName.Contains(SearchText, ESearchCase::IgnoreCase))
			return false;
	}

	// Dominant emotion filter.
	if (!FilterEmotion.IsEmpty())
	{
		if (!Take.Summary.DominantEmotion.Equals(FilterEmotion, ESearchCase::IgnoreCase))
			return false;
	}

	return true;
}

void SEmotionTakeLibrary::ApplyFilterAndSort()
{
	DisplayedTakes.Empty();

	for (const auto& TakePtr : AllTakes)
		if (TakePtr.IsValid() && PassesFilter(*TakePtr))
			DisplayedTakes.Add(TakePtr);

	// Sort.
	switch (SortMode)
	{
	case EEmotionTakeSortMode::NameAsc:
		DisplayedTakes.Sort([](const TSharedPtr<FEmotionTakeRecord>& A,
		                       const TSharedPtr<FEmotionTakeRecord>& B)
		{ return A->DisplayName.Compare(B->DisplayName, ESearchCase::IgnoreCase) < 0; });
		break;

	case EEmotionTakeSortMode::CreatedDesc:
		DisplayedTakes.Sort([](const TSharedPtr<FEmotionTakeRecord>& A,
		                       const TSharedPtr<FEmotionTakeRecord>& B)
		{ return A->CreatedAt > B->CreatedAt; }); // ISO 8601 lexicographic order
		break;

	case EEmotionTakeSortMode::CreatedAsc:
		DisplayedTakes.Sort([](const TSharedPtr<FEmotionTakeRecord>& A,
		                       const TSharedPtr<FEmotionTakeRecord>& B)
		{ return A->CreatedAt < B->CreatedAt; });
		break;

	case EEmotionTakeSortMode::DurationDesc:
		DisplayedTakes.Sort([](const TSharedPtr<FEmotionTakeRecord>& A,
		                       const TSharedPtr<FEmotionTakeRecord>& B)
		{ return A->Summary.TotalDurationSec > B->Summary.TotalDurationSec; });
		break;
	}

	if (ListView.IsValid())
		ListView->RequestListRefresh();

	UpdateCountLabel();
}

// ===========================================================================
// Public interface
// ===========================================================================

void SEmotionTakeLibrary::RefreshLibrary()
{
	AllTakes.Empty();
	TArray<FEmotionTakeRecord> Loaded;
	const int32 N = FEmotionTakeStore::LoadAllTakes(Loaded);
	for (FEmotionTakeRecord& R : Loaded)
		AllTakes.Add(MakeShared<FEmotionTakeRecord>(MoveTemp(R)));

	// Clear selection if the selected take was deleted.
	if (SelectedTake.IsValid())
	{
		const bool bStillPresent = AllTakes.ContainsByPredicate(
			[this](const TSharedPtr<FEmotionTakeRecord>& P)
			{ return P.IsValid() && P->TakeId == SelectedTake->TakeId; });
		if (!bStillPresent)
		{
			SelectedTake.Reset();
			if (DetailText.IsValid())
				DetailText->SetText(LOCTEXT("NoSelection", "Select a take above to see its details here."));
		}
	}

	ApplyFilterAndSort();

	UE_LOG(LogEmotionBridge, Log,
		TEXT("SEmotionTakeLibrary: refreshed — %d takes on disk"), N);
}

void SEmotionTakeLibrary::SelectTakeById(const FString& TakeId)
{
	for (const auto& P : DisplayedTakes)
	{
		if (P.IsValid() && P->TakeId == TakeId)
		{
			ListView->SetSelection(P, ESelectInfo::Direct);
			return;
		}
	}
}

// ===========================================================================
// Selection
// ===========================================================================

void SEmotionTakeLibrary::OnSelectionChanged(
	TSharedPtr<FEmotionTakeRecord> Item, ESelectInfo::Type /*Info*/)
{
	SelectedTake = Item;

	if (DetailText.IsValid())
		DetailText->SetText(BuildDetailText());
}

// ===========================================================================
// Detail text
// ===========================================================================

FText SEmotionTakeLibrary::BuildDetailText() const
{
	if (!SelectedTake.IsValid())
		return LOCTEXT("NoSelection", "Select a take above to see its details here.");

	const FEmotionTakeRecord& R = *SelectedTake;
	const FEmotionTakeSummary& S = R.Summary;

	// Audio availability.
	const bool bHasCopied = !R.CopiedAudioPath.IsEmpty()
		&& FPaths::FileExists(R.GetCopiedAudioFullPath());
	const bool bHasSource = !R.SourceAudioPath.IsEmpty()
		&& FPaths::FileExists(R.SourceAudioPath);

	// Build distribution string  e.g. "neutral 62%  happy 38%"
	FString DistStr;
	TArray<FString> DistParts;
	for (const auto& KV : S.EmotionDistribution)
		DistParts.Add(FString::Printf(TEXT("%s %.0f%%"), *KV.Key, KV.Value * 100.f));
	DistStr = FString::Join(DistParts, TEXT("  "));

	// Tags string.
	const FString TagStr = R.Tags.IsEmpty() ? TEXT("(none)") : FString::Join(R.Tags, TEXT(", "));

	FString Text = FString::Printf(
		TEXT("ID:           %s\n"
		     "Name:         %s\n"
		     "Created:      %s\n"
		     "Updated:      %s\n"
		     "Duration:     %.2f s  |  Sample rate: %d Hz  |  Segments: %d\n"
		     "Dominant:     %s  |  Avg confidence: %.0f%%\n"
		     "Distribution: %s\n"
		     "Source audio: %s  [%s]\n"
		     "Copied audio: %s  [%s]\n"
		     "Analysis:     window=%.2fs  hop=%.2fs  pad=%s  smooth=%s  minRun=%d\n"
		     "Tags:         %s\n"
		     "Notes:        %s\n"
		     "Schema v%d  |  Plugin v%s"),
		*R.TakeId,
		*R.DisplayName,
		*R.CreatedAt,
		*R.UpdatedAt,
		R.DurationSec, R.SampleRate, S.SegmentCount,
		*S.DominantEmotion,
		S.AverageConfidence * 100.f,
		*DistStr,
		*R.SourceAudioPath,  bHasSource ? TEXT("exists") : TEXT("MISSING"),
		*R.GetCopiedAudioFullPath(), bHasCopied ? TEXT("exists") : (R.CopiedAudioPath.IsEmpty() ? TEXT("not copied") : TEXT("MISSING")),
		R.Params.WindowSec, R.Params.HopSec, *R.Params.PadMode, *R.Params.SmoothingMethod, R.Params.HysteresisMinRun,
		*TagStr,
		R.Notes.IsEmpty() ? TEXT("(none)") : *R.Notes,
		R.SchemaVersion, *R.PluginVersion);

	return FText::FromString(Text);
}

// ===========================================================================
// Action callbacks
// ===========================================================================

FReply SEmotionTakeLibrary::OnLoadClicked()
{
	if (!SelectedTake.IsValid()) return FReply::Handled();
	OnLoadRequested.ExecuteIfBound(*SelectedTake);
	return FReply::Handled();
}

FReply SEmotionTakeLibrary::OnPlayClicked()
{
	if (!SelectedTake.IsValid()) return FReply::Handled();
	OnPlayRequested.ExecuteIfBound(*SelectedTake);
	return FReply::Handled();
}

FReply SEmotionTakeLibrary::OnDeleteClicked()
{
	if (!SelectedTake.IsValid()) return FReply::Handled();

	const FText Msg = FText::Format(
		LOCTEXT("ConfirmDelete",
			"Permanently delete take \"{0}\"?\n\nThis will remove all files in:\n  {1}\n\nThis cannot be undone."),
		FText::FromString(SelectedTake->DisplayName),
		FText::FromString(SelectedTake->GetFolderPath()));

	const EAppReturnType::Type Result = FMessageDialog::Open(EAppMsgType::YesNo, Msg);
	if (Result != EAppReturnType::Yes) return FReply::Handled();

	const FString Id = SelectedTake->TakeId;
	SelectedTake.Reset();

	if (!FEmotionTakeStore::DeleteTake(Id))
		UE_LOG(LogEmotionBridge, Error,
			TEXT("SEmotionTakeLibrary: delete failed for take '%s'"), *Id);

	RefreshLibrary();
	return FReply::Handled();
}

FReply SEmotionTakeLibrary::OnDuplicateClicked()
{
	if (!SelectedTake.IsValid()) return FReply::Handled();

	const FString NewName = SelectedTake->DisplayName + TEXT("_Copy");

	FEmotionTakeRecord NewRecord;
	if (FEmotionTakeStore::DuplicateTake(SelectedTake->TakeId, NewName, NewRecord))
	{
		RefreshLibrary();
		SelectTakeById(NewRecord.TakeId);
		UE_LOG(LogEmotionBridge, Log,
			TEXT("SEmotionTakeLibrary: duplicated '%s' → '%s'"),
			*SelectedTake->TakeId, *NewRecord.TakeId);
	}
	else
	{
		UE_LOG(LogEmotionBridge, Error,
			TEXT("SEmotionTakeLibrary: DuplicateTake failed for '%s'"),
			*SelectedTake->TakeId);
	}

	return FReply::Handled();
}

FReply SEmotionTakeLibrary::OnReanalyzeClicked()
{
	if (!SelectedTake.IsValid()) return FReply::Handled();

	const FString BestAudio = SelectedTake->GetBestAudioPath();
	if (BestAudio.IsEmpty())
	{
		FMessageDialog::Open(EAppMsgType::Ok,
			FText::Format(
				LOCTEXT("ReanalyzeMissingAudio",
					"Cannot reanalyze take \"{0}\":\n\nNo audio file is available.\n"
					"  Copied audio: {1}\n"
					"  Source audio: {2}\n\n"
					"Provide the source WAV via Browse in the main panel."),
				FText::FromString(SelectedTake->DisplayName),
				FText::FromString(SelectedTake->GetCopiedAudioFullPath()),
				FText::FromString(SelectedTake->SourceAudioPath)));
		return FReply::Handled();
	}

	OnReanalyzeRequested.ExecuteIfBound(*SelectedTake);
	return FReply::Handled();
}

FReply SEmotionTakeLibrary::OnRefreshClicked()
{
	RefreshLibrary();
	return FReply::Handled();
}

// ===========================================================================
// Helpers
// ===========================================================================

void SEmotionTakeLibrary::UpdateCountLabel()
{
	if (!HeaderCountText.IsValid()) return;
	const FString Txt = (AllTakes.Num() == DisplayedTakes.Num())
		? FString::Printf(TEXT("(%d takes)"), AllTakes.Num())
		: FString::Printf(TEXT("(%d / %d takes)"), DisplayedTakes.Num(), AllTakes.Num());
	HeaderCountText->SetText(FText::FromString(Txt));
}

FSlateColor SEmotionTakeLibrary::SlateColorForEmotion(const FString& Emotion) const
{
	const UEmotionBridgeSettings* S = UEmotionBridgeSettings::Get();
	if (S) return FSlateColor(S->GetColorForEmotion(Emotion));
	return FSlateColor::UseForeground();
}

#undef LOCTEXT_NAMESPACE
