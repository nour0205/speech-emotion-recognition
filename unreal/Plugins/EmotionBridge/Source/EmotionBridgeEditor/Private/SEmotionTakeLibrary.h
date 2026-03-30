// Copyright (c) EmotionDemo Project. All rights reserved.
// Phase 2A — Take Library Slate widget.
// Internal — not exported from the module.
#pragma once

#include "CoreMinimal.h"
#include "Widgets/SCompoundWidget.h"
#include "EmotionTakeTypes.h"
#include "EmotionTakeStore.h"

// ---------------------------------------------------------------------------
// Delegates fired by SEmotionTakeLibrary toward the parent panel
// ---------------------------------------------------------------------------

/** Fired when the user requests to load a take into the main panel. */
DECLARE_DELEGATE_OneParam(FOnTakeLoadRequested,      const FEmotionTakeRecord&);

/** Fired when the user requests to load + immediately start playback. */
DECLARE_DELEGATE_OneParam(FOnTakePlayRequested,      const FEmotionTakeRecord&);

/** Fired when the user requests to re-run analysis for a take. */
DECLARE_DELEGATE_OneParam(FOnTakeReanalyzeRequested, const FEmotionTakeRecord&);

// ---------------------------------------------------------------------------
// Sort mode enum (index matches SortOptions array in .cpp)
// ---------------------------------------------------------------------------
enum class EEmotionTakeSortMode : uint8
{
	NameAsc      = 0, // A → Z
	CreatedDesc  = 1, // newest first
	CreatedAsc   = 2, // oldest first
	DurationDesc = 3, // longest first
};

// ---------------------------------------------------------------------------
// SEmotionTakeLibrary
//
// Dockable Slate compound widget embedded inside SEmotionBridgePanel.
//
// Layout:
//   [TAKE LIBRARY header + take count]
//   [Search box] [Sort combo] [Emotion filter] [Refresh button]
//   [List view — columns: Name | Created | Duration | Emotion | Source ]
//   [Load] [Play] [Delete] [Duplicate] [Reanalyze]      (action row)
//   [Detail panel — expanded info for the selected take]
// ---------------------------------------------------------------------------
class SEmotionTakeLibrary : public SCompoundWidget
{
public:
	SLATE_BEGIN_ARGS(SEmotionTakeLibrary) {}
		SLATE_EVENT(FOnTakeLoadRequested,      OnLoadRequested)
		SLATE_EVENT(FOnTakePlayRequested,      OnPlayRequested)
		SLATE_EVENT(FOnTakeReanalyzeRequested, OnReanalyzeRequested)
	SLATE_END_ARGS()

	void Construct(const FArguments& InArgs);

	/** Reload all takes from disk and refresh the list. */
	void RefreshLibrary();

	/** Programmatically select the take with the given ID (if present in the list). */
	void SelectTakeById(const FString& TakeId);

private:
	// -----------------------------------------------------------------------
	// State
	// -----------------------------------------------------------------------

	/** All takes loaded from disk, unsorted and unfiltered. */
	TArray<TSharedPtr<FEmotionTakeRecord>> AllTakes;

	/** Subset of AllTakes after filter + sort — used as list source. */
	TArray<TSharedPtr<FEmotionTakeRecord>> DisplayedTakes;

	/** Currently selected item in the list view (may be null). */
	TSharedPtr<FEmotionTakeRecord> SelectedTake;

	// Filter + sort state
	FString               SearchText;
	EEmotionTakeSortMode  SortMode         = EEmotionTakeSortMode::CreatedDesc;
	FString               FilterEmotion;   // empty = show all

	// Combo option lists
	TArray<TSharedPtr<FString>> SortOptions;
	TSharedPtr<FString>         SelectedSortOption;

	TArray<TSharedPtr<FString>> FilterOptions;
	TSharedPtr<FString>         SelectedFilterOption;

	// -----------------------------------------------------------------------
	// Widget references
	// -----------------------------------------------------------------------
	TSharedPtr<SListView<TSharedPtr<FEmotionTakeRecord>>> ListView;
	TSharedPtr<SEditableTextBox>                          SearchBox;
	TSharedPtr<STextBlock>                                HeaderCountText;
	TSharedPtr<STextBlock>                                DetailText;

	// -----------------------------------------------------------------------
	// Delegates from parent panel
	// -----------------------------------------------------------------------
	FOnTakeLoadRequested      OnLoadRequested;
	FOnTakePlayRequested      OnPlayRequested;
	FOnTakeReanalyzeRequested OnReanalyzeRequested;

	// -----------------------------------------------------------------------
	// UI builders
	// -----------------------------------------------------------------------
	TSharedRef<SWidget> BuildToolbarRow();
	TSharedRef<SWidget> BuildListView();
	TSharedRef<SWidget> BuildActionRow();
	TSharedRef<SWidget> BuildDetailPanel();

	/** Generates one row for the list view. */
	TSharedRef<ITableRow> GenerateTakeRow(
		TSharedPtr<FEmotionTakeRecord>  Item,
		const TSharedRef<STableViewBase>& OwnerTable);

	// -----------------------------------------------------------------------
	// Filter / sort
	// -----------------------------------------------------------------------
	void  ApplyFilterAndSort();
	bool  PassesFilter(const FEmotionTakeRecord& Take) const;

	// -----------------------------------------------------------------------
	// Selection
	// -----------------------------------------------------------------------
	void OnSelectionChanged(TSharedPtr<FEmotionTakeRecord> Item, ESelectInfo::Type Info);

	// -----------------------------------------------------------------------
	// Action callbacks
	// -----------------------------------------------------------------------
	FReply OnLoadClicked();
	FReply OnPlayClicked();
	FReply OnDeleteClicked();
	FReply OnDuplicateClicked();
	FReply OnReanalyzeClicked();
	FReply OnRefreshClicked();

	// -----------------------------------------------------------------------
	// Internal helpers
	// -----------------------------------------------------------------------

	/** Build multi-line detail text for the currently selected take. */
	FText BuildDetailText() const;

	/** Get an emotion-tinted FSlateColor using EmotionBridgeSettings. */
	FSlateColor SlateColorForEmotion(const FString& Emotion) const;

	/** Update the header count label. */
	void UpdateCountLabel();
};
