// Copyright (c) EmotionDemo Project. All rights reserved.
// Phase 2A — Take Library persistence service.

#include "EmotionTakeStore.h"
#include "EmotionBridgeLog.h"

#include "HAL/FileManager.h"
#include "HAL/PlatformFileManager.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Misc/Guid.h"
#include "Misc/DateTime.h"
#include "Json.h"

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

FString FEmotionTakeStore::GetTakesRootDirectory()
{
	return FPaths::Combine(
		FPaths::ProjectSavedDir(),
		TEXT("EmotionBridge"),
		TEXT("Takes"));
}

FString FEmotionTakeStore::GetTakeFolderPath(const FString& TakeId)
{
	return FPaths::Combine(GetTakesRootDirectory(), TakeId);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

FString FEmotionTakeStore::GenerateTakeId()
{
	// Strip dashes from GUID to get a compact 32-char hex string.
	return FGuid::NewGuid().ToString(EGuidFormats::Digits);
}

FString FEmotionTakeStore::GenerateDefaultDisplayName()
{
	const FDateTime Now = FDateTime::UtcNow();
	return FString::Printf(TEXT("Take_%04d%02d%02d_%02d%02d%02d"),
		Now.GetYear(), Now.GetMonth(),  Now.GetDay(),
		Now.GetHour(), Now.GetMinute(), Now.GetSecond());
}

bool FEmotionTakeStore::TakeExists(const FString& TakeId)
{
	return IFileManager::Get().DirectoryExists(*GetTakeFolderPath(TakeId));
}

bool FEmotionTakeStore::IsDisplayNameUnique(
	const FString& DisplayName,
	const TArray<FEmotionTakeRecord>& ExistingTakes)
{
	const FString Lower = DisplayName.ToLower();
	for (const FEmotionTakeRecord& R : ExistingTakes)
		if (R.DisplayName.ToLower() == Lower) return false;
	return true;
}

// ---------------------------------------------------------------------------
// ComputeSummary
// ---------------------------------------------------------------------------

FEmotionTakeSummary FEmotionTakeStore::ComputeSummary(
	const FEmotionTimelineResponse& Timeline,
	const FString& SourcePath)
{
	FEmotionTakeSummary S;
	S.TotalDurationSec = Timeline.DurationSec;
	S.SegmentCount     = Timeline.Segments.Num();
	S.SourceFilename   = FPaths::GetCleanFilename(SourcePath);

	if (Timeline.Segments.IsEmpty())
		return S;

	// Accumulate covered duration per emotion label.
	TMap<FString, float> DurMap;
	float SumConf = 0.f;

	for (const FEmotionSegment& Seg : Timeline.Segments)
	{
		const FString Key = Seg.Emotion.ToLower();
		float& Acc = DurMap.FindOrAdd(Key, 0.f);
		Acc += FMath::Max(0.f, Seg.EndSec - Seg.StartSec);
		SumConf += Seg.Confidence;
	}

	S.AverageConfidence = SumConf / static_cast<float>(Timeline.Segments.Num());

	// Normalise to distribution and find dominant emotion.
	float TotalDur = 0.f;
	for (const auto& KV : DurMap) TotalDur += KV.Value;

	float MaxDur = -1.f;
	for (const auto& KV : DurMap)
	{
		const float Frac = TotalDur > 0.f ? KV.Value / TotalDur : 0.f;
		S.EmotionDistribution.Add(KV.Key, Frac);
		if (KV.Value > MaxDur)
		{
			MaxDur = KV.Value;
			S.DominantEmotion = KV.Key;
		}
	}

	return S;
}

// ---------------------------------------------------------------------------
// SaveTake
// ---------------------------------------------------------------------------

bool FEmotionTakeStore::SaveTake(FEmotionTakeRecord& InOutRecord, bool bCopyAudio)
{
	if (InOutRecord.TakeId.IsEmpty())
	{
		UE_LOG(LogEmotionBridge, Error, TEXT("FEmotionTakeStore::SaveTake — TakeId is empty. Assign one before saving."));
		return false;
	}

	const FString FolderPath = GetTakeFolderPath(InOutRecord.TakeId);

	if (!IFileManager::Get().MakeDirectory(*FolderPath, /*Tree=*/true))
	{
		UE_LOG(LogEmotionBridge, Error,
			TEXT("FEmotionTakeStore::SaveTake — cannot create directory '%s'"), *FolderPath);
		return false;
	}

	// Three required files: metadata, timeline, params.
	if (!WriteMetadataJson(InOutRecord, FolderPath)) return false;
	if (!WriteTimelineJson(InOutRecord.Timeline,  FolderPath)) return false;
	if (!WriteParamsJson  (InOutRecord.Params,    FolderPath)) return false;

	// Optional audio copy.
	if (bCopyAudio
		&& !InOutRecord.SourceAudioPath.IsEmpty()
		&& FPaths::FileExists(InOutRecord.SourceAudioPath))
	{
		const FString AudioDir  = FolderPath / TEXT("audio");
		const FString DestPath  = AudioDir   / TEXT("source.wav");

		IFileManager::Get().MakeDirectory(*AudioDir, true);

		const uint32 CopyResult = IFileManager::Get().Copy(*DestPath, *InOutRecord.SourceAudioPath);
		if (CopyResult == COPY_OK)
		{
			InOutRecord.CopiedAudioPath = TEXT("audio/source.wav");
			// Re-save metadata so CopiedAudioPath is persisted.
			WriteMetadataJson(InOutRecord, FolderPath);
			UE_LOG(LogEmotionBridge, Log,
				TEXT("FEmotionTakeStore: audio copied → '%s'"), *DestPath);
		}
		else
		{
			UE_LOG(LogEmotionBridge, Warning,
				TEXT("FEmotionTakeStore: audio copy failed (error %u) — source was '%s'. "
				     "Take saved without embedded audio."),
				CopyResult, *InOutRecord.SourceAudioPath);
		}
	}

	// Recompute summary in-memory (not saved to disk; recomputed on every load).
	InOutRecord.Summary = ComputeSummary(InOutRecord.Timeline, InOutRecord.SourceAudioPath);

	UE_LOG(LogEmotionBridge, Log,
		TEXT("FEmotionTakeStore: take '%s' (%s) saved → '%s'"),
		*InOutRecord.DisplayName, *InOutRecord.TakeId, *FolderPath);
	return true;
}

// ---------------------------------------------------------------------------
// LoadTake
// ---------------------------------------------------------------------------

bool FEmotionTakeStore::LoadTake(const FString& TakeId, FEmotionTakeRecord& OutRecord)
{
	const FString FolderPath = GetTakeFolderPath(TakeId);

	if (!IFileManager::Get().DirectoryExists(*FolderPath))
	{
		UE_LOG(LogEmotionBridge, Warning,
			TEXT("FEmotionTakeStore::LoadTake — folder does not exist: '%s'"), *FolderPath);
		return false;
	}

	OutRecord = FEmotionTakeRecord{};

	if (!ReadMetadataJson(FolderPath / TEXT("metadata.json"), OutRecord))
	{
		UE_LOG(LogEmotionBridge, Warning,
			TEXT("FEmotionTakeStore::LoadTake — metadata.json failed for TakeId='%s'"), *TakeId);
		return false;
	}

	if (!ReadTimelineJson(FolderPath / TEXT("timeline.json"), OutRecord.Timeline))
	{
		UE_LOG(LogEmotionBridge, Warning,
			TEXT("FEmotionTakeStore::LoadTake — timeline.json failed for TakeId='%s'"), *TakeId);
		return false;
	}

	// params.json is required — without it we cannot reanalyse faithfully.
	if (!ReadParamsJson(FolderPath / TEXT("params.json"), OutRecord.Params))
	{
		UE_LOG(LogEmotionBridge, Warning,
			TEXT("FEmotionTakeStore::LoadTake — params.json failed for TakeId='%s'"), *TakeId);
		return false;
	}

	// Derive convenience fields from the timeline.
	OutRecord.DurationSec = OutRecord.Timeline.DurationSec;
	OutRecord.SampleRate  = OutRecord.Timeline.SampleRate;

	// Recompute summary.
	OutRecord.Summary = ComputeSummary(OutRecord.Timeline, OutRecord.SourceAudioPath);

	UE_LOG(LogEmotionBridge, Verbose,
		TEXT("FEmotionTakeStore: loaded '%s' (%s) — %d segments, dominant=%s"),
		*OutRecord.DisplayName, *OutRecord.TakeId,
		OutRecord.Timeline.Segments.Num(), *OutRecord.Summary.DominantEmotion);
	return true;
}

// ---------------------------------------------------------------------------
// LoadAllTakes
// ---------------------------------------------------------------------------

int32 FEmotionTakeStore::LoadAllTakes(TArray<FEmotionTakeRecord>& OutTakes)
{
	const FString Root = GetTakesRootDirectory();

	if (!IFileManager::Get().DirectoryExists(*Root))
	{
		UE_LOG(LogEmotionBridge, Log,
			TEXT("FEmotionTakeStore::LoadAllTakes — takes root does not exist yet: '%s'"), *Root);
		return 0;
	}

	// Collect all immediate subdirectory names (= take IDs).
	class FSubDirVisitor : public IPlatformFile::FDirectoryVisitor
	{
	public:
		TArray<FString> TakeIds;
		virtual bool Visit(const TCHAR* Filename, bool bIsDirectory) override
		{
			if (bIsDirectory)
				TakeIds.Add(FPaths::GetCleanFilename(Filename));
			return true; // continue
		}
	};

	FSubDirVisitor Visitor;
	FPlatformFileManager::Get().GetPlatformFile().IterateDirectory(*Root, Visitor);

	int32 Loaded = 0;
	for (const FString& Id : Visitor.TakeIds)
	{
		FEmotionTakeRecord Record;
		if (LoadTake(Id, Record))
		{
			OutTakes.Add(MoveTemp(Record));
			++Loaded;
		}
		else
		{
			UE_LOG(LogEmotionBridge, Warning,
				TEXT("FEmotionTakeStore::LoadAllTakes — skipping failed take folder '%s'"), *Id);
		}
	}

	UE_LOG(LogEmotionBridge, Log,
		TEXT("FEmotionTakeStore::LoadAllTakes — loaded %d / %d takes from '%s'"),
		Loaded, Visitor.TakeIds.Num(), *Root);
	return Loaded;
}

// ---------------------------------------------------------------------------
// DeleteTake
// ---------------------------------------------------------------------------

bool FEmotionTakeStore::DeleteTake(const FString& TakeId)
{
	const FString FolderPath = GetTakeFolderPath(TakeId);

	if (!IFileManager::Get().DirectoryExists(*FolderPath))
	{
		// Already gone — treat as success.
		return true;
	}

	const bool bOk = IFileManager::Get().DeleteDirectory(*FolderPath, /*RequireExists=*/false, /*Tree=*/true);
	if (bOk)
		UE_LOG(LogEmotionBridge, Log, TEXT("FEmotionTakeStore: deleted take '%s'"), *TakeId);
	else
		UE_LOG(LogEmotionBridge, Error,
			TEXT("FEmotionTakeStore: failed to delete folder '%s'"), *FolderPath);
	return bOk;
}

// ---------------------------------------------------------------------------
// DuplicateTake
// ---------------------------------------------------------------------------

bool FEmotionTakeStore::DuplicateTake(
	const FString& SourceTakeId,
	const FString& NewDisplayName,
	FEmotionTakeRecord& OutNewRecord)
{
	// 1. Load the source.
	FEmotionTakeRecord Source;
	if (!LoadTake(SourceTakeId, Source))
	{
		UE_LOG(LogEmotionBridge, Error,
			TEXT("FEmotionTakeStore::DuplicateTake — cannot load source take '%s'"), *SourceTakeId);
		return false;
	}

	// 2. Assign new identity.
	OutNewRecord             = Source;
	OutNewRecord.TakeId      = GenerateTakeId();
	OutNewRecord.DisplayName = NewDisplayName.IsEmpty()
		? Source.DisplayName + TEXT("_Copy")
		: NewDisplayName;
	OutNewRecord.UpdatedAt   = FDateTime::UtcNow().ToIso8601();
	// CreatedAt is intentionally copied from source to show the original origin.

	const FString NewFolderPath = GetTakeFolderPath(OutNewRecord.TakeId);
	if (!IFileManager::Get().MakeDirectory(*NewFolderPath, true))
	{
		UE_LOG(LogEmotionBridge, Error,
			TEXT("FEmotionTakeStore::DuplicateTake — cannot create folder '%s'"), *NewFolderPath);
		return false;
	}

	// 3. Write JSON files into the new folder.
	if (!WriteMetadataJson(OutNewRecord, NewFolderPath)) return false;
	if (!WriteTimelineJson(OutNewRecord.Timeline, NewFolderPath)) return false;
	if (!WriteParamsJson  (OutNewRecord.Params,   NewFolderPath)) return false;

	// 4. Copy audio if the source has embedded audio.
	if (!Source.CopiedAudioPath.IsEmpty())
	{
		const FString SrcAudio = Source.GetCopiedAudioFullPath();
		if (FPaths::FileExists(SrcAudio))
		{
			const FString AudioDir = NewFolderPath / TEXT("audio");
			const FString DstAudio = AudioDir      / TEXT("source.wav");
			IFileManager::Get().MakeDirectory(*AudioDir, true);
			if (IFileManager::Get().Copy(*DstAudio, *SrcAudio) == COPY_OK)
			{
				OutNewRecord.CopiedAudioPath = TEXT("audio/source.wav");
				WriteMetadataJson(OutNewRecord, NewFolderPath); // persist updated path
			}
		}
	}

	OutNewRecord.Summary = ComputeSummary(OutNewRecord.Timeline, OutNewRecord.SourceAudioPath);

	UE_LOG(LogEmotionBridge, Log,
		TEXT("FEmotionTakeStore: duplicated '%s' → '%s' (%s)"),
		*SourceTakeId, *OutNewRecord.DisplayName, *OutNewRecord.TakeId);
	return true;
}

// ===========================================================================
// Private — JSON write helpers
// ===========================================================================

bool FEmotionTakeStore::WriteMetadataJson(const FEmotionTakeRecord& Record, const FString& FolderPath)
{
	TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
	J->SetNumberField(TEXT("schema_version"), Record.SchemaVersion);
	J->SetStringField(TEXT("plugin_version"), Record.PluginVersion);
	J->SetStringField(TEXT("take_id"),         Record.TakeId);
	J->SetStringField(TEXT("display_name"),    Record.DisplayName);
	J->SetStringField(TEXT("created_at"),      Record.CreatedAt);
	J->SetStringField(TEXT("updated_at"),      Record.UpdatedAt);
	J->SetStringField(TEXT("source_audio_path"),  Record.SourceAudioPath);
	J->SetStringField(TEXT("copied_audio_path"),  Record.CopiedAudioPath);
	J->SetStringField(TEXT("notes"), Record.Notes);

	TArray<TSharedPtr<FJsonValue>> TagsArr;
	for (const FString& T : Record.Tags)
		TagsArr.Add(MakeShared<FJsonValueString>(T));
	J->SetArrayField(TEXT("tags"), TagsArr);

	TSharedRef<FJsonObject> P2B = MakeShared<FJsonObject>();
	P2B->SetStringField(TEXT("cleaned_audio_path"),          Record.Phase2B.CleanedAudioPath);
	P2B->SetStringField(TEXT("sound_wave_asset"),            Record.Phase2B.SoundWaveAssetPath);
	P2B->SetStringField(TEXT("metahuman_performance_asset"), Record.Phase2B.MetaHumanPerformancePath);
	P2B->SetStringField(TEXT("level_sequence_asset"),        Record.Phase2B.LevelSequencePath);
	P2B->SetStringField(TEXT("emotion_preset_mapping"),      Record.Phase2B.EmotionPresetMappingPath);
	J->SetObjectField(TEXT("phase2b"), P2B);

	FString Out;
	TSharedRef<TJsonWriter<>> W = TJsonWriterFactory<>::Create(&Out);
	FJsonSerializer::Serialize(J, W);
	return SafeWriteStringToFile(FolderPath / TEXT("metadata.json"), Out);
}

bool FEmotionTakeStore::WriteTimelineJson(const FEmotionTimelineResponse& TL, const FString& FolderPath)
{
	TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
	J->SetStringField(TEXT("type"),       TL.Type);
	J->SetStringField(TEXT("source"),     TL.Source);
	J->SetStringField(TEXT("version"),    TL.Version);
	J->SetStringField(TEXT("model_name"), TL.ModelName);
	J->SetNumberField(TEXT("sample_rate"), TL.SampleRate);
	J->SetNumberField(TEXT("duration_sec"), TL.DurationSec);

	TArray<TSharedPtr<FJsonValue>> SegsArr;
	for (const FEmotionSegment& Seg : TL.Segments)
	{
		TSharedRef<FJsonObject> S = MakeShared<FJsonObject>();
		S->SetNumberField(TEXT("start_sec"),  Seg.StartSec);
		S->SetNumberField(TEXT("end_sec"),    Seg.EndSec);
		S->SetStringField(TEXT("emotion"),    Seg.Emotion);
		S->SetNumberField(TEXT("confidence"), Seg.Confidence);
		SegsArr.Add(MakeShared<FJsonValueObject>(S));
	}
	J->SetArrayField(TEXT("segments"), SegsArr);

	FString Out;
	TSharedRef<TJsonWriter<>> W = TJsonWriterFactory<>::Create(&Out);
	FJsonSerializer::Serialize(J, W);
	return SafeWriteStringToFile(FolderPath / TEXT("timeline.json"), Out);
}

bool FEmotionTakeStore::WriteParamsJson(const FEmotionAnalysisParams& P, const FString& FolderPath)
{
	TSharedRef<FJsonObject> J = MakeShared<FJsonObject>();
	J->SetNumberField(TEXT("window_sec"),          P.WindowSec);
	J->SetNumberField(TEXT("hop_sec"),             P.HopSec);
	J->SetStringField(TEXT("pad_mode"),            P.PadMode);
	J->SetStringField(TEXT("smoothing_method"),    P.SmoothingMethod);
	J->SetNumberField(TEXT("hysteresis_min_run"),  P.HysteresisMinRun);
	J->SetNumberField(TEXT("majority_window"),     P.MajorityWindow);
	J->SetNumberField(TEXT("ema_alpha"),           P.EmaAlpha);

	FString Out;
	TSharedRef<TJsonWriter<>> W = TJsonWriterFactory<>::Create(&Out);
	FJsonSerializer::Serialize(J, W);
	return SafeWriteStringToFile(FolderPath / TEXT("params.json"), Out);
}

// ===========================================================================
// Private — JSON read helpers
// ===========================================================================

bool FEmotionTakeStore::ReadMetadataJson(const FString& FilePath, FEmotionTakeRecord& OutRecord)
{
	FString Content;
	if (!FFileHelper::LoadFileToString(Content, *FilePath))
	{
		UE_LOG(LogEmotionBridge, Warning,
			TEXT("FEmotionTakeStore: cannot read '%s'"), *FilePath);
		return false;
	}

	TSharedPtr<FJsonObject> J;
	auto Reader = TJsonReaderFactory<>::Create(Content);
	if (!FJsonSerializer::Deserialize(Reader, J) || !J.IsValid())
	{
		UE_LOG(LogEmotionBridge, Warning,
			TEXT("FEmotionTakeStore: invalid JSON in '%s'"), *FilePath);
		return false;
	}

	double V = 0.0;
	if (J->TryGetNumberField(TEXT("schema_version"), V)) OutRecord.SchemaVersion = static_cast<int32>(V);
	J->TryGetStringField(TEXT("plugin_version"),    OutRecord.PluginVersion);
	J->TryGetStringField(TEXT("take_id"),           OutRecord.TakeId);
	J->TryGetStringField(TEXT("display_name"),      OutRecord.DisplayName);
	J->TryGetStringField(TEXT("created_at"),        OutRecord.CreatedAt);
	J->TryGetStringField(TEXT("updated_at"),        OutRecord.UpdatedAt);
	J->TryGetStringField(TEXT("source_audio_path"), OutRecord.SourceAudioPath);
	J->TryGetStringField(TEXT("copied_audio_path"), OutRecord.CopiedAudioPath);
	J->TryGetStringField(TEXT("notes"),             OutRecord.Notes);

	if (OutRecord.TakeId.IsEmpty())
	{
		UE_LOG(LogEmotionBridge, Warning,
			TEXT("FEmotionTakeStore: metadata missing 'take_id' in '%s'"), *FilePath);
		return false;
	}

	const TArray<TSharedPtr<FJsonValue>>* TagsArr = nullptr;
	if (J->TryGetArrayField(TEXT("tags"), TagsArr) && TagsArr)
		for (const auto& TV : *TagsArr)
		{
			FString Tag; if (TV->TryGetString(Tag)) OutRecord.Tags.Add(Tag);
		}

	const TSharedPtr<FJsonObject>* P2BObj = nullptr;
	if (J->TryGetObjectField(TEXT("phase2b"), P2BObj) && P2BObj)
	{
		(*P2BObj)->TryGetStringField(TEXT("cleaned_audio_path"),          OutRecord.Phase2B.CleanedAudioPath);
		(*P2BObj)->TryGetStringField(TEXT("sound_wave_asset"),            OutRecord.Phase2B.SoundWaveAssetPath);
		(*P2BObj)->TryGetStringField(TEXT("metahuman_performance_asset"), OutRecord.Phase2B.MetaHumanPerformancePath);
		(*P2BObj)->TryGetStringField(TEXT("level_sequence_asset"),        OutRecord.Phase2B.LevelSequencePath);
		(*P2BObj)->TryGetStringField(TEXT("emotion_preset_mapping"),      OutRecord.Phase2B.EmotionPresetMappingPath);
	}

	return true;
}

bool FEmotionTakeStore::ReadTimelineJson(const FString& FilePath, FEmotionTimelineResponse& OutTL)
{
	FString Content;
	if (!FFileHelper::LoadFileToString(Content, *FilePath))
	{
		UE_LOG(LogEmotionBridge, Warning,
			TEXT("FEmotionTakeStore: cannot read '%s'"), *FilePath);
		return false;
	}

	TSharedPtr<FJsonObject> J;
	auto Reader = TJsonReaderFactory<>::Create(Content);
	if (!FJsonSerializer::Deserialize(Reader, J) || !J.IsValid())
	{
		UE_LOG(LogEmotionBridge, Warning,
			TEXT("FEmotionTakeStore: invalid JSON in '%s'"), *FilePath);
		return false;
	}

	J->TryGetStringField(TEXT("type"),       OutTL.Type);
	J->TryGetStringField(TEXT("source"),     OutTL.Source);
	J->TryGetStringField(TEXT("version"),    OutTL.Version);
	J->TryGetStringField(TEXT("model_name"), OutTL.ModelName);

	double V = 0.0;
	if (J->TryGetNumberField(TEXT("sample_rate"),  V)) OutTL.SampleRate  = static_cast<int32>(V);
	if (J->TryGetNumberField(TEXT("duration_sec"), V)) OutTL.DurationSec = static_cast<float>(V);

	const TArray<TSharedPtr<FJsonValue>>* SegsArr = nullptr;
	if (!J->TryGetArrayField(TEXT("segments"), SegsArr) || !SegsArr)
	{
		UE_LOG(LogEmotionBridge, Warning,
			TEXT("FEmotionTakeStore: no 'segments' array in '%s'"), *FilePath);
		return false;
	}

	for (const auto& SV : *SegsArr)
	{
		const TSharedPtr<FJsonObject>* SO = nullptr;
		if (!SV.IsValid() || !SV->TryGetObject(SO) || !SO) continue;

		FEmotionSegment Seg;
		double D = 0.0;
		if ((*SO)->TryGetNumberField(TEXT("start_sec"),  D)) Seg.StartSec   = static_cast<float>(D);
		if ((*SO)->TryGetNumberField(TEXT("end_sec"),    D)) Seg.EndSec     = static_cast<float>(D);
		if ((*SO)->TryGetNumberField(TEXT("confidence"), D)) Seg.Confidence = static_cast<float>(D);
		(*SO)->TryGetStringField(TEXT("emotion"), Seg.Emotion);
		OutTL.Segments.Add(Seg);
	}

	OutTL.bIsValid = true;
	return true;
}

bool FEmotionTakeStore::ReadParamsJson(const FString& FilePath, FEmotionAnalysisParams& OutP)
{
	FString Content;
	if (!FFileHelper::LoadFileToString(Content, *FilePath))
	{
		UE_LOG(LogEmotionBridge, Warning,
			TEXT("FEmotionTakeStore: cannot read '%s'"), *FilePath);
		return false;
	}

	TSharedPtr<FJsonObject> J;
	auto Reader = TJsonReaderFactory<>::Create(Content);
	if (!FJsonSerializer::Deserialize(Reader, J) || !J.IsValid()) return false;

	double V = 0.0;
	if (J->TryGetNumberField(TEXT("window_sec"),         V)) OutP.WindowSec        = static_cast<float>(V);
	if (J->TryGetNumberField(TEXT("hop_sec"),            V)) OutP.HopSec           = static_cast<float>(V);
	if (J->TryGetNumberField(TEXT("hysteresis_min_run"), V)) OutP.HysteresisMinRun = static_cast<int32>(V);
	if (J->TryGetNumberField(TEXT("majority_window"),    V)) OutP.MajorityWindow   = static_cast<int32>(V);
	if (J->TryGetNumberField(TEXT("ema_alpha"),          V)) OutP.EmaAlpha         = static_cast<float>(V);
	J->TryGetStringField(TEXT("pad_mode"),         OutP.PadMode);
	J->TryGetStringField(TEXT("smoothing_method"), OutP.SmoothingMethod);
	return true;
}

// ---------------------------------------------------------------------------
// SafeWriteStringToFile — temp-then-rename write
// ---------------------------------------------------------------------------

bool FEmotionTakeStore::SafeWriteStringToFile(const FString& FilePath, const FString& Content)
{
	const FString TempPath = FilePath + TEXT(".tmp");

	if (!FFileHelper::SaveStringToFile(
			Content, *TempPath,
			FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM))
	{
		IFileManager::Get().Delete(*TempPath);
		UE_LOG(LogEmotionBridge, Error,
			TEXT("FEmotionTakeStore: failed to write temp file '%s'"), *TempPath);
		return false;
	}

	// Rename temp → final (atomic on most file systems).
	if (!IFileManager::Get().Move(*FilePath, *TempPath, /*Replace=*/true, /*EvenIfReadOnly=*/true))
	{
		IFileManager::Get().Delete(*TempPath);
		UE_LOG(LogEmotionBridge, Error,
			TEXT("FEmotionTakeStore: rename failed '%s' → '%s'"), *TempPath, *FilePath);
		return false;
	}

	return true;
}
