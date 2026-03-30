// Copyright (c) EmotionDemo Project. All rights reserved.

#include "EmotionApiClient.h"
#include "EmotionBridgeLog.h"
#include "HttpModule.h"
#include "Interfaces/IHttpResponse.h"
#include "Json.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

FEmotionApiClient::FEmotionApiClient(const FString& InBaseUrl) : BaseUrl(InBaseUrl) {}
FEmotionApiClient::~FEmotionApiClient() {}

void FEmotionApiClient::SetBaseUrl(const FString& InBaseUrl) { BaseUrl = InBaseUrl; }

TSharedRef<IHttpRequest, ESPMode::ThreadSafe> FEmotionApiClient::MakeRequest()
{
	return FHttpModule::Get().CreateRequest();
}

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------

void FEmotionApiClient::CheckHealth(FOnHealthCheckComplete Callback)
{
	auto Request = MakeRequest();
	Request->SetURL(BaseUrl + TEXT("/health"));
	Request->SetVerb(TEXT("GET"));
	Request->SetTimeout(5.f);

	Request->OnProcessRequestComplete().BindLambda(
		[Callback](FHttpRequestPtr, FHttpResponsePtr Response, bool bConnected)
		{
			const bool bHealthy = bConnected && Response.IsValid()
				&& Response->GetResponseCode() == 200;
			if (!bHealthy)
			{
				UE_LOG(LogEmotionBridge, Warning,
					TEXT("Health check failed — is the backend running?"));
			}
			else
			{
				UE_LOG(LogEmotionBridge, Log, TEXT("Health check OK (HTTP 200)."));
			}
			Callback.ExecuteIfBound(bHealthy);
		});

	Request->ProcessRequest();
}

// ---------------------------------------------------------------------------
// POST /timeline/unreal  (multipart/form-data)
// ---------------------------------------------------------------------------

void FEmotionApiClient::RequestTimeline(
	const FString& WavFilePath,
	float WindowSec,
	float HopSec,
	const FString& PadMode,
	const FString& SmoothingMethod,
	int32 HysteresisMinRun,
	float MajorityWindow,
	float EmaAlpha,
	FOnTimelineComplete Callback)
{
	// 1. Read WAV bytes.
	TArray<uint8> FileBytes;
	if (!FFileHelper::LoadFileToArray(FileBytes, *WavFilePath))
	{
		UE_LOG(LogEmotionBridge, Error,
			TEXT("RequestTimeline: cannot read '%s'"), *WavFilePath);
		FEmotionTimelineResponse Err;
		Err.bIsValid    = false;
		Err.ErrorMessage = FString::Printf(TEXT("Cannot read WAV file: %s"), *WavFilePath);
		Callback.ExecuteIfBound(Err);
		return;
	}

	// 2. Build multipart/form-data body manually (RFC 2046 §5.1.1).
	const FString Boundary = TEXT("----UnrealEmotionBridgeBoundary7MA4YWxkTrZu0gW");
	TArray<uint8> Body;
	Body.Reserve(FileBytes.Num() + 4096);

	auto AppendStr = [&Body](const FString& S)
	{
		FTCHARToUTF8 U(*S);
		Body.Append(reinterpret_cast<const uint8*>(U.Get()), U.Length());
	};
	auto AddTextField = [&](const FString& Name, const FString& Value)
	{
		AppendStr(FString::Printf(TEXT("--%s\r\n"), *Boundary));
		AppendStr(FString::Printf(
			TEXT("Content-Disposition: form-data; name=\"%s\"\r\n\r\n"), *Name));
		AppendStr(Value);
		AppendStr(TEXT("\r\n"));
	};

	// File part.
	const FString FileName = FPaths::GetCleanFilename(WavFilePath);
	AppendStr(FString::Printf(TEXT("--%s\r\n"), *Boundary));
	AppendStr(FString::Printf(
		TEXT("Content-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n"), *FileName));
	AppendStr(TEXT("Content-Type: audio/wav\r\n\r\n"));
	Body.Append(FileBytes);
	AppendStr(TEXT("\r\n"));

	// Timeline parameter fields.
	AddTextField(TEXT("window_sec"),         FString::Printf(TEXT("%.4f"), WindowSec));
	AddTextField(TEXT("hop_sec"),            FString::Printf(TEXT("%.4f"), HopSec));
	AddTextField(TEXT("pad_mode"),           PadMode);
	AddTextField(TEXT("smoothing_method"),   SmoothingMethod);
	AddTextField(TEXT("hysteresis_min_run"), FString::FromInt(HysteresisMinRun));
	AddTextField(TEXT("majority_window"),    FString::FromInt(static_cast<int32>(MajorityWindow)));
	AddTextField(TEXT("ema_alpha"),          FString::Printf(TEXT("%.4f"), EmaAlpha));

	AppendStr(FString::Printf(TEXT("--%s--\r\n"), *Boundary));

	// 3. Send request to the Unreal-specific endpoint.
	auto Request = MakeRequest();
	Request->SetURL(BaseUrl + TEXT("/timeline/unreal"));
	Request->SetVerb(TEXT("POST"));
	Request->SetHeader(TEXT("Content-Type"),
		FString::Printf(TEXT("multipart/form-data; boundary=%s"), *Boundary));
	Request->SetTimeout(180.f); // model download can take ~2 min on first run
	Request->SetContent(Body);

	Request->OnProcessRequestComplete().BindLambda(
		[Callback, WavFilePath](FHttpRequestPtr, FHttpResponsePtr Response, bool bConnected)
		{
			FEmotionTimelineResponse Result;

			if (!bConnected || !Response.IsValid())
			{
				UE_LOG(LogEmotionBridge, Error,
					TEXT("/timeline/unreal: connection failed. Is the backend running?"));
				Result.bIsValid    = false;
				Result.ErrorMessage = TEXT(
					"HTTP connection failed. Start the backend:\n"
					"  docker compose up api\nor\n"
					"  uvicorn src.api.main:app --port 8000");
				Callback.ExecuteIfBound(Result);
				return;
			}

			const int32 Code     = Response->GetResponseCode();
			const FString BodyStr = Response->GetContentAsString();
			UE_LOG(LogEmotionBridge, Log, TEXT("/timeline/unreal HTTP %d"), Code);

			if (Code != 200)
			{
				UE_LOG(LogEmotionBridge, Error,
					TEXT("/timeline/unreal error body: %s"), *BodyStr);
				Result.bIsValid = false;

				// Try to extract structured backend error message.
				TSharedPtr<FJsonObject> ErrJson;
				auto Reader = TJsonReaderFactory<>::Create(BodyStr);
				if (FJsonSerializer::Deserialize(Reader, ErrJson) && ErrJson.IsValid())
				{
					const TSharedPtr<FJsonObject>* ErrObj;
					if (ErrJson->TryGetObjectField(TEXT("error"), ErrObj))
					{
						FString Msg;
						if ((*ErrObj)->TryGetStringField(TEXT("message"), Msg))
						{
							Result.ErrorMessage = FString::Printf(
								TEXT("Backend (HTTP %d): %s"), Code, *Msg);
							Callback.ExecuteIfBound(Result);
							return;
						}
					}
				}
				Result.ErrorMessage = FString::Printf(
					TEXT("HTTP %d. Body: %s"), Code, *BodyStr.Left(200));
				Callback.ExecuteIfBound(Result);
				return;
			}

			bool bOk = false;
			Result = ParseTimelineResponse(BodyStr, bOk);
			if (!bOk)
			{
				UE_LOG(LogEmotionBridge, Error,
					TEXT("JSON parse failed. Raw (first 512 chars): %s"),
					*BodyStr.Left(512));
			}
			Callback.ExecuteIfBound(Result);
		});

	UE_LOG(LogEmotionBridge, Log,
		TEXT("POST /timeline/unreal  file=%s  window=%.2f  hop=%.2f  pad=%s  smooth=%s"),
		*WavFilePath, WindowSec, HopSec, *PadMode, *SmoothingMethod);

	Request->ProcessRequest();
}

// ---------------------------------------------------------------------------
// JSON parsing for /timeline/unreal response
// ---------------------------------------------------------------------------

FEmotionTimelineResponse FEmotionApiClient::ParseTimelineResponse(
	const FString& JsonString, bool& bOutSuccess)
{
	FEmotionTimelineResponse R;
	bOutSuccess = false;

	TSharedPtr<FJsonObject> Root;
	auto Reader = TJsonReaderFactory<>::Create(JsonString);
	if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
	{
		R.bIsValid    = false;
		R.ErrorMessage = TEXT("Response is not valid JSON.");
		return R;
	}

	// Envelope fields.
	Root->TryGetStringField(TEXT("type"),       R.Type);
	Root->TryGetStringField(TEXT("source"),     R.Source);
	Root->TryGetStringField(TEXT("version"),    R.Version);
	Root->TryGetStringField(TEXT("model_name"), R.ModelName); // Phase 2A — stored in takes

	double Tmp = 0.0;
	if (Root->TryGetNumberField(TEXT("sample_rate"),  Tmp))   // Phase 2A — stored in takes
		R.SampleRate = static_cast<int32>(Tmp);
	if (Root->TryGetNumberField(TEXT("duration_sec"), Tmp))
		R.DurationSec = static_cast<float>(Tmp);

	// Segments array.
	const TArray<TSharedPtr<FJsonValue>>* SegsArr = nullptr;
	if (!Root->TryGetArrayField(TEXT("segments"), SegsArr) || !SegsArr)
	{
		R.bIsValid    = false;
		R.ErrorMessage = TEXT("Response is missing the 'segments' array.");
		return R;
	}

	for (const auto& SegVal : *SegsArr)
	{
		const TSharedPtr<FJsonObject>* SegObj = nullptr;
		if (!SegVal.IsValid() || !SegVal->TryGetObject(SegObj) || !SegObj)
			continue;

		FEmotionSegment Seg;
		double D = 0.0;
		if ((*SegObj)->TryGetNumberField(TEXT("start_sec"),  D)) Seg.StartSec   = static_cast<float>(D);
		if ((*SegObj)->TryGetNumberField(TEXT("end_sec"),    D)) Seg.EndSec     = static_cast<float>(D);
		if ((*SegObj)->TryGetNumberField(TEXT("confidence"), D)) Seg.Confidence = static_cast<float>(D);
		(*SegObj)->TryGetStringField(TEXT("emotion"), Seg.Emotion);
		R.Segments.Add(Seg);
	}

	UE_LOG(LogEmotionBridge, Log,
		TEXT("Parsed /timeline/unreal: type=%s source=%s model=%s sr=%d duration=%.2f segments=%d"),
		*R.Type, *R.Source, *R.ModelName, R.SampleRate, R.DurationSec, R.Segments.Num());

	R.bIsValid = true;
	bOutSuccess = true;
	return R;
}
