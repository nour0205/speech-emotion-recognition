// Copyright (c) EmotionDemo Project. All rights reserved.

#include "EmotionApiClient.h"
#include "EmotionBridgeLog.h"
#include "HttpModule.h"
#include "Interfaces/IHttpResponse.h"
#include "Json.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

FEmotionApiClient::FEmotionApiClient(const FString& InBaseUrl)
	: BaseUrl(InBaseUrl)
{
}

FEmotionApiClient::~FEmotionApiClient()
{
}

void FEmotionApiClient::SetBaseUrl(const FString& InBaseUrl)
{
	BaseUrl = InBaseUrl;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

TSharedRef<IHttpRequest, ESPMode::ThreadSafe> FEmotionApiClient::MakeRequest()
{
	return FHttpModule::Get().CreateRequest();
}

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------

void FEmotionApiClient::CheckHealth(FOnHealthCheckComplete Callback)
{
	TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Request = MakeRequest();
	Request->SetURL(BaseUrl + TEXT("/health"));
	Request->SetVerb(TEXT("GET"));
	Request->SetTimeout(5.f);

	Request->OnProcessRequestComplete().BindLambda(
		[Callback](FHttpRequestPtr /*Req*/, FHttpResponsePtr Response, bool bConnected)
		{
			const bool bHealthy = bConnected
				&& Response.IsValid()
				&& Response->GetResponseCode() == 200;

			if (!bConnected || !Response.IsValid())
			{
				UE_LOG(LogEmotionBridge, Warning,
					TEXT("Health check: backend unreachable (connection failed)."));
			}
			else
			{
				UE_LOG(LogEmotionBridge, Log,
					TEXT("Health check: HTTP %d"), Response->GetResponseCode());
			}

			Callback.ExecuteIfBound(bHealthy);
		}
	);

	Request->ProcessRequest();
}

// ---------------------------------------------------------------------------
// POST /timeline  (multipart/form-data)
// ---------------------------------------------------------------------------

void FEmotionApiClient::RequestTimeline(
	const FString& WavFilePath,
	float WindowSec,
	float HopSec,
	const FString& PadMode,
	const FString& SmoothingMethod,
	int32 HysteresisMinRun,
	bool bIncludeWindows,
	bool bIncludeScores,
	FOnTimelineComplete Callback)
{
	// ------------------------------------------------------------------
	// 1. Read the WAV file from disk into a byte buffer.
	// ------------------------------------------------------------------
	TArray<uint8> FileBytes;
	if (!FFileHelper::LoadFileToArray(FileBytes, *WavFilePath))
	{
		UE_LOG(LogEmotionBridge, Error,
			TEXT("RequestTimeline: cannot read file '%s'. Does it exist?"), *WavFilePath);

		FEmotionTimelineResponse ErrorResp;
		ErrorResp.bIsValid = false;
		ErrorResp.ErrorMessage = FString::Printf(
			TEXT("Cannot read WAV file: %s"), *WavFilePath);
		Callback.ExecuteIfBound(ErrorResp);
		return;
	}

	UE_LOG(LogEmotionBridge, Log,
		TEXT("RequestTimeline: read %d bytes from '%s'"), FileBytes.Num(), *WavFilePath);

	// ------------------------------------------------------------------
	// 2. Build the multipart/form-data body manually.
	//
	//    UE's HTTP module does not have a built-in multipart helper, so we
	//    construct the raw bytes ourselves according to RFC 2046 §5.1.1.
	// ------------------------------------------------------------------
	const FString Boundary = TEXT("----UnrealEmotionBridgeBoundary7MA4YWxkTrZu0gW");

	TArray<uint8> Body;
	Body.Reserve(FileBytes.Num() + 2048);

	// Helper: append a UTF-8 encoded FString into Body.
	auto AppendStr = [&Body](const FString& Str)
	{
		FTCHARToUTF8 Utf8(*Str);
		Body.Append(reinterpret_cast<const uint8*>(Utf8.Get()), Utf8.Length());
	};

	// Helper: append a named text field part.
	auto AddTextField = [&](const FString& FieldName, const FString& FieldValue)
	{
		AppendStr(FString::Printf(TEXT("--%s\r\n"), *Boundary));
		AppendStr(FString::Printf(
			TEXT("Content-Disposition: form-data; name=\"%s\"\r\n\r\n"), *FieldName));
		AppendStr(FieldValue);
		AppendStr(TEXT("\r\n"));
	};

	// --- File part ---
	const FString FileName = FPaths::GetCleanFilename(WavFilePath);
	AppendStr(FString::Printf(TEXT("--%s\r\n"), *Boundary));
	AppendStr(FString::Printf(
		TEXT("Content-Disposition: form-data; name=\"file\"; filename=\"%s\"\r\n"), *FileName));
	AppendStr(TEXT("Content-Type: audio/wav\r\n\r\n"));
	Body.Append(FileBytes);
	AppendStr(TEXT("\r\n"));

	// --- Optional text fields ---
	AddTextField(TEXT("window_sec"),         FString::Printf(TEXT("%.4f"), WindowSec));
	AddTextField(TEXT("hop_sec"),            FString::Printf(TEXT("%.4f"), HopSec));
	AddTextField(TEXT("pad_mode"),           PadMode);
	AddTextField(TEXT("smoothing_method"),   SmoothingMethod);
	AddTextField(TEXT("hysteresis_min_run"), FString::FromInt(HysteresisMinRun));
	AddTextField(TEXT("include_windows"),    bIncludeWindows ? TEXT("true") : TEXT("false"));
	AddTextField(TEXT("include_scores"),     bIncludeScores  ? TEXT("true") : TEXT("false"));

	// --- Closing boundary ---
	AppendStr(FString::Printf(TEXT("--%s--\r\n"), *Boundary));

	// ------------------------------------------------------------------
	// 3. Create and dispatch the HTTP request.
	// ------------------------------------------------------------------
	TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Request = MakeRequest();
	Request->SetURL(BaseUrl + TEXT("/timeline"));
	Request->SetVerb(TEXT("POST"));
	Request->SetHeader(
		TEXT("Content-Type"),
		FString::Printf(TEXT("multipart/form-data; boundary=%s"), *Boundary));
	// 180 s timeout — the first request may trigger a model download (~30–120 s).
	Request->SetTimeout(180.f);
	Request->SetContent(Body);

	Request->OnProcessRequestComplete().BindLambda(
		[Callback, WavFilePath](FHttpRequestPtr /*Req*/, FHttpResponsePtr Response, bool bConnected)
		{
			FEmotionTimelineResponse Result;

			// --- Connection-level failure ---
			if (!bConnected || !Response.IsValid())
			{
				UE_LOG(LogEmotionBridge, Error,
					TEXT("/timeline: connection failed. Is the backend running?"));
				Result.bIsValid    = false;
				Result.ErrorMessage = TEXT(
					"HTTP connection failed. Is the backend running at the configured URL?\n"
					"Start it with: cd speech-emotion-recognition && docker compose up api");
				Callback.ExecuteIfBound(Result);
				return;
			}

			const int32 HttpCode = Response->GetResponseCode();
			const FString BodyStr = Response->GetContentAsString();
			UE_LOG(LogEmotionBridge, Log, TEXT("/timeline: HTTP %d"), HttpCode);

			// --- HTTP-level error (4xx / 5xx) ---
			if (HttpCode != 200)
			{
				UE_LOG(LogEmotionBridge, Error,
					TEXT("/timeline error body: %s"), *BodyStr);
				Result.bIsValid = false;

				// Try to extract the structured error message from the JSON body.
				TSharedPtr<FJsonObject> ErrorJson;
				TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(BodyStr);
				if (FJsonSerializer::Deserialize(Reader, ErrorJson) && ErrorJson.IsValid())
				{
					const TSharedPtr<FJsonObject>* ErrorObj = nullptr;
					if (ErrorJson->TryGetObjectField(TEXT("error"), ErrorObj))
					{
						FString Msg;
						if ((*ErrorObj)->TryGetStringField(TEXT("message"), Msg))
						{
							Result.ErrorMessage = FString::Printf(
								TEXT("Backend error (HTTP %d): %s"), HttpCode, *Msg);
							Callback.ExecuteIfBound(Result);
							return;
						}
					}
				}

				Result.ErrorMessage = FString::Printf(
					TEXT("HTTP %d from backend. Body: %s"), HttpCode, *BodyStr.Left(256));
				Callback.ExecuteIfBound(Result);
				return;
			}

			// --- Parse the success JSON ---
			bool bParsed = false;
			Result = FEmotionApiClient::ParseTimelineResponse(BodyStr, bParsed);
			if (!bParsed)
			{
				UE_LOG(LogEmotionBridge, Error,
					TEXT("/timeline: JSON parse failed. Raw body (first 512): %s"),
					*BodyStr.Left(512));
			}
			Callback.ExecuteIfBound(Result);
		}
	);

	UE_LOG(LogEmotionBridge, Log,
		TEXT("Dispatching /timeline request (window=%.2f hop=%.2f pad=%s smooth=%s) for: %s"),
		WindowSec, HopSec, *PadMode, *SmoothingMethod, *WavFilePath);

	Request->ProcessRequest();
}

// ---------------------------------------------------------------------------
// JSON parsing
// ---------------------------------------------------------------------------

FEmotionTimelineResponse FEmotionApiClient::ParseTimelineResponse(
	const FString& JsonString, bool& bOutSuccess)
{
	FEmotionTimelineResponse Response;
	bOutSuccess = false;

	TSharedPtr<FJsonObject> Root;
	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonString);

	if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
	{
		Response.bIsValid    = false;
		Response.ErrorMessage = TEXT("Response body is not valid JSON.");
		return Response;
	}

	// Top-level scalar fields — use double for all numeric reads (UE5 JSON API).
	Root->TryGetStringField(TEXT("model_name"), Response.ModelName);

	double Tmp = 0.0;
	if (Root->TryGetNumberField(TEXT("sample_rate"), Tmp))
		Response.SampleRate = static_cast<int32>(FMath::RoundToInt(Tmp));
	if (Root->TryGetNumberField(TEXT("duration_sec"), Tmp))
		Response.DurationSec = static_cast<float>(Tmp);
	if (Root->TryGetNumberField(TEXT("window_sec"), Tmp))
		Response.WindowSec = static_cast<float>(Tmp);
	if (Root->TryGetNumberField(TEXT("hop_sec"), Tmp))
		Response.HopSec = static_cast<float>(Tmp);

	// Segments array
	const TArray<TSharedPtr<FJsonValue>>* SegmentsArray = nullptr;
	if (!Root->TryGetArrayField(TEXT("segments"), SegmentsArray) || !SegmentsArray)
	{
		Response.bIsValid    = false;
		Response.ErrorMessage = TEXT("Response JSON is missing the 'segments' array.");
		return Response;
	}

	for (const TSharedPtr<FJsonValue>& SegVal : *SegmentsArray)
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
		Response.Segments.Add(Seg);
	}

	UE_LOG(LogEmotionBridge, Log,
		TEXT("Parsed /timeline response: model=%s duration=%.2fs segments=%d"),
		*Response.ModelName, Response.DurationSec, Response.Segments.Num());

	Response.bIsValid = true;
	bOutSuccess       = true;
	return Response;
}
