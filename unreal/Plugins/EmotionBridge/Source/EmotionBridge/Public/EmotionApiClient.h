// Copyright (c) EmotionDemo Project. All rights reserved.
#pragma once

#include "CoreMinimal.h"
#include "EmotionTimelineTypes.h"
#include "Http.h"

/** Called on the game thread when the /timeline/unreal request completes. */
DECLARE_DELEGATE_OneParam(FOnTimelineComplete, const FEmotionTimelineResponse& /*Response*/);

/** Called on the game thread when the /health check completes. */
DECLARE_DELEGATE_OneParam(FOnHealthCheckComplete, bool /*bIsHealthy*/);

/**
 * HTTP client for the speech-emotion-recognition backend.
 *
 * Endpoints used:
 *   GET  /health                — liveness check (5 s timeout)
 *   POST /timeline/unreal       — Unreal-specific timeline contract (180 s timeout)
 *
 * The /timeline/unreal endpoint returns a simplified JSON envelope:
 *   { "type": "timeline", "source": "ser_api", "version": "1.0",
 *     "duration_sec": N, "segments": [ { start_sec, end_sec, emotion, confidence } ] }
 *
 * All callbacks are delivered on the game thread (UE HTTP module guarantee).
 */
class EMOTIONBRIDGE_API FEmotionApiClient
{
public:
	explicit FEmotionApiClient(const FString& InBaseUrl);
	~FEmotionApiClient();

	/** Update the base URL without recreating the client. */
	void SetBaseUrl(const FString& InBaseUrl);

	/**
	 * POST /timeline/unreal with the WAV file as multipart/form-data.
	 *
	 * Reads the file from disk synchronously before dispatching the async request.
	 * Timeout is 180 s to accommodate first-run model downloads (~30–120 s).
	 *
	 * Optional smoothing parameters default to backend settings when omitted.
	 */
	void RequestTimeline(
		const FString& WavFilePath,
		float WindowSec,
		float HopSec,
		const FString& PadMode,
		const FString& SmoothingMethod,
		int32 HysteresisMinRun,
		float MajorityWindow,
		float EmaAlpha,
		FOnTimelineComplete Callback
	);

	/** GET /health — lightweight liveness check, 5 s timeout. */
	void CheckHealth(FOnHealthCheckComplete Callback);

private:
	FString BaseUrl;

	static TSharedRef<IHttpRequest, ESPMode::ThreadSafe> MakeRequest();
	static FEmotionTimelineResponse ParseTimelineResponse(const FString& JsonString, bool& bOutSuccess);
};
