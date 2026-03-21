// Copyright (c) EmotionDemo Project. All rights reserved.
#pragma once

#include "CoreMinimal.h"
#include "EmotionTimelineTypes.h"
#include "Http.h"

/** Called on the game thread when the /timeline request completes (success or failure). */
DECLARE_DELEGATE_OneParam(FOnTimelineComplete, const FEmotionTimelineResponse& /*Response*/);

/** Called on the game thread when the /health check completes. */
DECLARE_DELEGATE_OneParam(FOnHealthCheckComplete, bool /*bIsHealthy*/);

/**
 * Thin HTTP client that wraps the two relevant endpoints of the speech-emotion-recognition backend:
 *   GET  /health
 *   POST /timeline   (multipart/form-data)
 *
 * All callbacks are delivered on the game thread (UE HTTP module guarantee).
 * Create one instance per panel/component; call SetBaseUrl to switch endpoints at runtime.
 */
class EMOTIONBRIDGE_API FEmotionApiClient
{
public:
	explicit FEmotionApiClient(const FString& InBaseUrl);
	~FEmotionApiClient();

	/** Update the backend base URL without recreating the client. */
	void SetBaseUrl(const FString& InBaseUrl);

	/**
	 * POST /timeline with the WAV file as multipart/form-data plus optional parameters.
	 * Reads the file from disk synchronously before dispatching the async HTTP request.
	 *
	 * @param WavFilePath   Absolute path to the WAV file.
	 * @param Callback      Invoked on game thread with the parsed response (or an error response).
	 *
	 * NOTE: The first call may take 30–120 s if the model has not been downloaded yet.
	 *       The HTTP timeout is set to 180 s to accommodate this.
	 */
	void RequestTimeline(
		const FString& WavFilePath,
		float WindowSec,
		float HopSec,
		const FString& PadMode,
		const FString& SmoothingMethod,
		int32 HysteresisMinRun,
		bool bIncludeWindows,
		bool bIncludeScores,
		FOnTimelineComplete Callback
	);

	/**
	 * GET /health — lightweight liveness check.
	 * Times out after 5 s.
	 */
	void CheckHealth(FOnHealthCheckComplete Callback);

private:
	FString BaseUrl;

	/** Build an IHttpRequest with the module's default settings. */
	static TSharedRef<IHttpRequest, ESPMode::ThreadSafe> MakeRequest();

	/** Deserialize the /timeline JSON body into FEmotionTimelineResponse. */
	static FEmotionTimelineResponse ParseTimelineResponse(const FString& JsonString, bool& bOutSuccess);
};
