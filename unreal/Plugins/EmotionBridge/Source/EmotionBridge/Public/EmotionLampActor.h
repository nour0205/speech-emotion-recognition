// Copyright (c) EmotionDemo Project. All rights reserved.
#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "EmotionLampActor.generated.h"

class UStaticMeshComponent;
class UPointLightComponent;
class UMaterialInstanceDynamic;

/**
 * AEmotionLampActor
 *
 * A simple demonstration actor consisting of a cube mesh and a point light.
 * Call ApplyEmotion() to change the light color (and mesh tint if the material
 * exposes a "BaseColor" vector parameter) according to the EmotionBridgeSettings color map.
 *
 * Place this actor in your editor world. The Emotion Bridge panel will find it
 * automatically via TActorIterator, or spawn a new one if none exists.
 */
UCLASS(BlueprintType, Blueprintable, meta=(DisplayName="Emotion Lamp Actor"))
class EMOTIONBRIDGE_API AEmotionLampActor : public AActor
{
	GENERATED_BODY()

public:
	AEmotionLampActor();

	/**
	 * Apply a visual color representing the given emotion.
	 * Drives both the point light color and, if available, a "BaseColor" vector
	 * parameter on the first material slot.
	 * Unknown emotion labels fall back to neutral (white) and log a warning.
	 *
	 * @param Emotion     Lowercase emotion string, e.g. "happy".
	 * @param Confidence  Value in [0,1] used to scale light intensity.
	 */
	UFUNCTION(BlueprintCallable, Category="EmotionBridge")
	void ApplyEmotion(const FString& Emotion, float Confidence = 1.0f);

	/** Reset to a neutral white state. Equivalent to ApplyEmotion("neutral", 1.0). */
	UFUNCTION(BlueprintCallable, Category="EmotionBridge")
	void ResetToNeutral();

	// -----------------------------------------------------------------------
	// Components
	// -----------------------------------------------------------------------

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Components")
	TObjectPtr<USceneComponent> RootSceneComponent;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Components")
	TObjectPtr<UStaticMeshComponent> MeshComponent;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Components")
	TObjectPtr<UPointLightComponent> PointLightComponent;

	// -----------------------------------------------------------------------
	// Tunables
	// -----------------------------------------------------------------------

	/** Maximum point light intensity (lumens). Scaled down at lower confidence values. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge", meta=(ClampMin="0.0"))
	float LightIntensityBase = 3000.f;

	/** Minimum fraction of LightIntensityBase applied at zero confidence. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge", meta=(ClampMin="0.0", ClampMax="1.0"))
	float LightIntensityMinFraction = 0.3f;

protected:
	virtual void BeginPlay() override;

private:
	/** Dynamic material instance created at BeginPlay to allow mesh tinting without asset changes. */
	UPROPERTY()
	TObjectPtr<UMaterialInstanceDynamic> DynamicMaterial;
};
