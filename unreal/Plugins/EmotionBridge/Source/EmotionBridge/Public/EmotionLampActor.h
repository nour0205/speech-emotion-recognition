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
 * A glowing sphere + point light driven by emotion color.
 * Works in both the editor world (panel preview) and PIE/packaged builds.
 *
 * Visual strategy:
 *  - UPointLightComponent placed INSIDE the sphere illuminates the scene.
 *  - A UMaterialInstanceDynamic tints the sphere mesh itself.
 *    The "Color" / "BaseColor" parameter is set to an HDR value (> 1.0)
 *    so that UE5 Bloom creates a visible glow on the mesh surface.
 *  - Both are updated simultaneously by ApplyEmotion().
 *
 * OnConstruction() is overridden so the dynamic material is ready in the
 * editor world (BeginPlay is NOT called there).
 *
 * For the richest result, assign a material with an EmissiveColor or
 * BaseColor vector parameter to the sphere mesh. See MANUAL_ASSET_SETUP.md.
 */
UCLASS(BlueprintType, Blueprintable, meta=(DisplayName="Emotion Lamp Actor"))
class EMOTIONBRIDGE_API AEmotionLampActor : public AActor
{
	GENERATED_BODY()

public:
	AEmotionLampActor();

	/**
	 * Apply color for the given emotion to both the point light and mesh material.
	 * Unknown labels fall back to neutral white and log a warning.
	 * Safe to call in the editor world and during PIE.
	 *
	 * @param Emotion    Lowercase emotion label ("angry","happy","sad","neutral").
	 * @param Confidence [0,1] — scales light intensity.
	 */
	UFUNCTION(BlueprintCallable, Category="EmotionBridge")
	void ApplyEmotion(const FString& Emotion, float Confidence = 1.0f);

	/** Reset to neutral white. Equivalent to ApplyEmotion("neutral", 1.0). */
	UFUNCTION(BlueprintCallable, Category="EmotionBridge")
	void ResetToNeutral();

	// -----------------------------------------------------------------------
	// Components
	// -----------------------------------------------------------------------

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Components")
	TObjectPtr<USceneComponent> RootSceneComponent;

	/** Sphere mesh — represents the glowing bulb. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Components")
	TObjectPtr<UStaticMeshComponent> MeshComponent;

	/**
	 * Point light placed at the center of the sphere.
	 * Placing it INSIDE the mesh makes it appear as if the sphere itself emits light.
	 */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category="Components")
	TObjectPtr<UPointLightComponent> PointLightComponent;

	// -----------------------------------------------------------------------
	// Tunables
	// -----------------------------------------------------------------------

	/** Maximum point light intensity in lumens. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge", meta=(ClampMin="0"))
	float LightIntensityBase = 5000.f;

	/** Minimum intensity fraction at zero confidence. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge",
		meta=(ClampMin="0", ClampMax="1"))
	float LightIntensityMinFraction = 0.2f;

	/**
	 * HDR multiplier for the mesh material color parameter.
	 * Values > 1.0 create emissive bloom when post-processing is enabled.
	 * Default 4.0 produces strong bloom with the default Lumen/post-process settings.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge", meta=(ClampMin="1"))
	float MeshColorHDRMultiplier = 4.0f;

protected:
	/** Called in BOTH editor and PIE — used to initialise the dynamic material. */
	virtual void OnConstruction(const FTransform& Transform) override;

	virtual void BeginPlay() override;

private:
	/** Dynamic material instance for mesh color/emissive tinting. */
	UPROPERTY()
	TObjectPtr<UMaterialInstanceDynamic> DynamicMaterial;

	/** Creates DynamicMaterial from the mesh's slot 0 if not already done. */
	void EnsureDynamicMaterial();
};
