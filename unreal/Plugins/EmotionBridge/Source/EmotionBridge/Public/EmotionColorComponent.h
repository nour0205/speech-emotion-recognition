// Copyright (c) EmotionDemo Project. All rights reserved.
#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "EmotionColorComponent.generated.h"

class UMeshComponent;
class UMaterialInstanceDynamic;

/**
 * UEmotionColorComponent
 *
 * Add this component to ANY actor (character, static mesh, prop, Blueprint…)
 * to make it respond to emotion timeline playback from the Emotion Bridge panel.
 *
 * QUICK SETUP (3 steps, no C++ required):
 *   1. Select your actor in the viewport.
 *   2. Details panel → + Add Component → search "Emotion Color" → click it.
 *   3. Click "Play Demo" in the Emotion Bridge panel — the mesh changes color.
 *
 * HOW IT WORKS:
 *   - Automatically detects the first SkeletalMesh or StaticMesh on the owner.
 *   - Creates a UMaterialInstanceDynamic at startup so no manual material setup
 *     is needed for basic usage.
 *   - Sets "BaseColor", "Color", "EmissiveColor", and "Emissive" vector params so
 *     it works with most built-in and custom materials.
 *   - Also tints any UPointLightComponent on the same actor (optional).
 *
 * For richer visuals assign an emissive material — see MANUAL_ASSET_SETUP.md.
 */
UCLASS(ClassGroup=EmotionBridge, BlueprintType, Blueprintable,
	meta=(BlueprintSpawnableComponent, DisplayName="Emotion Color"))
class EMOTIONBRIDGE_API UEmotionColorComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UEmotionColorComponent();

	// -----------------------------------------------------------------------
	// Settings (editable in Details panel)
	// -----------------------------------------------------------------------

	/**
	 * The mesh to tint. Leave empty to auto-detect the first
	 * SkeletalMeshComponent or StaticMeshComponent on the owner actor.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge")
	TObjectPtr<UMeshComponent> TargetMesh;

	/** Material slot index to tint. 0 = first slot (almost always correct). */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge",
		meta=(ClampMin="0", DisplayName="Material Slot"))
	int32 MaterialSlot = 0;

	/**
	 * Multiplier applied to the emotion color before writing it to the material.
	 * Values > 1.0 produce emissive bloom in Lumen / post-process pipelines.
	 * Use 1.0 for Unlit materials; 4.0 for Lit materials with Bloom enabled.
	 */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge",
		meta=(ClampMin="0.1", DisplayName="Color HDR Multiplier"))
	float HDRMultiplier = 4.0f;

	/** If true, also tints any UPointLightComponent found on the same actor. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="EmotionBridge",
		DisplayName="Tint Point Light")
	bool bTintPointLight = true;

	// -----------------------------------------------------------------------
	// API (callable from Blueprint or C++)
	// -----------------------------------------------------------------------

	/**
	 * Apply the color for the given emotion to the target mesh material.
	 * Also updates the point light if bTintPointLight is true.
	 *
	 * @param Emotion    Lowercase label ("happy", "angry", "sad", "neutral").
	 * @param Confidence [0,1] — scales light intensity.
	 */
	UFUNCTION(BlueprintCallable, Category="EmotionBridge")
	void ApplyEmotion(const FString& Emotion, float Confidence = 1.0f);

	/** Reset to neutral white. */
	UFUNCTION(BlueprintCallable, Category="EmotionBridge")
	void ResetToNeutral();

protected:
	// OnRegister fires in BOTH editor world and PIE (unlike BeginPlay).
	virtual void OnRegister() override;
	virtual void BeginPlay() override;

private:
	UPROPERTY()
	TObjectPtr<UMaterialInstanceDynamic> DynamicMaterial;

	/** Create/refresh the dynamic material from the resolved mesh slot. */
	void EnsureDynamicMaterial();

	/** Returns TargetMesh if set, otherwise auto-detects from the owner. */
	UMeshComponent* ResolveMesh() const;
};
