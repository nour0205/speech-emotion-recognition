// Copyright (c) EmotionDemo Project. All rights reserved.

#include "EmotionColorComponent.h"
#include "EmotionBridgeSettings.h"
#include "EmotionBridgeLog.h"

#include "Components/StaticMeshComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "Components/PointLightComponent.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Materials/Material.h"

UEmotionColorComponent::UEmotionColorComponent()
{
	PrimaryComponentTick.bCanEverTick = false;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void UEmotionColorComponent::OnRegister()
{
	Super::OnRegister();
	// OnRegister is called in both the editor world AND PIE,
	// so the dynamic material is always ready.
	EnsureDynamicMaterial();
}

void UEmotionColorComponent::BeginPlay()
{
	Super::BeginPlay();
	EnsureDynamicMaterial();
	ResetToNeutral();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

UMeshComponent* UEmotionColorComponent::ResolveMesh() const
{
	if (TargetMesh)
		return TargetMesh;

	AActor* Owner = GetOwner();
	if (!Owner) return nullptr;

	// Prefer SkeletalMesh (characters/MetaHumans) then StaticMesh.
	if (USkeletalMeshComponent* Skel = Owner->FindComponentByClass<USkeletalMeshComponent>())
		return Skel;
	if (UStaticMeshComponent* Stat = Owner->FindComponentByClass<UStaticMeshComponent>())
		return Stat;

	return nullptr;
}

void UEmotionColorComponent::EnsureDynamicMaterial()
{
	if (DynamicMaterial) return;

	UMeshComponent* Mesh = ResolveMesh();
	if (!Mesh) return;

	UMaterialInterface* BaseMat = Mesh->GetMaterial(MaterialSlot);
	if (!BaseMat)
	{
		// Fall back to the engine's BasicShapeMaterial which has both
		// a "Color" and "BaseColor" vector parameter — no asset creation needed.
		BaseMat = LoadObject<UMaterial>(
			nullptr, TEXT("/Engine/BasicShapes/BasicShapeMaterial"));
		if (BaseMat)
			Mesh->SetMaterial(MaterialSlot, BaseMat);
	}

	if (!BaseMat) return;

	DynamicMaterial = UMaterialInstanceDynamic::Create(BaseMat, this);
	Mesh->SetMaterial(MaterialSlot, DynamicMaterial);

	UE_LOG(LogEmotionBridge, Log,
		TEXT("EmotionColorComponent [%s]: DynamicMaterial created from '%s'."),
		GetOwner() ? *GetOwner()->GetName() : TEXT("?"),
		*BaseMat->GetName());
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

void UEmotionColorComponent::ApplyEmotion(const FString& Emotion, float Confidence)
{
	EnsureDynamicMaterial();

	const UEmotionBridgeSettings* Settings = UEmotionBridgeSettings::Get();
	const FLinearColor Color = Settings
		? Settings->GetColorForEmotion(Emotion)
		: FLinearColor::White;

	// --- Mesh tint ---
	if (DynamicMaterial)
	{
		const FLinearColor HDR(
			Color.R * HDRMultiplier,
			Color.G * HDRMultiplier,
			Color.B * HDRMultiplier,
			1.f);

		// Try all common material parameter names so this works with
		// built-in materials, custom materials, and emissive materials.
		DynamicMaterial->SetVectorParameterValue(TEXT("BaseColor"),    HDR);
		DynamicMaterial->SetVectorParameterValue(TEXT("Color"),        HDR);
		DynamicMaterial->SetVectorParameterValue(TEXT("EmissiveColor"), HDR);
		DynamicMaterial->SetVectorParameterValue(TEXT("Emissive"),     HDR);
	}

	// --- Point light tint ---
	if (bTintPointLight)
	{
		if (AActor* Owner = GetOwner())
		{
			if (UPointLightComponent* Light =
				Owner->FindComponentByClass<UPointLightComponent>())
			{
				Light->SetLightColor(Color);
				const float Scale =
					FMath::Lerp(0.2f, 1.f, FMath::Clamp(Confidence, 0.f, 1.f));
				Light->SetIntensity(5000.f * Scale);
			}
		}
	}
}

void UEmotionColorComponent::ResetToNeutral()
{
	ApplyEmotion(TEXT("neutral"), 1.0f);
}
