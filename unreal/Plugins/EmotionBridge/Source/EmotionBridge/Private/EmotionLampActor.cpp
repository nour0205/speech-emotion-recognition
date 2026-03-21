// Copyright (c) EmotionDemo Project. All rights reserved.

#include "EmotionLampActor.h"
#include "EmotionBridgeSettings.h"
#include "EmotionBridgeLog.h"
#include "Components/StaticMeshComponent.h"
#include "Components/PointLightComponent.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "UObject/ConstructorHelpers.h"

AEmotionLampActor::AEmotionLampActor()
{
	PrimaryActorTick.bCanEverTick = false;

	// Root
	RootSceneComponent = CreateDefaultSubobject<USceneComponent>(TEXT("RootScene"));
	SetRootComponent(RootSceneComponent);

	// Sphere mesh — a sphere looks like a lamp bulb; the cube was replaced.
	MeshComponent = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Mesh"));
	MeshComponent->SetupAttachment(RootSceneComponent);

	static ConstructorHelpers::FObjectFinder<UStaticMesh> SphereFinder(
		TEXT("/Engine/BasicShapes/Sphere.Sphere"));
	if (SphereFinder.Succeeded())
	{
		MeshComponent->SetStaticMesh(SphereFinder.Object);
	}
	else
	{
		// Fallback to cube if sphere not found.
		static ConstructorHelpers::FObjectFinder<UStaticMesh> CubeFinder(
			TEXT("/Engine/BasicShapes/Cube.Cube"));
		if (CubeFinder.Succeeded())
			MeshComponent->SetStaticMesh(CubeFinder.Object);
	}

	// Scale the sphere to ~40 cm diameter — a visible but not huge bulb.
	MeshComponent->SetRelativeScale3D(FVector(0.4f));

	// Point light INSIDE the sphere — this is what makes the sphere appear to glow.
	// With the light at the center, the surrounding geometry is illuminated in the
	// emotion color, and the sphere mesh catches the light from the inside when
	// viewed with Lumen or Screen-Space GI.
	PointLightComponent = CreateDefaultSubobject<UPointLightComponent>(TEXT("PointLight"));
	PointLightComponent->SetupAttachment(RootSceneComponent);
	PointLightComponent->SetRelativeLocation(FVector::ZeroVector); // center of sphere
	PointLightComponent->SetIntensity(LightIntensityBase);
	PointLightComponent->SetLightColor(FLinearColor::White);
	PointLightComponent->SetAttenuationRadius(800.f);
	PointLightComponent->SetCastShadows(true);

	// SourceRadius makes UE5 render the light as a visible physical sphere.
	// This is the key to seeing a "glowing ball" even without a custom emissive material.
	PointLightComponent->SourceRadius      = 20.f;  // 20 cm emissive sphere radius
	PointLightComponent->SoftSourceRadius  = 30.f;  // soft penumbra halo
	PointLightComponent->SourceLength      = 0.f;   // sphere, not capsule
}

// ---------------------------------------------------------------------------
// OnConstruction — called in BOTH editor placement AND PIE start.
// This is where we create the dynamic material so it's always valid.
// ---------------------------------------------------------------------------

void AEmotionLampActor::OnConstruction(const FTransform& Transform)
{
	Super::OnConstruction(Transform);
	EnsureDynamicMaterial();
	// Apply neutral white on first construction so the actor has a consistent default.
	ResetToNeutral();
}

void AEmotionLampActor::BeginPlay()
{
	Super::BeginPlay();
	// Ensure material is ready in PIE as well (OnConstruction also runs, but be safe).
	EnsureDynamicMaterial();
	ResetToNeutral();
}

// ---------------------------------------------------------------------------

void AEmotionLampActor::EnsureDynamicMaterial()
{
	if (DynamicMaterial)
		return; // already created

	if (!MeshComponent)
		return;

	UMaterialInterface* BaseMat = MeshComponent->GetMaterial(0);
	if (!BaseMat)
		return;

	DynamicMaterial = MeshComponent->CreateAndSetMaterialInstanceDynamic(0);
	UE_LOG(LogEmotionBridge, Log,
		TEXT("Lamp [%s]: DynamicMaterial created from '%s'."),
		*GetName(), *BaseMat->GetName());
}

void AEmotionLampActor::ApplyEmotion(const FString& Emotion, float Confidence)
{
	// Resolve color from settings.
	const UEmotionBridgeSettings* Settings = UEmotionBridgeSettings::Get();
	const FLinearColor Color = Settings
		? Settings->GetColorForEmotion(Emotion)
		: FLinearColor::White;

	// --- Point light ---
	const float IntensityScale = FMath::Lerp(
		LightIntensityMinFraction, 1.f, FMath::Clamp(Confidence, 0.f, 1.f));

	if (PointLightComponent)
	{
		PointLightComponent->SetLightColor(Color);
		PointLightComponent->SetIntensity(LightIntensityBase * IntensityScale);
	}

	// --- Mesh material tint ---
	// Lazily create the dynamic material if OnConstruction hasn't run yet
	// (e.g. when auto-spawned from the editor panel).
	EnsureDynamicMaterial();

	if (DynamicMaterial)
	{
		// Use an HDR color value so UE5 Bloom makes the sphere surface appear to glow.
		// The multiplier only affects how bright the mesh looks — the light color is
		// always set to the canonical [0,1] range.
		const FLinearColor HDRColor = FLinearColor(
			Color.R * MeshColorHDRMultiplier,
			Color.G * MeshColorHDRMultiplier,
			Color.B * MeshColorHDRMultiplier,
			1.f);

		// Try the most common material parameter names.
		// One of these will match whatever material is assigned to the sphere.
		DynamicMaterial->SetVectorParameterValue(TEXT("BaseColor"),    HDRColor);
		DynamicMaterial->SetVectorParameterValue(TEXT("Color"),        HDRColor);
		DynamicMaterial->SetVectorParameterValue(TEXT("EmissiveColor"), HDRColor);
		DynamicMaterial->SetVectorParameterValue(TEXT("Emissive"),     HDRColor);
	}

	UE_LOG(LogEmotionBridge, Verbose,
		TEXT("Lamp [%s]: emotion=%s conf=%.2f color=(%.2f,%.2f,%.2f) intensity=%.0f"),
		*GetName(), *Emotion, Confidence,
		Color.R, Color.G, Color.B,
		LightIntensityBase * IntensityScale);
}

void AEmotionLampActor::ResetToNeutral()
{
	ApplyEmotion(TEXT("neutral"), 1.0f);
}
