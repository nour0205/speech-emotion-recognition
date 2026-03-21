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

	// Mesh — default to the engine's built-in Cube so no extra content is required.
	MeshComponent = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("Mesh"));
	MeshComponent->SetupAttachment(RootSceneComponent);

	static ConstructorHelpers::FObjectFinder<UStaticMesh> CubeFinder(
		TEXT("/Engine/BasicShapes/Cube.Cube"));
	if (CubeFinder.Succeeded())
	{
		MeshComponent->SetStaticMesh(CubeFinder.Object);
	}
	else
	{
		UE_LOG(LogEmotionBridge, Warning,
			TEXT("AEmotionLampActor: could not find /Engine/BasicShapes/Cube — assign a mesh in the editor."));
	}

	// Point light — positioned above the mesh so it casts light on the scene.
	PointLightComponent = CreateDefaultSubobject<UPointLightComponent>(TEXT("PointLight"));
	PointLightComponent->SetupAttachment(RootSceneComponent);
	PointLightComponent->SetRelativeLocation(FVector(0.f, 0.f, 100.f));
	PointLightComponent->SetIntensity(LightIntensityBase);
	PointLightComponent->SetLightColor(FLinearColor::White);
	PointLightComponent->SetAttenuationRadius(600.f);
	PointLightComponent->SetCastShadows(true);
}

void AEmotionLampActor::BeginPlay()
{
	Super::BeginPlay();

	// Create a dynamic material instance so we can tint the mesh without
	// modifying any shared material asset.  If the material does not expose
	// a "BaseColor" vector parameter the SetVectorParameterValue call below
	// is a no-op, which is safe.
	if (MeshComponent && MeshComponent->GetMaterial(0))
	{
		DynamicMaterial = MeshComponent->CreateAndSetMaterialInstanceDynamic(0);
	}

	ResetToNeutral();
}

void AEmotionLampActor::ApplyEmotion(const FString& Emotion, float Confidence)
{
	const UEmotionBridgeSettings* Settings = UEmotionBridgeSettings::Get();
	const FLinearColor Color = Settings
		? Settings->GetColorForEmotion(Emotion)
		: FLinearColor::White;

	// Scale light intensity between LightIntensityMinFraction and 1.0 based on confidence.
	const float Scale = FMath::Lerp(
		LightIntensityMinFraction, 1.0f,
		FMath::Clamp(Confidence, 0.f, 1.f));
	const float Intensity = LightIntensityBase * Scale;

	// Drive the point light.
	if (PointLightComponent)
	{
		PointLightComponent->SetLightColor(Color);
		PointLightComponent->SetIntensity(Intensity);
	}

	// Attempt to tint the mesh by setting a "BaseColor" vector parameter.
	// This works automatically with any material that exposes that parameter
	// (e.g. a simple M_Color material with a BaseColor param).
	if (DynamicMaterial)
	{
		DynamicMaterial->SetVectorParameterValue(TEXT("BaseColor"), Color);
	}

	UE_LOG(LogEmotionBridge, Verbose,
		TEXT("Lamp [%s]: emotion=%s color=(%.2f,%.2f,%.2f) intensity=%.0f"),
		*GetName(), *Emotion, Color.R, Color.G, Color.B, Intensity);
}

void AEmotionLampActor::ResetToNeutral()
{
	ApplyEmotion(TEXT("neutral"), 1.0f);
}
