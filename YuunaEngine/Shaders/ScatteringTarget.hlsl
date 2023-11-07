#include "Header.hlsli"

struct PixelInput
{
    float4 pos : SV_Position;
    float2 uv : UV;
};
    
PixelInput VS(VertexUV input)
{
    PixelInput output;
    output.pos = input.pos;
    output.uv = input.uv;
    return output;
}

static const float PI = 3.141592f;
static const float InnerRadius = 6356.7523142f;
static const float OuterRadius = InnerRadius * 1.0157313f;

static const float KrESun = 0.0025f * 20.0f; //레일리 상수 * 태양 밝기
static const float KmESun = 0.001f * 20.0f; //미 상수 * 태양 밝기
static const float Kr4PI = 0.0025f * 4.0f * PI;
static const float Km4PI = 0.001f * 4.0f * PI;

static const float2 RayleighMieScaleHeight = { 0.25f, 0.1f };
static const float Scale = 1.0f / (OuterRadius - InnerRadius);

cbuffer TargetBuffer : register(b10)
{
    float3 waveLength;
    int sampleCount;
    
    float3 invWaveLength;
    float tagetPadding;
    
    float3 waveLengthMie;
}

struct PixelOutput
{
    float4 RColor : SV_Target0;
    float4 MColor : SV_Target1;
};

float HitOuterSphere(float3 position, float3 dir)
{
    float3 light = -position;
    
    float b = dot(light, dir);
    float c = dot(light, light);
    
    float d = c - b * b;
    float q = sqrt(OuterRadius * OuterRadius - d);
    
    return b + q;
}

float2 GetDensityRatio(float height)
{
    float altitude = (height - InnerRadius) * Scale;
    
    return exp(-altitude / RayleighMieScaleHeight);
}

float2 GetDistance(float3 p1, float3 p2)
{
    float2 opticalDepth = 0;
    
    float3 temp = p2 - p1;
    float far = length(temp);
    float3 dir = temp / far;
    
    float sampleLength = far / sampleCount;
    float scaleLength = sampleLength * Scale;
    
    float3 sampleRay = dir * sampleLength;
    p1 += sampleRay * 0.5f;
    
    for (int i = 0; i < sampleCount; i++)
    {
        float height = length(p1);
        opticalDepth += GetDensityRatio(height);
        
        p1 += sampleRay;
    }

    return opticalDepth * scaleLength;
}

PixelOutput PS(PixelInput input)
{
    PixelOutput output;
    
    float3 sunDirection = -normalize(lights[0].direction);
    float2 uv = input.uv;
    
    float3 pointPv = float3(0, InnerRadius + 1e-3f, 0.0f);
    float angleXZ = PI * uv.y;
    float angleY = 100.0f * uv.x * PI / 180.0f;
    
    float3 dir;
    dir.x = sin(angleY) * cos(angleXZ);
    dir.y = cos(angleY);
    dir.z = sin(angleY) * sin(angleXZ);
    dir = normalize(dir);
    
    float farPvPa = HitOuterSphere(pointPv, dir);
    float3 ray = dir;
    
    float3 pointP = pointPv;
    float sampleLength = farPvPa / sampleCount;
    float scaleLength = sampleLength * Scale;
    float3 sampleRay = ray * sampleLength;
    pointP += sampleRay * 0.5f;

    float3 rayleighSum = 0;
    float3 mieSum = 0;
    
    for (int i = 0; i < sampleCount; i++)
    {
        float pHeight = length(pointP);
        
        float2 densityRatio = GetDensityRatio(pHeight);
        densityRatio *= scaleLength;
        
        float2 viewerOpticalDepth = GetDistance(pointP, pointPv);
        
        float farPPc = HitOuterSphere(pointP, sunDirection);
        float2 sunOpticalDepth = GetDistance(pointP, pointP + sunDirection * farPPc);
        
        float2 opticalDepthP = sunOpticalDepth.xy + viewerOpticalDepth.xy;
        float3 attenunation = exp(-Kr4PI * invWaveLength * opticalDepthP.x - Km4PI * opticalDepthP.y);
        
        rayleighSum += densityRatio.x * attenunation;
        mieSum += densityRatio.y * attenunation;
        
        pointP += sampleRay;
    }

    float3 rayleigh = rayleighSum * KrESun;
    float3 mie = mieSum * KmESun;
    
    rayleigh *= invWaveLength;
    mie *= waveLength;
    
    output.RColor = float4(rayleigh, 1.0f);
    output.MColor = float4(mie, 1.0f);
    
    return output;
}