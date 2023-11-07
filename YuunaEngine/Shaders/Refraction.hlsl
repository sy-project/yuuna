#include "Header.hlsli"

struct PixelInput
{
    float4 pos : SV_Position;
    float2 uv : UV;    
    float4 wvpPos : Position;
};

PixelInput VS(VertexUVNormalTangentBlend input)
{
    PixelInput output;
    
    matrix transform = 0;
    
    [flatten]
    if (modelType == 2)
        transform = mul(SkinWorld(input.indices, input.weights), world);
    else if (modelType == 1)
        transform = mul(BoneWorld(input.indices, input.weights), world);
    else
        transform = world;
    
    output.pos = mul(input.pos, transform);     
    output.pos = mul(output.pos, view);
    output.pos = mul(output.pos, projection);
    
    output.uv = input.uv;
    output.wvpPos = output.pos;
    
    return output;
}

Texture2D refractionMap : register(t11);

cbuffer Time : register(b11)
{
    float time;
}

float4 PS(PixelInput input) : SV_Target
{
    float4 refPos = input.wvpPos;
    
    float2 refraction;
    refraction.x = refPos.x / refPos.w * 0.5f + 0.5f;
    refraction.y = -refPos.y / refPos.w * 0.5f + 0.5f;
    
    input.uv.x += (time * 0.25f);
    input.uv.y += (time * 0.25f);
    
    float4 normal = normalMap.Sample(samp, input.uv) * 2.0f - 1.0f;
    refraction += normal.xy * 0.02f;
    
    return refractionMap.Sample(samp, refraction);
}