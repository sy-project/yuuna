#include "Header.hlsli"

cbuffer ReflectionBuffer : register(b10)
{
    matrix reflectionView;
}

struct PixelInput
{
    float4 pos : SV_Position;
    float4 reflectionPos : Position;
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
    
    output.reflectionPos = mul(output.pos, reflectionView);
    output.reflectionPos = mul(output.reflectionPos, projection);
    
    output.pos = mul(output.pos, view);
    output.pos = mul(output.pos, projection);
    
    return output;
}

Texture2D reflectionMap : register(t10);

float4 PS(PixelInput input) : SV_Target
{
    float4 refPos = input.reflectionPos;
    
    float2 reflection;
    reflection.x = refPos.x / refPos.w * 0.5f + 0.5f;
    reflection.y = -refPos.y / refPos.w * 0.5f + 0.5f;
    
    return reflectionMap.Sample(samp, reflection);
}