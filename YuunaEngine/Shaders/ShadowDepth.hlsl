#include "Header.hlsli"

struct PixelInput
{
    float4 pos : SV_Position;    
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
    
    return output;
}

float4 PS(PixelInput input) : SV_Target
{
    float depth = input.pos.z / input.pos.w;
    
    return float4(depth.xxx, 1);
}