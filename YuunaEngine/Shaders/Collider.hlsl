#include "Header.hlsli"

struct PixelInput
{
    float4 pos : SV_Position;    
};

PixelInput VS(Vertex input)
{
    PixelInput output;
    
    output.pos = mul(input.pos, world);
    output.pos = mul(output.pos, view);
    output.pos = mul(output.pos, projection);    
    
    return output;
}

float4 PS(PixelInput input) : SV_Target
{
    return mDiffuse;
}