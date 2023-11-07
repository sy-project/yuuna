#include "Header.hlsli"

struct PixelInput
{
    float4 pos : SV_Position;
    float4 color : Color;
};

PixelInput VS(VertexColor input)
{
    PixelInput output;
    
    output.pos = mul(input.pos, world);
    output.pos = mul(output.pos, view);
    output.pos = mul(output.pos, projection);
    
    output.color = input.color;
    
    return output;
}

float4 PS(PixelInput input) : SV_Target
{
    return input.color;
}