#include "Header.hlsli"

struct PixelInput
{
    float4 pos : SV_Position;
    float2 uv : UV;
};

PixelInput VS(VertexUV input)
{
    PixelInput output;
    
    output.pos = mul(input.pos, world);
    output.pos = mul(output.pos, view);
    output.pos = mul(output.pos, projection);
    
    output.uv = input.uv;
    
    return output;
}

cbuffer Info : register(b10)
{
    int value;
    int range;
    float2 imageSize;
}

float4 PS(PixelInput input) : SV_Target
{
    float count = 0;
    
    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            float2 offset = (float2(x, y) / imageSize) * (range + 1);
            float4 result = diffuseMap.Sample(samp, input.uv + offset);
            
            count += result.a;
        }
    }
    
    if(count > value && count < 9 - value)
        return mDiffuse;
    
    return float4(0, 0, 0, 0);
}