#include "Header.hlsli"

struct VertexOutput
{
    float3 pos : Position;
    float2 size : Size;
};

VertexOutput VS(VertexOutput input)
{
    VertexOutput output;
    output.pos = input.pos;
    output.size = input.size;
    
    return output;
}

struct PixelInput
{
    float4 pos : SV_Position;
    float2 uv : UV;
};

static const float2 TEXCOORD[4] =
{
    float2(0.0f, 1.0f),
    float2(0.0f, 0.0f),
    float2(1.0f, 1.0f),
    float2(1.0f, 0.0f)
};

[maxvertexcount(4)]
void GS(point VertexOutput input[1], inout TriangleStream<PixelInput> output)
{
    float3 camPos = invView._41_42_43;
    
    float3 up = float3(0, 1, 0);
    float3 forward = camPos - input[0].pos;
    forward.y = 0.0f;
    forward = normalize(forward);
    
    float3 right = normalize(cross(up, forward));
    
    float halfWidth = input[0].size.x * 0.5f;
    float halfHeight = input[0].size.y * 0.5f;
    
    float4 vertices[4];
    vertices[0] = float4(input[0].pos + halfWidth * right - halfHeight * up, 1.0f);
    vertices[1] = float4(input[0].pos + halfWidth * right + halfHeight * up, 1.0f);
    vertices[2] = float4(input[0].pos - halfWidth * right - halfHeight * up, 1.0f);
    vertices[3] = float4(input[0].pos - halfWidth * right + halfHeight * up, 1.0f);
    
    PixelInput pixelInput;
    
    [unroll]
    for (int i = 0; i < 4; i++)
    {
        pixelInput.pos = mul(vertices[i], view);
        pixelInput.pos = mul(pixelInput.pos, projection);
        
        pixelInput.uv = TEXCOORD[i];
        
        output.Append(pixelInput);
    }

}

float4 PS(PixelInput input) : SV_Target
{
    return diffuseMap.Sample(samp, input.uv) * mDiffuse;
}