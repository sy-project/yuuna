#include "Header.hlsli"

cbuffer SparkBuffer : register(b10)
{
    float3 direction;
    float duration;
    float time;
}

struct VertexInput
{
    float4 pos : Position;
    float2 size : Size;
    float3 velocity : Velocity;
};

struct VertexOutput
{
    float3 pos : Position;
    float2 size : Size;
    float time : Time;
};

VertexOutput VS(VertexInput input)
{
    VertexOutput output;
    
    output.time = time / duration;
    
    input.velocity += direction * time;
    
    input.pos = mul(input.pos, world);    
    output.pos = input.pos.xyz + input.velocity * time;
    
    output.size = input.size;
    
    return output;
}

struct PixelInput
{
    float4 pos : SV_Position;
    float2 uv : UV;
    float time : Time;
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
    
    float3 up = invView._21_22_23;
    float3 forward = camPos - input[0].pos;    
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
    pixelInput.time = input[0].time;
    
    [unroll]
    for (int i = 0; i < 4; i++)
    {
        pixelInput.pos = mul(vertices[i], view);
        pixelInput.pos = mul(pixelInput.pos, projection);
        
        pixelInput.uv = TEXCOORD[i];
        
        output.Append(pixelInput);
    }

}

cbuffer StartColorBuffer : register(b10)
{
    float4 startColor;
}

cbuffer EndColorBuffer : register(b11)
{
    float4 endColor;
}

float4 PS(PixelInput input) : SV_Target
{
    float4 albedo = diffuseMap.Sample(samp, input.uv) * mDiffuse;
    
    float4 color = lerp(startColor, endColor, input.time);
    return albedo * color;
}