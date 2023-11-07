#include "Header.hlsli"

cbuffer RainBuffer : register(b10)
{
    float3 veloctiy;
    float drawDistance;
    
    float4 color;
    
    float3 origin;
    float time;
    
    float3 size;
}

struct VertexInput
{
    float3 pos : Position;
    float2 size : Size;
};

struct VertexOutput
{
    float3 pos : Position;
    float2 size : Size;
    float2 distance : Distance;
    float4 color : Color;
    float3 velocity : Velocity;
};

VertexOutput VS(VertexInput input)
{
    VertexOutput output;
    
    float3 v = veloctiy;
    v.xz /= input.size.y * 0.1f;
    output.velocity = v;
    
    float3 displace = time * v;
    input.pos = origin + (size + (input.pos + displace) % size) % size - (size * 0.5f);
    
    float alpha = cos(time + (input.pos.x + input.pos.z));
    alpha = saturate(1.5f + alpha / drawDistance * 2);
    
    output.distance.x = drawDistance;
    output.distance.y = alpha;
    
    output.color = color;
    
    output.pos = input.pos;
    output.size = input.size;
    
    return output;
}

struct PixelInput
{
    float4 pos : SV_Position;
    float2 uv : UV;
    float4 color : Color;
    float alpha : Alpha;
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
    
    float3 up = normalize(-input[0].velocity);
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
    pixelInput.color = input[0].color;
    
    [unroll]
    for (int i = 0; i < 4; i++)
    {
        pixelInput.pos = mul(vertices[i], view);
        pixelInput.pos = mul(pixelInput.pos, projection);
        
        pixelInput.uv = TEXCOORD[i];
        
        pixelInput.alpha = 0.2f * saturate(1 - pixelInput.pos.z / input[0].distance.x)
        * input[0].distance.y;
        
        output.Append(pixelInput);
    }

}

float4 PS(PixelInput input) : SV_Target
{
    float4 result = diffuseMap.Sample(samp, input.uv) * mDiffuse;
    
    result.rgb *= input.color.rgb * (1 + input.alpha) * 2.0f;
    result.a *= input.alpha;
        
    return result;
}