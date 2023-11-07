cbuffer World : register(b0)
{
    matrix world;
}

cbuffer View : register(b1)
{
    matrix view;
}

cbuffer Projection : register(b2)
{
    matrix projection;
}

cbuffer Light : register(b3)
{
    float3 lightDirection;
}

struct VertexInput
{
    float4 pos : Position;
    float2 uv : UV;
    float3 normal : Normal;
};

struct PixelInput
{
    float4 pos : SV_Position;
    float2 uv : UV;
    float diffuse : Diffuse;
};

PixelInput VS(VertexInput input)
{
    PixelInput output;
    
    output.pos = mul(input.pos, world);
    output.pos = mul(output.pos, view);
    output.pos = mul(output.pos, projection);
    
    float3 light = normalize(lightDirection);
    float3 normal = normalize(mul(input.normal, (float3x3)world));
    
    output.diffuse = saturate(dot(normal, -light));
    
    output.uv = input.uv;
    
    return output;
}

Texture2D map : register(t0);
SamplerState samp : register(s0);

float4 PS(PixelInput input) : SV_Target
{
    float4 color = map.Sample(samp, input.uv);
    
    float4 ambient = color * 0.1f;
    
    return color * input.diffuse + ambient;
}