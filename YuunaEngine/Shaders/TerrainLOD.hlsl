#include "Header.hlsli"

//VS///////////////////////////////////////////
struct VertexOutput
{
    float4 pos : Position;
    float2 uv : UV;
};

VertexOutput VS(VertexOutput input)
{
    VertexOutput output;
    output.pos = input.pos;
    output.uv = input.uv;
    
    return output;
}

//HS/////////////////////////////////////////////////
struct CHullOutput
{
    float edge[4] : SV_TessFactor;
    float inside[2] : SV_InsideTessFactor;
};

cbuffer TerrainBuffer : register(b10)
{
    float2 dist;
    float2 tessFactor;
    
    float cellSpacing;
    float cellSpacingU;
    float cellSpacingV;
    float heightScale;
    
    float4 culling[6];
}

float CalcTessFactor(float3 position)
{
    float d = distance(position, invView._41_42_43);
    float f = saturate((d - dist.y) / (dist.x - dist.y));
    
    return lerp(tessFactor.x, tessFactor.y, f);
}

bool OutFrustumPlane(float3 center, float3 extent, float4 plane)
{
    float3 n = abs(plane.xyz);
    float r = dot(extent, n);
    float s = dot(float4(center, 1), plane);
    
    return (s + r) < 0.0f;
}

bool OutFrustum(float3 center, float3 extent)
{
    [unroll(6)]
    for (int i = 0; i < 6; i++)
    {
        [flatten]
        if (OutFrustumPlane(center, extent, culling[i]))
            return true;
    }
    
    return false;
}

#define NUM_CONTROL_POINTS 4

CHullOutput
    CHS(
    InputPatch<VertexOutput, NUM_CONTROL_POINTS> input)
{
    float4 position[4];
    position[0] = mul(input[0].pos, world);
    position[1] = mul(input[1].pos, world);
    position[2] = mul(input[2].pos, world);
    position[3] = mul(input[3].pos, world);
    
    float minY = 0.0f;
    float maxY = heightScale;
    
    float3 minBox = float3(position[2].x, minY, position[2].z);
    float3 maxBox = float3(position[1].x, maxY, position[1].z);
    
    float3 boxCenter = (minBox + maxBox) * 0.5f;
    float3 boxExtent = abs(maxBox - minBox) * 0.5f;
    
    CHullOutput output;
    
    [flatten]
    if (OutFrustum(boxCenter, boxExtent))
    {
        output.edge[0] = -1;
        output.edge[1] = -1;
        output.edge[2] = -1;
        output.edge[3] = -1;
    
        output.inside[0] = -1;
        output.inside[1] = -1;
        
        return output;
    }
    
    float3 e0 = (position[0] + position[2]).xyz * 0.5f;
    float3 e1 = (position[0] + position[1]).xyz * 0.5f;
    float3 e2 = (position[1] + position[3]).xyz * 0.5f;
    float3 e3 = (position[2] + position[3]).xyz * 0.5f;
    
    float3 center = (e0 + e2) * 0.5f;
    
    output.edge[0] = CalcTessFactor(e0);
    output.edge[1] = CalcTessFactor(e1);
    output.edge[2] = CalcTessFactor(e2);
    output.edge[3] = CalcTessFactor(e3);
    
    output.inside[0] = CalcTessFactor(center);
    output.inside[1] = CalcTessFactor(center);
    
    return output;
}

struct HullOutput
{
    float4 pos : Position;
    float2 uv : UV;
};

[domain("quad")]
//[partitioning("integer")]
[partitioning("fractional_even")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(4)]
[patchconstantfunc("CHS")]
HullOutput HS(InputPatch<VertexOutput, NUM_CONTROL_POINTS> input,
uint i : SV_OutputControlPointID)
{
    HullOutput output;
    output.pos = input[i].pos;
    output.uv = input[i].uv;
    
    return output;
}

struct DomainOutput
{
    float4 pos : SV_Position;
    float2 uv : UV;
};

Texture2D heightMap : register(t0);

[domain("quad")]
DomainOutput DS(CHullOutput input, float2 uv : SV_DomainLocation,
const OutputPatch<HullOutput, NUM_CONTROL_POINTS> patch)
{
    DomainOutput output;
    
    float4 v1 = lerp(patch[0].pos, patch[1].pos, uv.x);
    float4 v2 = lerp(patch[2].pos, patch[3].pos, uv.x);
    float4 position = lerp(v1, v2, uv.y);
    
    float2 uv1 = lerp(patch[0].uv, patch[1].uv, uv.x);
    float2 uv2 = lerp(patch[2].uv, patch[3].uv, uv.x);
    float2 texCoord = lerp(uv1, uv2, uv.y);
    
    position.y = heightMap.SampleLevel(samp, texCoord, 0).r * heightScale;
    
    output.pos = float4(position.xyz, 1.0f);
    output.pos = mul(output.pos, world);
    output.pos = mul(output.pos, view);
    output.pos = mul(output.pos, projection);
    
    output.uv = texCoord;
    
    return output;
}

float4 PS(DomainOutput input) : SV_Target
{
    return diffuseMap.Sample(samp, input.uv);
}