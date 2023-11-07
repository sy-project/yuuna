#include "Header.hlsli"

struct VertexOutput
{
    float4 pos : Position;
};

VertexOutput VS(Vertex input)
{
    VertexOutput output;
    output.pos = input.pos;
    
    return output;
}

struct CHullOutput
{
    float edge[4] : SV_TessFactor;
    float inside[2] : SV_InsideTessFactor;
};

cbuffer EdgeInfo : register(b10)
{
    int edge0;
    int edge1;
    int edge2;
    int edge3;
}

cbuffer InsideInfo : register(b11)
{
    int inside0;
    int inside1;
}

#define NUM_CONTROL_POINTS 4

CHullOutput CHS(InputPatch<VertexOutput, NUM_CONTROL_POINTS> input)
{
    CHullOutput output;
    
    output.edge[0] = edge0;
    output.edge[1] = edge1;
    output.edge[2] = edge2;
    output.edge[3] = edge3;
    
    output.inside[0] = inside0;
    output.inside[1] = inside1;
    
    return output;
}

struct HullOutput
{
    float4 pos : Position;
};

[domain("quad")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(4)]
[patchconstantfunc("CHS")]
HullOutput HS(InputPatch<VertexOutput, NUM_CONTROL_POINTS> input,
uint i : SV_OutputControlPointID)
{
    HullOutput output;
    output.pos = input[i].pos;
    
    return output;
}

struct DomainOutput
{
    float4 pos : SV_Position;
};

[domain("quad")]
DomainOutput DS(CHullOutput input, float2 uv : SV_DomainLocation,
const OutputPatch<HullOutput, NUM_CONTROL_POINTS> patch)
{
    DomainOutput output;
    
    float4 v1 = lerp(patch[0].pos, patch[2].pos, uv.x);
    float4 v2 = lerp(patch[1].pos, patch[3].pos, uv.x);
    float4 position = lerp(v1, v2, 1 - uv.y);
    
    output.pos = float4(position.xyz, 1.0f);
    
    return output;
}

float4 PS(DomainOutput input) : SV_Target
{
    return float4(0, 1, 0, 1);
}