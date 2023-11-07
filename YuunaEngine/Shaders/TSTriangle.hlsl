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
    float edge[3] : SV_TessFactor;
    float inside : SV_InsideTessFactor;
};

cbuffer EdgeInfo : register(b10)
{
    int edge0;
    int edge1;
    int edge2;
    int inside;
}

#define NUM_CONTROL_POINTS 3

CHullOutput CHS(InputPatch<VertexOutput, NUM_CONTROL_POINTS> input)
{
    CHullOutput output;
    
    output.edge[0] = edge0;
    output.edge[1] = edge1;
    output.edge[2] = edge2;
    output.inside = inside;
    
    return output;
}

struct HullOutput
{
    float4 pos : Position;
};

[domain("tri")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
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

[domain("tri")]
DomainOutput DS(CHullOutput input, float3 uvw : SV_DomainLocation,
const OutputPatch<HullOutput, NUM_CONTROL_POINTS> patch)
{
    DomainOutput output;
    
    float4 position = patch[0].pos * uvw.x + patch[1].pos * uvw.y + patch[2].pos * uvw.z;
    output.pos = float4(position.xyz, 1.0f);
    
    return output;
}

float4 PS(DomainOutput input) : SV_Target
{
    return float4(0, 1, 0, 1);
}