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

struct VertexInput
{
    float4 pos : Position;
    float4 color : Color;
};

struct PixelInput
{
    float4 pos : SV_Position;
    float4 color : Color;
};

PixelInput VS(VertexInput input)
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