#include "Header.hlsli"

struct PixelInput
{
    float4 pos : SV_Position;
    float2 uv : UV;
    float3 normal : Normal;
    float3 tangent : Tangent;
    float3 binormal : Binormal;
    float3 camPos : CamPos;
    float3 worldPos : Position;
};

PixelInput VS(VertexInstancing input)
{
    PixelInput output;
    
    matrix transform = 0;    
    
    [flatten]
    if (modelType == 2)
        transform = mul(SkinWorld(input.index, input.indices, input.weights), input.transform);
    else if (modelType == 1)
        transform = mul(BoneWorld(input.indices, input.weights), input.transform);
    else
        transform = input.transform;
    
    output.pos = mul(input.pos, transform);
    
    output.camPos = invView._41_42_43;
    output.worldPos = output.pos;
    
    output.pos = mul(output.pos, view);
    output.pos = mul(output.pos, projection);
    
    output.normal = mul(input.normal, (float3x3) transform);
    output.tangent = mul(input.tangent, (float3x3) transform);
    output.binormal = cross(output.normal, output.tangent);
    
    output.uv = input.uv;
    
    return output;
}

float4 PS(PixelInput input) : SV_Target
{
    float4 albedo = float4(1, 1, 1, 1);
    if (hasDiffuseMap)
        albedo = diffuseMap.Sample(samp, input.uv);
    
    Material material;
    material.normal = NormalMapping(input.tangent, input.binormal, input.normal, input.uv);
    material.diffuseColor = albedo;
    material.camPos = input.camPos;
    material.emissive = mEmissive;
    material.shininess = shininess;
    material.specularIntensity = SpecularMapping(input.uv);
    material.worldPos = input.worldPos;
    
    float4 ambient = CalcAmbient(material) * mAmbient;
    
    float4 result = 0;
    
    for (int i = 0; i < lightCount; i++)
    {
        if (!lights[i].active)
            continue;
        
        [flatten]
        if (lights[i].type == 0)
            result += CalcDirectional(material, lights[i]);
        else if (lights[i].type == 1)
            result += CalcPoint(material, lights[i]);
        else if (lights[i].type == 2)
            result += CalcSpot(material, lights[i]);
        else if (lights[i].type == 3)
            result += CalcCapsule(material, lights[i]);
    }
    
    float4 emissive = CalcEmissive(material);
    
    return result + ambient + emissive;
}