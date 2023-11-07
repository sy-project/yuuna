#include "Header.hlsli"

struct VertexInput
{
    float4 pos : Position;
    float2 uv : UV;
    float3 normal : Normal;
    float3 tangent : Tangent;
    float4 indices : BlendIndices;
    float4 weights : BlendWeights;
    
    matrix transform : Instance_World;
    int skinNum : Instance_SkinNum;
    uint instanceID : SV_InstanceID;
};

struct PixelInput
{
    float4 pos : SV_Position;
    float2 uv : UV;
    float3 normal : Normal;
    float3 tangent : Tangent;
    float3 binormal : Binormal;
    float3 camPos : CamPos;
    float3 worldPos : Position;
    int skinNum : SkinNum;
};

PixelInput VS(VertexInput input)
{
    PixelInput output;
    
    matrix transform = world;
    
    [flatten]
    if (modelType == 2)
        transform = mul(SkinWorld(input.indices, input.weights), input.transform);
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
    
    output.skinNum = input.skinNum;
    
    return output;
}

Texture2D multiMap[4] : register(t10);

float4 PS(PixelInput input) : SV_Target
{
    float4 albedo = float4(1, 1, 1, 1);
    
    [flatten]
    if(input.skinNum == 0)
        albedo = multiMap[0].Sample(samp, input.uv);
    if (input.skinNum == 1)
        albedo = multiMap[1].Sample(samp, input.uv);
    if (input.skinNum == 2)
        albedo = multiMap[2].Sample(samp, input.uv);
    if (input.skinNum == 3)
        albedo = multiMap[3].Sample(samp, input.uv);
    
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