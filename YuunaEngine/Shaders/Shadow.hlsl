#include "Header.hlsli"

cbuffer LightView : register(b11)
{
    matrix lightView;
}

cbuffer LightProjection : register(b12)
{
    matrix lightProjection;
}

struct PixelInput
{
    float4 pos : SV_Position;
    float2 uv : UV;
    float3 normal : Normal;
    float3 tangent : Tangent;
    float3 binormal : Binormal;
    float3 camPos : CamPos;
    float3 worldPos : Position0;
    float4 clipPos : Position1;
};

PixelInput VS(VertexUVNormalTangentBlend input)
{
    PixelInput output;
    
    matrix transform = 0;
    
    [flatten]
    if (modelType == 2)
        transform = mul(SkinWorld(input.indices, input.weights), world);
    else if (modelType == 1)
        transform = mul(BoneWorld(input.indices, input.weights), world);
    else
        transform = world;
    
    output.pos = mul(input.pos, transform);
    
    output.camPos = invView._41_42_43;
    output.worldPos = output.pos;
    
    output.pos = mul(output.pos, view);
    output.pos = mul(output.pos, projection);
    
    output.clipPos = mul(input.pos, transform);
    output.clipPos = mul(output.clipPos, lightView);
    output.clipPos = mul(output.clipPos, lightProjection);
    
    output.normal = mul(input.normal, (float3x3) transform);
    output.tangent = mul(input.tangent, (float3x3) transform);
    output.binormal = cross(output.normal, output.tangent);
    
    output.uv = input.uv;
    
    return output;
}

Texture2D depthMap : register(t10);

cbuffer Quality : register(b10)
{
    int quality;
}

cbuffer SizeBuffer : register(b11)
{
    float2 mapSize;
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
    
    float4 color = result + ambient + emissive;
    
    float currentDepth = input.clipPos.z / input.clipPos.w;
    
    float2 uv = input.clipPos.xy / input.clipPos.w;
    uv.y = -uv.y;
    uv = uv * 0.5f + 0.5f;
    
    
    if(uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f)
        return color;
    
    if(currentDepth < 0.0f || currentDepth > 1.0f)
        return color;
    
    
    float shadowDepth = depthMap.Sample(samp, uv).r;
    
    float factor = 0;
    
    [flatten]
    if(quality == 0)
    {
        if (currentDepth > shadowDepth + 0.0001f)
            factor = 0.5f;
    }else
    {
        float sum = 0;        
        int count = 1;
        for (float y = -0.5f; y <= 0.5f; y += 0.2f)
        {
            for (float x = -0.5f; x <= 0.5f; x += 0.2f)
            {
                float2 offset = float2(x / mapSize.x, y / mapSize.y);
                shadowDepth = depthMap.Sample(samp, uv + offset).r;
                
                if(currentDepth > shadowDepth + 0.0001f)
                {                    
                    sum += shadowDepth;
                }
                count ++;
            }
        }

        factor = sum / count;
    }   
    
    factor = saturate(factor);
    
    if(factor < 1)
        factor = 1.0f - factor;
    
    return color * factor;
}