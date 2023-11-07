#include "Header.hlsli"

cbuffer ReflectionBuffer : register(b10)
{
    matrix reflectionView;
}

struct PixelInput
{
    float4 pos : SV_Position;
    float2 uv : UV;        
    float4 worldPos : Position0;
    float3 camPos : Position1;    
    float4 reflectionPos : Position2;
    float4 refractionPos : Position3;
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
    
    output.worldPos = output.pos;
    output.camPos = invView._41_42_43;
    
    output.reflectionPos = mul(output.pos, reflectionView);
    output.reflectionPos = mul(output.reflectionPos, projection);
    
    output.pos = mul(output.pos, view);
    output.pos = mul(output.pos, projection);
    
    output.refractionPos = output.pos;
    
    output.uv = input.uv;
    
    return output;
}

Texture2D reflectionMap : register(t10);
Texture2D refractionMap : register(t11);

cbuffer WaterBuffer : register(b10)
{
    float4 waterColor;
    
    float waveTranslation;
    float waveScale;
    float waterShininess;
    float waterAlpha;
    
    float waveSpeed;
}

float4 PS(PixelInput input) : SV_Target
{
    input.uv += waveTranslation;
    
    float3 normal = normalMap.Sample(samp, input.uv).rgb * 2.0f - 1.0f;
    
    float4 refPos = input.reflectionPos;
    
    float2 reflection;
    reflection.x = refPos.x / refPos.w * 0.5f + 0.5f;
    reflection.y = -refPos.y / refPos.w * 0.5f + 0.5f;
    reflection += normal.xy * waveScale;
    float4 reflectionColor = reflectionMap.Sample(samp, reflection);
    
    refPos = input.refractionPos;
    float2 refraction;
    refraction.x = refPos.x / refPos.w * 0.5f + 0.5f;
    refraction.y = -refPos.y / refPos.w * 0.5f + 0.5f;
    refraction += normal.xy * waveScale;
    float4 refractionColor = refractionMap.Sample(samp, refraction);
    
    float3 viewDir = normalize(input.worldPos.xyz - input.camPos);
    
    float3 heightView = -viewDir.yyy;
    
    float r = (1.2f - 1.0f) / (1.2f / 1.0f);
    float fresnel = saturate(min(1, r + (1 - r) * pow(1 - dot(normal, heightView), 2)));
    float4 diffuse = lerp(refractionColor, reflectionColor, fresnel) * waterColor;
    
    float3 light = lights[0].direction;
    light.y *= -1.0f;
    light.z *= -1.0f;
    
    float3 halfWay = normalize(viewDir + light);
    float specularIntensity = saturate(dot(halfWay, normal));
    
    [flatten]
    if(specularIntensity > 0.0f)
    {
        specularIntensity = pow(specularIntensity, waterShininess);
        diffuse = saturate(diffuse + specularIntensity);
    }
    
    return float4(diffuse.rgb, waterAlpha);
}