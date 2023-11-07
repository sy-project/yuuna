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
    
    output.normal = mul(input.normal, (float3x3) transform);
    output.tangent = mul(input.tangent, (float3x3) transform);
    output.binormal = cross(output.normal, output.tangent);
    
    output.uv = input.uv;
    
    return output;
}

struct PixelOutput
{
    float4 diffuse : SV_Target0;
    float4 specular : SV_Target1;
    float4 normal : SV_Target2;
};

PixelOutput PackGBuffer(float3 baseColor, float3 normal, float specIntensity)
{
    PixelOutput output;
    
    float specPowNorm = (mSpecular.a - specPowerRange.x) / specPowerRange.y;
    
    output.diffuse = float4(baseColor, specIntensity);
    output.specular = float4(specPowNorm, 0, 0, 0);
    output.normal = float4(normal * 0.5f + 0.5f, 0);
    
    return output;
}

PixelOutput PS(PixelInput input) : SV_Target
{
    float3 albedo = float3(1, 1, 1);
    if(hasDiffuseMap)
        albedo = diffuseMap.Sample(samp, input.uv).rgb;
    albedo *= albedo;
    
    float specIntensity = 1.0f;
    
    if(hasSpecularMap)
        specIntensity = specularMap.Sample(samp, input.uv).r;
    
    float3 normal = NormalMapping(input.tangent, input.binormal, input.normal, input.uv);
    
    return PackGBuffer(albedo, normal, specIntensity);
}