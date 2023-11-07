#include "Header.hlsli"

cbuffer View : register(b3)
{
    matrix Pview;
    matrix PinvView;
}

struct PixelInput
{
    float4 pos : SV_Position;
    float2 uv : UV;
};

static const float2 arrBasePos[4] =
{
    float2(-1.0f, 1.0f),
    float2(1.0f, 1.0f),
    float2(-1.0f, -1.0f),
    float2(1.0f, -1.0f)
};

PixelInput VS(uint vertexID : SV_VertexID)
{
    PixelInput output;
    output.pos = float4(arrBasePos[vertexID].xy, 0.0f, 1.0f);
    output.uv = output.pos.xy;
    
    return output;
}

float ConvertDepthToLinear(float depth)
{
    float linearDepth = projection._43 / (depth - projection._33);
    
    return linearDepth;
}

struct SurfaceData
{
    float linearDepth;
    float3 color;
    float3 normal;
    float specInt;
    float specPow;
};

SurfaceData UnpackGBuffer(int2 location)
{
    SurfaceData output;
    
    int3 location3 = int3(location, 0);
    
    float depth = depthTexture.Load(location3).x;
    output.linearDepth = ConvertDepthToLinear(depth);
    
    float4 diffuse = diffuseTexture.Load(location3);
    
    output.color = diffuse.rgb;
    output.specInt = diffuse.w;
    
    output.normal = normalTexture.Load(location3).xyz;
    output.normal = normalize(output.normal * 2.0f - 1.0f);
    
    float specular = specularTexture.Load(location3).x;
    output.specPow = specPowerRange.x + specular * specPowerRange.y;
    
    return output;
}

float3 CalcWorldPos(float2 csPos, float linearDepth)
{
    float4 position;
    
    float2 temp;
    temp.x = 1 / projection._11;
    temp.y = 1 / projection._22;
    position.xy = csPos.xy * temp * linearDepth;
    position.z = linearDepth;
    position.w = 1.0f;
    
    return mul(position, invView).xyz;
}

float4 PS(PixelInput input) : SV_Target
{   
    SurfaceData data = UnpackGBuffer(input.pos.xy);
    
    Material material;
    material.normal = data.normal;
    material.diffuseColor = float4(data.color, 1.0f);
    material.camPos = PinvView._41_42_43;
    material.shininess = data.specPow;
    material.specularIntensity = data.specInt.xxxx;
    material.worldPos = CalcWorldPos(input.uv, data.linearDepth);
    material.emissive = float4(0, 0, 0, 0);
    
    //float4 ambient = CalcAmbient(material) * mAmbient;
    
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
    
    return result;
    //return result + ambient;
    
    //float4 emissive = CalcEmissive(material);    
    //return result + ambient + emissive;
}