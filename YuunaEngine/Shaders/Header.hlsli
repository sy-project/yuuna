#define MAX_BONE 256
#define MAX_LIGHT 10
#define MAX_INSTANCE 400

//VertexShaderBuffer
cbuffer World : register(b0)
{
    matrix world;    
}

cbuffer View : register(b1)
{
    matrix view;
    matrix invView;
}

cbuffer Projection : register(b2)
{
    matrix projection;
}

cbuffer Bone : register(b3)
{
    matrix bones[MAX_BONE];        
}

struct KeyFrame
{
    int clip;
    uint curFrame;
    uint nextFrame;
    float time;
    
    float runninTime;    
    float speed;
    float2 padding;
};

struct TweenFrame
{
    float takeTime;
    float tweenTime;
    float runningTime;
    float padding;
    
    KeyFrame cur;
    KeyFrame next;
};

cbuffer Frame : register(b4)
{
    TweenFrame tweenFrame[MAX_INSTANCE];
}

cbuffer ModelType : register(b5)
{
    int modelType;
}

Texture2DArray transformMap : register(t0);

//PixelShaderBuffer
static const float2 specPowerRange = { 0.1f, 250.f };

struct Light
{
    float4 color;
    
    float3 direction;
    int type;

    float3 position;
    float range;
    
    float inner;
    float outer;
    
    float length;
    int active;
};

cbuffer Light : register(b0)
{
    Light lights[MAX_LIGHT];
    uint lightCount;
    
    float3 padding;
    
    float4 ambient;
    float4 ambientCeil;
}

struct Material
{
    float3 normal;    
    float4 diffuseColor;
    float4 emissive;    
    float4 specularIntensity;
    float shininess;
    float3 camPos;    
    float3 worldPos;    
};

cbuffer MaterialInfo : register(b1)
{
    float4 mDiffuse;        
    float4 mSpecular;        
    float4 mAmbient;
    float4 mEmissive;
    
    float shininess;
    
    int hasDiffuseMap;
    int hasSpecularMap;
    int hasNormalMap;
}

SamplerState samp : register(s0);

Texture2D diffuseMap : register(t0);
Texture2D specularMap : register(t1);
Texture2D normalMap : register(t2);

Texture2D depthTexture : register(t3);
Texture2D diffuseTexture : register(t4);
Texture2D specularTexture : register(t5);
Texture2D normalTexture : register(t6);

//VertexLayouts
struct Vertex
{
    float4 pos : Position;    
};

struct VertexColor
{
    float4 pos : Position;
    float4 color : Color;
};

struct VertexUV
{
    float4 pos : Position;
    float2 uv : UV;    
};

struct VertexUVNormal
{
    float4 pos : Position;
    float2 uv : UV;
    float3 normal : Normal;
};

struct VertexUVNormalTangent
{
    float4 pos : Position;
    float2 uv : UV;
    float3 normal : Normal;
    float3 tangent : Tangent;
};

struct VertexUVNormalTangentAlpha
{
    float4 pos : Position;
    float2 uv : UV;
    float3 normal : Normal;
    float3 tangent : Tangent;
    float4 alpha : Alpha;
};

struct VertexUVNormalTangentBlend
{
    float4 pos : Position;
    float2 uv : UV;
    float3 normal : Normal;
    float3 tangent : Tangent;
    float4 indices : BlendIndices;
    float4 weights : BlendWeights;
};

struct VertexInstancing
{
    float4 pos : Position;
    float2 uv : UV;
    float3 normal : Normal;
    float3 tangent : Tangent;
    float4 indices : BlendIndices;
    float4 weights : BlendWeights;
    
    matrix transform : Instance_Transform;
    uint index : Instance_Index;
    uint instanceID : SV_InstanceID;
};

//VertexShader////////////////////////////////////////////////////////////////
matrix BoneWorld(float4 indices, float4 weights)
{
    matrix transform = 0;
    
    transform += mul(weights.x, bones[(uint) indices.x]);
    transform += mul(weights.y, bones[(uint) indices.y]);
    transform += mul(weights.z, bones[(uint) indices.z]);
    transform += mul(weights.w, bones[(uint) indices.w]);
    
    return transform;
}

matrix SkinWorld(float4 indices, float4 weights)
{
    float4 c0, c1, c2, c3;
    float4 n0, n1, n2, n3;
    
    matrix transform = 0;
    matrix cur = 0, next = 0;
    matrix curAnim = 0, nextAnim = 0;
    
    int clip[2];
    int curFrame[2];
    int nextFrame[2];
    float time[2];
    
    clip[0] = tweenFrame[0].cur.clip;
    curFrame[0] = tweenFrame[0].cur.curFrame;
    nextFrame[0] = tweenFrame[0].cur.nextFrame;
    time[0] = tweenFrame[0].cur.time;
    
    clip[1] = tweenFrame[0].next.clip;
    curFrame[1] = tweenFrame[0].next.curFrame;
    nextFrame[1] = tweenFrame[0].next.nextFrame;
    time[1] = tweenFrame[0].next.time;
    
    [unroll(4)]
    for (int i = 0; i < 4; i++)
    {
        c0 = transformMap.Load(int4(indices[i] * 4 + 0, curFrame[0], clip[0], 0));
        c1 = transformMap.Load(int4(indices[i] * 4 + 1, curFrame[0], clip[0], 0));
        c2 = transformMap.Load(int4(indices[i] * 4 + 2, curFrame[0], clip[0], 0));
        c3 = transformMap.Load(int4(indices[i] * 4 + 3, curFrame[0], clip[0], 0));        
        cur = matrix(c0, c1, c2, c3);
        
        n0 = transformMap.Load(int4(indices[i] * 4 + 0, nextFrame[0], clip[0], 0));
        n1 = transformMap.Load(int4(indices[i] * 4 + 1, nextFrame[0], clip[0], 0));
        n2 = transformMap.Load(int4(indices[i] * 4 + 2, nextFrame[0], clip[0], 0));
        n3 = transformMap.Load(int4(indices[i] * 4 + 3, nextFrame[0], clip[0], 0));
        next = matrix(n0, n1, n2, n3);
        
        curAnim = lerp(cur, next, time[0]);
        
        [flatten]
        if(clip[1] > -1)
        {
            c0 = transformMap.Load(int4(indices[i] * 4 + 0, curFrame[1], clip[1], 0));
            c1 = transformMap.Load(int4(indices[i] * 4 + 1, curFrame[1], clip[1], 0));
            c2 = transformMap.Load(int4(indices[i] * 4 + 2, curFrame[1], clip[1], 0));
            c3 = transformMap.Load(int4(indices[i] * 4 + 3, curFrame[1], clip[1], 0));
            cur = matrix(c0, c1, c2, c3);
        
            n0 = transformMap.Load(int4(indices[i] * 4 + 0, nextFrame[1], clip[1], 0));
            n1 = transformMap.Load(int4(indices[i] * 4 + 1, nextFrame[1], clip[1], 0));
            n2 = transformMap.Load(int4(indices[i] * 4 + 2, nextFrame[1], clip[1], 0));
            n3 = transformMap.Load(int4(indices[i] * 4 + 3, nextFrame[1], clip[1], 0));
            next = matrix(n0, n1, n2, n3);
        
            nextAnim = lerp(cur, next, time[1]);
            
            curAnim = lerp(curAnim, nextAnim, tweenFrame[0].tweenTime);
        }
        
        transform += mul(weights[i], curAnim);
    }

    return transform;
}

matrix SkinWorld(int instanceID, float4 indices, float4 weights)
{
    float4 c0, c1, c2, c3;
    float4 n0, n1, n2, n3;
    
    matrix transform = 0;
    matrix cur = 0, next = 0;
    matrix curAnim = 0, nextAnim = 0;
    
    int clip[2];
    int curFrame[2];
    int nextFrame[2];
    float time[2];
    
    clip[0] = tweenFrame[instanceID].cur.clip;
    curFrame[0] = tweenFrame[instanceID].cur.curFrame;
    nextFrame[0] = tweenFrame[instanceID].cur.nextFrame;
    time[0] = tweenFrame[instanceID].cur.time;
    
    clip[1] = tweenFrame[instanceID].next.clip;
    curFrame[1] = tweenFrame[instanceID].next.curFrame;
    nextFrame[1] = tweenFrame[instanceID].next.nextFrame;
    time[1] = tweenFrame[instanceID].next.time;
    
    [unroll(4)]
    for (int i = 0; i < 4; i++)
    {
        c0 = transformMap.Load(int4(indices[i] * 4 + 0, curFrame[0], clip[0], 0));
        c1 = transformMap.Load(int4(indices[i] * 4 + 1, curFrame[0], clip[0], 0));
        c2 = transformMap.Load(int4(indices[i] * 4 + 2, curFrame[0], clip[0], 0));
        c3 = transformMap.Load(int4(indices[i] * 4 + 3, curFrame[0], clip[0], 0));
        cur = matrix(c0, c1, c2, c3);
        
        n0 = transformMap.Load(int4(indices[i] * 4 + 0, nextFrame[0], clip[0], 0));
        n1 = transformMap.Load(int4(indices[i] * 4 + 1, nextFrame[0], clip[0], 0));
        n2 = transformMap.Load(int4(indices[i] * 4 + 2, nextFrame[0], clip[0], 0));
        n3 = transformMap.Load(int4(indices[i] * 4 + 3, nextFrame[0], clip[0], 0));
        next = matrix(n0, n1, n2, n3);
        
        curAnim = lerp(cur, next, time[0]);
        
        [flatten]
        if (clip[1] > -1)
        {
            c0 = transformMap.Load(int4(indices[i] * 4 + 0, curFrame[1], clip[1], 0));
            c1 = transformMap.Load(int4(indices[i] * 4 + 1, curFrame[1], clip[1], 0));
            c2 = transformMap.Load(int4(indices[i] * 4 + 2, curFrame[1], clip[1], 0));
            c3 = transformMap.Load(int4(indices[i] * 4 + 3, curFrame[1], clip[1], 0));
            cur = matrix(c0, c1, c2, c3);
        
            n0 = transformMap.Load(int4(indices[i] * 4 + 0, nextFrame[1], clip[1], 0));
            n1 = transformMap.Load(int4(indices[i] * 4 + 1, nextFrame[1], clip[1], 0));
            n2 = transformMap.Load(int4(indices[i] * 4 + 2, nextFrame[1], clip[1], 0));
            n3 = transformMap.Load(int4(indices[i] * 4 + 3, nextFrame[1], clip[1], 0));
            next = matrix(n0, n1, n2, n3);
        
            nextAnim = lerp(cur, next, time[1]);
            
            curAnim = lerp(curAnim, nextAnim, tweenFrame[instanceID].tweenTime);
        }
        
        transform += mul(weights[i], curAnim);
    }

    return transform;
}

//PixelShader//////////////////////////////////////////////
float3 NormalMapping(float3 T, float3 B, float3 N, float2 uv)
{
    T = normalize(T);
    B = normalize(B);
    N = normalize(N);
    
    if(hasNormalMap)
    {
        float3 normal = normalMap.Sample(samp, uv).rgb;
                
        float3x3 TBN = float3x3(T, B, N);
        N = normal * 2.0f - 1.0f;
        N = normalize(mul(N, TBN));
    }
    
    return N;
}

float4 SpecularMapping(float2 uv)
{
    [flatten]
    if(hasSpecularMap)
        return specularMap.Sample(samp, uv);
    
    return float4(1, 1, 1, 1);
}

float4 CalcAmbient(Material material)
{
    float up = material.normal.y * 0.5f + 0.5f;
    
    float4 resultAmbient = ambient + up * ambientCeil;
    
    return resultAmbient * material.diffuseColor;
}

float4 CalcDirectional(Material material, Light light)
{
    float3 toLight = -normalize(light.direction);
    
    float NDotL = dot(toLight, material.normal);
    float4 finalColor = light.color * saturate(NDotL);
        
    float3 toEye = normalize(material.camPos - material.worldPos);
    float3 halfWay = normalize(toEye + toLight);
    float NDotH = saturate(dot(material.normal, halfWay));
    
    finalColor += light.color * pow(NDotH, material.shininess) * material.specularIntensity;
    
    return finalColor * material.diffuseColor;
}

float4 CalcPoint(Material material, Light light)
{
    float3 toLight = light.position - material.worldPos;
    float distanceToLight = length(toLight);
    toLight /= distanceToLight;
    
    float NDotL = dot(toLight, material.normal);
    float4 finalColor = light.color * saturate(NDotL);
    
    /*
    float3 toEye = normalize(material.camPos - material.worldPos);
    float3 halfWay = normalize(toEye + toLight);
    float NDotH = saturate(dot(material.normal, halfWay));
    
    finalColor += light.color * pow(NDotH, material.shininess) * material.specularIntensity;
    */    
    float distanceToLightNormal = 1.0f - saturate(distanceToLight / light.range);
    float attention = distanceToLightNormal * distanceToLightNormal;

    return finalColor * material.diffuseColor;
    return finalColor * material.diffuseColor * attention;
}

float4 CalcSpot(Material material, Light light)
{
    float3 toLight = light.position - material.worldPos;
    float distanceToLight = length(toLight);
    toLight /= distanceToLight;
    
    float NDotL = dot(toLight, material.normal);
    float4 finalColor = light.color * saturate(NDotL);
        
    float3 toEye = normalize(material.camPos - material.worldPos);
    float3 halfWay = normalize(toEye + toLight);
    float NDotH = saturate(dot(material.normal, halfWay));
    
    finalColor += light.color * pow(NDotH, material.shininess) * material.specularIntensity;
    
    float3 dir = -normalize(light.direction);
    float cosAngle = dot(dir, toLight);
    
    float outer = cos(radians(light.outer));
    float inner = 1.0f / cos(radians(light.inner));
    
    float conAttention = saturate((cosAngle - outer) * inner);
    conAttention *= conAttention;
    
    float distanceToLightNormal = 1.0f - saturate(distanceToLight / light.range);
    float attention = distanceToLightNormal * distanceToLightNormal;
    
    return finalColor * material.diffuseColor * attention * conAttention;
}

float4 CalcCapsule(Material material, Light light)
{
    float3 direction = normalize(light.direction);
    float3 start = material.worldPos - light.position;
    float distanceOnLine = dot(start, direction) / light.length;
    distanceOnLine = saturate(distanceOnLine) * light.length;
    
    float3 pointOnLine = light.position + direction * distanceOnLine;
    float3 toLight = pointOnLine - material.worldPos;
    float distanceToLight = length(toLight);
    toLight /= distanceToLight;
    
    float NDotL = dot(toLight, material.normal);
    float4 finalColor = light.color * saturate(NDotL);
        
    float3 toEye = normalize(material.camPos - material.worldPos);
    float3 halfWay = normalize(toEye + toLight);
    float NDotH = saturate(dot(material.normal, halfWay));
    
    finalColor += light.color * pow(NDotH, material.shininess) * material.specularIntensity;
    
    float distanceToLightNormal = 1.0f - saturate(distanceToLight / light.range);
    float attention = distanceToLightNormal * distanceToLightNormal;
    
    return finalColor * material.diffuseColor * attention;
}

float4 CalcEmissive(Material material)
{
    float emissive = 0.0f;
    
    [flatten]
    if (material.emissive.a > 0.0f)
    {
        float3 E = normalize(material.camPos - material.worldPos);
        
        float NdotE = dot(material.normal, E);
        emissive = smoothstep(1.0f - mEmissive.a, 1.0f, 1.0f - saturate(NdotE));
    }
    
    return float4(material.emissive.rgb * emissive, 1.0f);
}