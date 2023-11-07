#include "Header.hlsli"

struct ParticleDesc
{
    float4 minColor;
    float4 maxColor;
    
    float3 gravity;
    float endVelocity;
    
    float2 startSize;
    float2 endSize;
    
    float2 rotateSpeed;
    float readyTime;
    float readyRandomTime;
    
    float curTime;
};

cbuffer EffectBuffer : register(b10)
{
    ParticleDesc particle;
}

struct VertexInput
{
    float4 pos : Position;
    float2 uv : UV;
    float3 velocity : Velocity;
    float4 random : Random;
    float time : Time;
};

struct PixelInput
{
    float4 pos : SV_Position;
    float4 color : Color;
    float2 uv : UV;
};

float4 ComputePosition(float3 position, float3 velocity, float age, float normalizedAge)
{
    float start = length(velocity);
    float end = start * particle.endVelocity;
    
    float integral = start * normalizedAge + (end - start) * normalizedAge / 2;
    
    position += normalize(velocity) * integral * particle.readyTime;
    position += particle.gravity * normalizedAge * age;
    
    float4 result = float4(position, 1.0f);
    result = mul(result, view);
    result = mul(result, projection);    
    
    return result;
}

float ComputeSize(float value, float normalizedAge)
{
    float start = lerp(particle.startSize.x, particle.startSize.y, value);
    float end = lerp(particle.endSize.x, particle.endSize.y, value);
    
    return lerp(start, end, normalizedAge);
}

float2x2 ComputeRotation(float value, float age)
{
    float speed = lerp(particle.rotateSpeed.x, particle.rotateSpeed.y, value);
    float radian = speed * age;
    
    float c = cos(radian);
    float s = sin(radian);
    
    return float2x2(c, -s, s, c);
}

float4 ComputeColor(float value, float normalizedAge)
{
    float4 color = lerp(particle.minColor, particle.maxColor, value);
    color.a *= normalizedAge * (1 - normalizedAge) * (1 - normalizedAge) * 6.7f;
    
    return color;
}

PixelInput VS(VertexInput input)
{
    PixelInput output;
    
    float age = particle.curTime - input.time;
    age *= input.random.x * particle.readyRandomTime + 1;
    
    float normalizedAge = saturate(age / particle.readyTime);
    
    output.pos = ComputePosition(input.pos.xyz, input.velocity, age, normalizedAge);
    
    float size = ComputeSize(input.random.y, normalizedAge);
    float2x2 rotation = ComputeRotation(input.random.z, age);    
    
    output.pos.xy += mul(input.uv, rotation) * size * 0.5f;
    
    output.uv = (input.uv + 1.0f) * 0.5f;
    output.color = ComputeColor(input.random.w, normalizedAge);
    
    return output;
}

float4 PS(PixelInput input) : SV_Target
{    
    return diffuseMap.Sample(samp, input.uv) * input.color;
}