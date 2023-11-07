#pragma once

class MatrixBuffer : public ConstBuffer
{
private:
	struct Data
	{
		Matrix matrix;
	}data;

public:
	MatrixBuffer() : ConstBuffer(&data, sizeof(Data))
	{
		data.matrix = XMMatrixIdentity();
	}

	void Set(Matrix value)
	{
		data.matrix = XMMatrixTranspose(value);
	}
};

#define MAX_LIGHT 10

struct Light
{
	enum Type
	{
		DIRECTIONAL,
		POINT,
		SPOT,
		CAPSULE
	};

	Float4 color;
	Float3 direction;
	Type type;

	Float3 position;
	float range;

	float inner;
	float outer;

	float length;
	int active;

	Light()
	{
		color = Float4(1, 1, 1, 1);

		direction = Float3(0, -1, 1);
		type = DIRECTIONAL;

		position = Float3(0, 0, 0);
		range = 80.0f;

		inner = 55.0f;
		outer = 65.0f;

		length = 50;
		active = 1;
	}
};

class LightBuffer : public ConstBuffer
{
public:
	struct Data
	{
		Light lights[MAX_LIGHT];
		UINT lightCount;
		float padding[3];

		Float4 ambient;
		Float4 ambientCeil;
	}data;

	LightBuffer() : ConstBuffer(&data, sizeof(Data))
	{
		data.lightCount = 0;
		data.ambient = { 0.1f, 0.1f, 0.1f, 1.0f };
		data.ambientCeil = { 0.1f, 0.1f, 0.1f, 1.0f };
	}

	void Add(Light light)
	{
		data.lights[data.lightCount++] = light;
	}

	void Add()
	{
		data.lightCount++;
	}
};

class RayBuffer : public ConstBuffer
{
public:
	struct Data
	{
		Float3 position;
		float size;

		Float3 direction;
		float padding;
	}data;

	RayBuffer() : ConstBuffer(&data, sizeof(Data))
	{
		data.position = Float3(0, 0, 0);
		data.size = 0.0f;
		data.direction = Float3(0, 0, 0);
	}
};

class BoneBuffer : public ConstBuffer
{
public:
	struct Data
	{
		Matrix bones[MAX_BONE];		
	}data;

	BoneBuffer() : ConstBuffer(&data, sizeof(Data))
	{
		for (UINT i = 0; i < MAX_BONE; i++)
			data.bones[i] = XMMatrixIdentity();		
	}

	void Add(Matrix matrix, UINT index)
	{
		data.bones[index] = XMMatrixTranspose(matrix);		
	}
};

class FrameBuffer : public ConstBuffer
{
public:
	struct KeyFrameDesc
	{
		int clip = 0;
		UINT curFrame = 0;
		UINT nextFrame = 0;
		float time = 0.0f;

		float runningTime = 0.0f;
		float speed = 1.0f;
		float padding[2];
	};

	struct TweenDesc
	{
		float takeTime;
		float tweenTime;
		float runningTime;
		float padding;

		KeyFrameDesc cur;
		KeyFrameDesc next;

		TweenDesc() : takeTime(0.5f), tweenTime(0.5f), runningTime(0.0f)
		{
			cur.clip = 0;
			next.clip = -1;
		}
	};

	struct Data
	{		
		TweenDesc tweenDesc[MAX_INSTANCE];
	}data;

	FrameBuffer() : ConstBuffer(&data, sizeof(Data))
	{
	}
};

class TypeBuffer : public ConstBuffer
{
public:
	struct Data
	{
		int values[4];

		Data() : values{}
		{}
	}data;

	TypeBuffer() : ConstBuffer(&data, sizeof(Data))
	{		
	}
};

class TerrainBuffer : public ConstBuffer
{
public:
	struct Data
	{
		Float2 distance = Float2(1, 1000);
		Float2 factor = Float2(1, 64);

		float cellSpacing = 5.0f;
		float cellSpacingU;
		float cellSpacingV;
		float heightScale = 20.0f;

		Float4 cullings[6];
	}data;

	TerrainBuffer() : ConstBuffer(&data, sizeof(Data))
	{
	}
};

class ColorBuffer : public ConstBuffer
{
public:
	struct Data
	{
		Float4 color = Float4(1, 1, 1, 1);
	}data;

	ColorBuffer() : ConstBuffer(&data, sizeof(Data))
	{
	}
};

class SizeBuffer : public ConstBuffer
{
public:
	struct Data
	{
		Float2 size = Float2(0, 0);

		float padding[2];
	}data;

	SizeBuffer() : ConstBuffer(&data, sizeof(Data))
	{
	}
};

class TimeBuffer : public ConstBuffer
{
public:
	struct Data
	{
		float time = 0.0f;

		float padding[3];
	}data;

	TimeBuffer() : ConstBuffer(&data, sizeof(Data))
	{
	}
};

class EffectBuffer : public ConstBuffer
{
public:
	struct Data
	{
		Float4 minColor;
		Float4 maxColor;

		Float3 gravity;
		float endVelocity;

		Float2 startSize;
		Float2 endSize;

		Float2 rotateSpeed;
		float readyTime;
		float readyRandomTime;

		float curTime;
		float padding[3];
	}data;

	EffectBuffer() : ConstBuffer(&data, sizeof(Data))
	{
	}
};