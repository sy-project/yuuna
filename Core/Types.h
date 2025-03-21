#pragma once
#include <iostream>
namespace Vector
{
	typedef struct Vector2D {
		float x;
		float y;
	};
	typedef struct Vector3D {
		float x;
		float y;
		float z;
	};
	typedef struct Vector4D {
		float x;
		float y;
		float z;
		float w;
	};

	__inline Vector2D Vector2D_init() {
		Vector2D vect2d;
		vect2d.x = 0;
		vect2d.y = 0;
		return vect2d;
	}
	__inline Vector2D Vector2D_init(Vector2D value) {
		Vector2D vect2d;
		vect2d.x = value.x;
		vect2d.y = value.y;
		return vect2d;
	}
	__inline Vector2D Vector2D_init(float x, float y)
	{
		Vector2D vect2d;
		vect2d.x = x;
		vect2d.y = y;
		return vect2d;
	}

	__inline Vector3D Vector3D_init() {
		Vector3D vect3d;
		vect3d.x = 0;
		vect3d.y = 0;
		vect3d.z = 0;
		return vect3d;
	}

	__inline Vector3D Vector3D_init(Vector3D value)
	{
		Vector3D vect3d;
		vect3d.x = value.x;
		vect3d.y = value.y;
		vect3d.z = value.z;
		return vect3d;
	}

	__inline Vector3D Vector3D_init(float x, float y, float z)
	{
		Vector3D vect3d;
		vect3d.x = x;
		vect3d.y = y;
		vect3d.z = z;
		return vect3d;
	}

	__inline Vector3D Vector3D_SetX(Vector3D& value, float x)
	{
		Vector3D vect3d;
		vect3d.x = x;
		vect3d.y = value.y;
		vect3d.z = value.z;
		return vect3d;
	}

	__inline Vector3D Vector3D_SetY(Vector3D& value, float y)
	{
		Vector3D vect3d;
		vect3d.x = value.x;
		vect3d.y = y;
		vect3d.z = value.z;
		return vect3d;
	}

	__inline Vector3D Vector3D_SetZ(Vector3D& value, float z)
	{
		Vector3D vect3d;
		vect3d.x = value.x;
		vect3d.y = value.y;
		vect3d.z = z;
		return vect3d;
	}

	__inline float Vector3D_GetX(Vector3D& value)
	{
		return value.x;
	}

	__inline float Vector3D_GetY(Vector3D& value)
	{
		return value.y;
	}

	__inline float Vector3D_GetZ(Vector3D& value)
	{
		return value.z;
	}

	__inline Vector4D Vector4D_init() {
		Vector4D vect4d;
		vect4d.x = 0;
		vect4d.y = 0;
		vect4d.z = 0;
		vect4d.w = 0;
		return vect4d;
	}
}

struct Circle2D {
	Vector::Vector2D center;
	float radius;
};
struct Line {
	float d[3];
	float determinant;
};
struct Vertex2D {
	Vector::Vector2D p;
	Vector::Vector2D uv;
};
struct Vertex3D {
	Vector::Vector3D p;
	Vector::Vector2D uv;
};
struct Triangle2D {
	unsigned int objId;
	Vertex2D v1, v2, v3;
};
struct Triangle3D {
	unsigned int objId;
	Vertex3D v1, v2, v3;
};
struct Image2D {
	unsigned int objId;
	uint8_t* img;
	Vector::Vector2D size;
};