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
	__inline Vector2D operator+(const Vector2D& a, const Vector2D& b) {
		return Vector2D{ a.x + b.x, a.y + b.y };
	}

	__inline Vector2D operator-(const Vector2D& a, const Vector2D& b) {
		return Vector2D{ a.x - b.x, a.y - b.y };
	}

	__inline Vector2D operator*(const Vector2D& a, float scalar) {
		return Vector2D{ a.x * scalar, a.y * scalar };
	}

	__inline Vector2D operator/(const Vector2D& a, float scalar) {
		return Vector2D{ a.x / scalar, a.y / scalar };
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

	__inline Vector3D operator+(const Vector3D& a, const Vector3D& b) {
		return Vector3D{ a.x + b.x, a.y + b.y, a.z + b.z };
	}

	__inline Vector3D operator-(const Vector3D& a, const Vector3D& b) {
		return Vector3D{ a.x - b.x, a.y - b.y, a.z - b.z };
	}

	__inline Vector3D operator*(const Vector3D& a, float scalar) {
		return Vector3D{ a.x * scalar, a.y * scalar, a.z * scalar };
	}

	__inline Vector3D operator/(const Vector3D& a, float scalar) {
		return Vector3D{ a.x / scalar, a.y / scalar, a.z / scalar };
	}

	__inline Vector4D Vector4D_init() {
		Vector4D vect4d;
		vect4d.x = 0;
		vect4d.y = 0;
		vect4d.z = 0;
		vect4d.w = 0;
		return vect4d;
	}
	__inline Vector4D operator+(const Vector4D& a, const Vector4D& b) {
		return Vector4D{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
	}

	__inline Vector4D operator-(const Vector4D& a, const Vector4D& b) {
		return Vector4D{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
	}

	__inline Vector4D operator*(const Vector4D& a, float scalar) {
		return Vector4D{ a.x * scalar, a.y * scalar, a.z * scalar, a.w * scalar };
	}

	__inline Vector4D operator/(const Vector4D& a, float scalar) {
		return Vector4D{ a.x / scalar, a.y / scalar, a.z / scalar, a.w / scalar };
	}
}
enum eRTFlag
{
	NONE = 0,				// DO NOT RAY TRACING
	DIFFRACT = 1 << 0,		// DIFFRACT ON
	REFLECT = 1 << 1,		// REFLECT ON
	REVERB = 1 << 2			// REVERB ON
};
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
	unsigned int texId;
	Vector::Vector3D normal;
	Vertex3D v1, v2, v3;
};
struct Image2D {
	unsigned int objId;
	unsigned int texId;
	uint8_t* img;
	Vector::Vector2D size;
};
struct Mesh3D {
	Vertex3D vert;
	Vector::Vector3D normalv;
	eRTFlag GRTFlag;	//Graphic
	eRTFlag SRTFlag;	//Sound
};

enum LightType {
	L_POINT,
	L_SPOT,
	L_DIRECTIONAL
};
struct Light {
	Vector::Vector3D position;
	Vector::Vector3D direction;
	float intensity;
	Vector::Vector3D color;
	LightType type;
	float angle;
};

struct Matrix4x4 {
	float m[4][4];

	static Matrix4x4 identity() {
		Matrix4x4 mat = {};
		for (int i = 0; i < 4; ++i)
			mat.m[i][i] = 1.0f;
		return mat;
	}
};

struct Camera {
	Vector::Vector3D position;
	Vector::Vector3D forward;
	Vector::Vector3D up;
	float fov;
	float aspectRatio;
};