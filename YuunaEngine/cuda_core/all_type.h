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

	__inline Vector3D Vector3D_init() {
		Vector3D vect3d;
		vect3d.x = 0;
		vect3d.y = 0;
		vect3d.z = 0;
		return vect3d;
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

struct Circle {
	Vector::Vector2D center;
	float radius;
};
struct Line {
	float d[3];
	float determinant;
};
struct Vertex {
	Vector::Vector2D p;
	Vector::Vector2D uv;
};
struct Triangle2D {
	unsigned int objId;
	Vertex v1, v2, v3;
};
struct Image2D {
	unsigned int objId;
	uint8_t* img;
	Vector::Vector2D size;
};