#pragma once
#include <iostream>
#include <algorithm>
#include <vector>

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
struct AABB {
	Vector::Vector3D _min;
	Vector::Vector3D _max;

	void expand(const Vector::Vector3D& p) {
		_min.x = std::min(_min.x, p.x);
		_min.y = std::min(_min.y, p.y);
		_min.z = std::min(_min.z, p.z);
		_max.x = std::max(_max.x, p.x);
		_max.y = std::max(_max.y, p.y);
		_max.z = std::max(_max.z, p.z);
	}
};
struct BVHNode {
	AABB bbox;
	int left = -1, right = -1;
	int start = -1, count = 0;
	bool isLeaf = false;
};
struct BVH {
	std::vector<BVHNode> nodes;
	std::vector<int> triangleIndices;

	void build(const std::vector<Triangle3D>& triangles) {
		triangleIndices.resize(triangles.size());
		for (int i = 0; i < triangleIndices.size(); ++i)
			triangleIndices[i] = i;
		std::cout << "== BVH BUILD BEGIN ==" << std::endl;
		std::cout << "triangleIndices size: " << triangleIndices.size() << std::endl;
		std::cout << "triangles size: " << triangles.size() << std::endl;
		for (int i = 0; i < triangleIndices.size(); ++i) {
			if (triangleIndices[i] < 0 || triangleIndices[i] >= (int)triangles.size()) {
				std::cerr << "invalid triangleIndices[" << i << "] = " << triangleIndices[i] << std::endl;
			}
		}
		nodes.clear();
		buildRecursive(triangles, 0, triangleIndices.size());
	}

	int buildRecursive(const std::vector<Triangle3D>& triangles, int start, int end) {
		//std::cout << "[buildRecursive] start: " << start << ", end: " << end << std::endl;
		//std::cout << "triangleIndices size: " << triangleIndices.size() << ", triangles size: " << triangles.size() << std::endl;
		
		AABB bbox = {};
		bbox._min = { FLT_MAX, FLT_MAX, FLT_MAX };
		bbox._max = { -FLT_MAX, -FLT_MAX, -FLT_MAX };

		for (int i = start; i < end; ++i) {
			if (triangleIndices[i] >= (int)triangles.size()) {
				std::cerr << "triangleIndices[" << i << "] = " << triangleIndices[i] << " is out of bounds!" << std::endl;
			}
			const Triangle3D& tri = triangles[triangleIndices[i]];
			bbox.expand(tri.v1.p);
			bbox.expand(tri.v2.p);
			bbox.expand(tri.v3.p);
		}

		int nodeIdx = nodes.size();
		nodes.emplace_back();
		BVHNode& node = nodes[nodeIdx];
		node.bbox = bbox;

		int count = end - start;
		if (count <= 4 || start >= end - 1) {
			node.isLeaf = true;
			node.start = start;
			node.count = count;
			return nodeIdx;
		}

		Vector::Vector3D extent = {
			bbox._max.x - bbox._min.x,
			bbox._max.y - bbox._min.y,
			bbox._max.z - bbox._min.z
		};

		int axis = 0;
		if (extent.y > extent.x) axis = 1;
		if (extent.z > (axis == 0 ? extent.x : extent.y)) axis = 2;

		for (int i = start; i < end; ++i) {
			if (triangleIndices[i] < 0 || triangleIndices[i] >= (int)triangles.size()) {
				std::cerr << "triangleIndices[" << i << "] = " << triangleIndices[i]
					<< " (out of range: 0 ~ " << (triangles.size() - 1) << ")" << std::endl;
			}
		}

		std::sort(triangleIndices.begin() + start, triangleIndices.begin() + end,
			[&](int a, int b) {
				if (a >= (int)triangles.size() || b >= (int)triangles.size()) {
					std::cerr << "sort index out of range: a=" << a << ", b=" << b << std::endl;
					return false;
				}

				const Triangle3D& ta = triangles[a];
				const Triangle3D& tb = triangles[b];

				float ca = 0.0f, cb = 0.0f;
				switch (axis) {
				case 0:
					ca = (ta.v1.p.x + ta.v2.p.x + ta.v3.p.x) / 3.0f;
					cb = (tb.v1.p.x + tb.v2.p.x + tb.v3.p.x) / 3.0f;
					break;
				case 1:
					ca = (ta.v1.p.y + ta.v2.p.y + ta.v3.p.y) / 3.0f;
					cb = (tb.v1.p.y + tb.v2.p.y + tb.v3.p.y) / 3.0f;
					break;
				case 2:
					ca = (ta.v1.p.z + ta.v2.p.z + ta.v3.p.z) / 3.0f;
					cb = (tb.v1.p.z + tb.v2.p.z + tb.v3.p.z) / 3.0f;
					break;
				}

				return ca < cb;
			});

		int mid = (start + end) / 2;
		node.left = buildRecursive(triangles, start, mid);
		node.right = buildRecursive(triangles, mid, end);
		return nodeIdx;
	}
};
struct LinearBVHNode {
	Vector::Vector3D bboxMin;
	Vector::Vector3D bboxMax;
	int left;
	int right;
	int start;
	int count;
	int isLeaf;
};
struct Ray {
	Vector::Vector3D origin;
	Vector::Vector3D direction;
};
struct Camera {
	Vector::Vector3D position;
	Vector::Vector3D forward;
	Vector::Vector3D up;
	float fov;
	float aspectRatio;
};