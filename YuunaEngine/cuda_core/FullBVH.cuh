#include <cmath>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "cuda_header.h"
#include <Types.h>
#include <Singleton.h>
#include <vector>

#define Max(a,b)            (((a) > (b)) ? (a) : (b))
#define Min(a,b)            (((a) < (b)) ? (a) : (b))

struct BBox
{
	Vector::Vector3D min, max;
	__device__ Vector::Vector3D GetCenter() {
		return (min + max) * 0.5f;
	}
	__device__ __host__ Vector::Vector3D GetDiag() const {
		return max - min;
	}
	__host__ BBox& MakeEmpty() {
		BBox a;
		a.min = { 0.0f,0.0f,0.0f };
		a.max = { 0.0f,0.0f,0.0f };
		return a;
	}
	BBox& MakeEmpty(const BBox& other) {
		BBox a;
		a.min = { other.min.x,other.min.y,other.min.z };
		a.max = { other.max.x,other.max.y,other.max.z };
		return a;
	}
	BBox& MakeEmpty(const Vector::Vector3D& v1, const Vector::Vector3D& v2) {
		BBox a;
		a.min = { v1.x,v1.y,v1.z };
		a.max = { v2.x,v2.y,v2.z };
		return a;
	}
	BBox& MakeEmpty(const Vector::Vector3D& v1) {
		BBox a;
		a.min = { v1.x,v1.y,v1.z };
		a.max = { v1.x,v1.y,v1.z };
		return a;
	}
	BBox Union(const BBox& other)	{
		BBox a;
		a.min.x = Min(this->min.x, other.min.x);
		a.min.y = Min(this->min.y, other.min.y);
		a.min.z = Min(this->min.z, other.min.z);
		a.max.x = Max(this->max.x, other.max.x);
		a.max.y = Max(this->max.y, other.max.y);
		a.max.z = Max(this->max.z, other.max.z);
		return a;
	}
	Vector::Vector3D get_offset(const Vector::Vector3D& p) const {
		Vector::Vector3D o = p - min;
		Vector::Vector3D d = GetDiag();
		if (d.x > 1e-5f) o.x /= d.x;
		if (d.y > 1e-5f) o.y /= d.y;
		if (d.z > 1e-5f) o.z /= d.z;
		return o;
	}
	float get_surface_area() const {
		Vector::Vector3D d = GetDiag();
		if (d.x < 0 || d.y < 0 || d.z < 0) return 0.0f;
		return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
	}
};

struct Node
{
	BBox box;
	Node* left = nullptr;
	Node* right = nullptr;
	int first_primitive_index = -1;
	int num_primitives = 0;

	Node(int idx, int n, const BBox& b)
	{
		first_primitive_index = idx;
		num_primitives = n;
		box = b;
	}
	Node(Node* left_child, Node* right_child)
	{
		left = left_child;
		right = right_child;
		box = (left_child->box).Union(right_child->box);
		first_primitive_index = left->first_primitive_index;
		num_primitives = left->num_primitives + right->num_primitives;
	}
};

struct SplitInfo
{
	BBox centro_bound;
	BBox node_bound;
	bool use_binning = false;
	int NUM_BINS = 0;
	float min_cost = 3.40282e+38;
	int mid;
	int min_cost_split_bin = -1;
	float split_position = 0.0f;
	BBox left_bound;
	BBox right_bound;
	SplitInfo(const BBox& node)
		: node_bound(node) {
	}
};

struct Bin
{
	BBox bounds;
	int count = 0;
	int enter = 0;
	int exit = 0;
};

class BVH : public Singleton<BVH>
{
	friend class Singleton;
private:
	struct Prim {
		int tri_start_ind;
		BBox box;
		Vector::Vector3D cent;
	};
	struct BVHData
	{
		std::vector<Node*> BVHNode;
		std::vector<std::vector<BBox>> BVHLeafBox;
		std::vector<std::vector<BBox>> BVHBox;
	};

private:
	BVHData data;
	std::vector<std::pair<BVHData, int>> BVHobj;

public:
	bool BuildBVH(Render3DData& _ObjData, int _objId, int _BVHType = 1);
	__device__ BVHData GetBVH(int _objId);

private:
	void BuildAABB(Render3DData& _ObjData, int _objId);
	Node* build_recursive_SAH(std::vector<Triangle3D>& tris, std::vector<Prim>& prims, std::vector<int>& prim_indices, int start, int end, int depth, std::vector<Prim>& ord_prim, int& total_count);

private:
	bool isLeaf(Node* cur);
	Node* create_leaf(std::vector<Triangle3D>& tris, std::vector<Prim>& prims, std::vector<int>& primitive_indices, int start, int end, std::vector<Prim>& ordered_primitives);
	void FindBoundingVolumes(Node* cur, int depth, int& max_current_depth);

private:
	BVH();
	~BVH();
};
//extern void flattenBVH(const std::vector<BVHNode>& nodes, std::vector<LinearBVHNode>& linearNodes);
//extern __device__ bool intersectAABB(const AABB& box, const Ray& ray);
//extern __device__ bool intersectTriangle(const Triangle3D& tri, const Ray& ray, float& t, float& u, float& v);
//extern __device__ bool traverseBVH(const LinearBVHNode* nodes, const Triangle3D* triangles, const int* triIndices, const Ray& ray, int& hitTriIdx, float& tHit, float& u, float& v);

//extern __device__ int traverseBVH(const LinearBVHNode* bvh, const Ray& ray, const Triangle3D* tris, int nodeCount, float& closestT, int& hitTriIdx);