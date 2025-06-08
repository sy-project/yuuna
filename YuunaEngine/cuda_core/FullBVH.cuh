#include <cmath>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "cuda_header.h"
#include <Types.h>
#include <vector>

extern void flattenBVH(const std::vector<BVHNode>& nodes, std::vector<LinearBVHNode>& linearNodes);
extern __device__ bool intersectAABB(const AABB& box, const Ray& ray);
extern __device__ bool intersectTriangle(const Triangle3D& tri, const Ray& ray, float& t, float& u, float& v);
extern __device__ bool traverseBVH(const LinearBVHNode* nodes, const Triangle3D* triangles, const int* triIndices, const Ray& ray, int& hitTriIdx, float& tHit, float& u, float& v);

//extern __device__ int traverseBVH(const LinearBVHNode* bvh, const Ray& ray, const Triangle3D* tris, int nodeCount, float& closestT, int& hitTriIdx);