#pragma once
#include "cuda_header.h"
#include "all_type.h"
extern __device__ bool isInside2DTriangle(int x, int y, Vector::Vector2D p0, Vector::Vector2D p1, Vector::Vector2D p2);
extern __device__ void GInverse(float* out, const float* in, int _index);
extern __device__ void GFMultiply(float* out, const float* x, const float* y, int _index);
extern __device__ void GIMultiply(int* out, const int* x, const int* y, int _index);
extern __device__ void GNormalize(float* out, const float* InVec[3], const float* OutVec[3], int _index);
extern __device__ void testfunc(double* val);
//cudaError_t Math_Test(int* c, const int* a, const int* b, unsigned int size);