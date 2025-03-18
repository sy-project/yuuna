#pragma once
#ifndef CUDACC
#define CUDACC
#endif
#include <cmath>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "cuda_header.h"
#include "all_type.h"
extern __device__ bool isInside2DTriangle(Vector::Vector2D p, Vector::Vector2D a, Vector::Vector2D b, Vector::Vector2D c);
extern __device__ void GInverse(float* out, const float* in, int _index);
extern __device__ void GFMultiply(float* out, const float* x, const float* y, int _index);
extern __device__ void GIMultiply(int* out, const int* x, const int* y, int _index);
extern __device__ void GNormalize(float* out, const float* InVec[3], const float* OutVec[3], int _index);
extern __device__ void testfunc(double* val);
//cudaError_t Math_Test(int* c, const int* a, const int* b, unsigned int size);