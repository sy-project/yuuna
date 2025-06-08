#pragma once
#ifndef CUDACC
#define CUDACC
#endif
#include <cmath>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "cuda_header.h"
#include <Types.h>

extern __host__ __device__ Triangle2D* Input2DVertex(unsigned int objId, int tricount, Vertex2D v0, Vertex2D v1, Vertex2D v2, Triangle2D* src);
