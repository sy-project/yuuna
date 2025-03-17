#include "cuda_math.cuh"
#include <math.h>
#include "cuda_profiler_api.h"

__device__ bool isInside2DTriangle(int x, int y, Vector::Vector2D p0, Vector::Vector2D p1, Vector::Vector2D p2) {
    int a = (p1.x - p0.x) * (y - p0.y) - (p1.y - p0.y) * (x - p0.x);
    int b = (p2.x - p1.x) * (y - p1.y) - (p2.y - p1.y) * (x - p1.x);
    int c = (p0.x - p2.x) * (y - p2.y) - (p0.y - p2.y) * (x - p2.x);

    return (a >= 0 && b >= 0 && c >= 0) || (a <= 0 && b <= 0 && c <= 0);
}
__global__ void GMath_Test(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    //GMultiply((float)c[i], (float)a[i], (float)b[i]);
    //if(i % 2 == 0)
        c[i] = a[i] + b[i];
    //else
    //    c[i] = b[i] - a[i];
}
__device__ void GInverse(float* out, const float* in, int _index)
{
    out[_index] = (1.0f / in[_index]);
}
__device__ void GFMultiply(float* out, const float* x, const float* y, int _index)
{
    out[_index] = x[_index] * y[_index];
}
__device__ void GIMultiply(int* out, const int* x, const int* y, int _index)
{
    out[_index] = x[_index] * y[_index];
}
__device__ void GNormalize(float* out, const float* InVec[3], const float* OutVec[3], int _index)
{
    float* Length, InvLength;
    cudaMalloc((void**)&Length, sizeof(out) * sizeof(float));
    cudaMalloc((void**)&InvLength, sizeof(out) * sizeof(float));

    float* mul0[3];
    cudaMalloc((void**)&mul0[0], sizeof(out) * sizeof(float));
    cudaMalloc((void**)&mul0[1], sizeof(out) * sizeof(float));
    cudaMalloc((void**)&mul0[2], sizeof(out) * sizeof(float));

    GFMultiply(mul0[0], InVec[0], InVec[0], _index);
    GFMultiply(mul0[1], InVec[1], InVec[1], _index);
    GFMultiply(mul0[2], InVec[2], InVec[2], _index);

    Length[_index] = mul0[0][_index] + mul0[1][_index] + mul0[2][_index];
}
__device__ void testfunc(double* val)
{
    int i = threadIdx.x;
    sqrt(val[i]);
}
__global__ void GMath_Test1(int* c, const int* a, const int* b) 
{
    int i = threadIdx.x;
    //c[i] = a[i] * b[i];
    GIMultiply(c, a, b, i);
    //double* db = new double[8];
    //testfunc(db);
    //GMath_Test KERNEL_ARG2(1,1) (c, a, b);
}
