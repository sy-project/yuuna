#include "cuda_math.cuh"
#include <math.h>
#include "cuda_profiler_api.h"

__device__ bool isInside2DTriangle(Vector::Vector2D p, Vector::Vector2D a, Vector::Vector2D b, Vector::Vector2D c) {
    Vector::Vector2D AB = { b.x - a.x, b.y - a.y };
    Vector::Vector2D BC = { c.x - b.x, c.y - b.y };
    Vector::Vector2D CA = { a.x - c.x, a.y - c.y };

    Vector::Vector2D AP = { p.x - a.x, p.y - a.y };
    Vector::Vector2D BP = { p.x - b.x, p.y - b.y };
    Vector::Vector2D CP = { p.x - c.x, p.y - c.y };

    float cross1 = AB.x * AP.y - AB.y * AP.x;  // AB x AP
    float cross2 = BC.x * BP.y - BC.y * BP.x;  // BC x BP
    float cross3 = CA.x * CP.y - CA.y * CP.x;  // CA x CP

    return (cross1 >= 0 && cross2 >= 0 && cross3 >= 0) || (cross1 <= 0 && cross2 <= 0 && cross3 <= 0);
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
