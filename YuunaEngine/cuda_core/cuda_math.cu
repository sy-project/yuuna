#include "cuda_main.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_math.h"
#include <math.h>
#include "cuda_profiler_api.h"

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
    printf("a");
    //double* db = new double[8];
    //testfunc(db);
    //GMath_Test KERNEL_ARG2(1,1) (c, a, b);
}
cudaError_t Math_Test(int* c, const int* a, const int* b, unsigned int size)
{
    cudaProfilerStart();
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    GMath_Test1 KERNEL_ARG2(1, size) (dev_c, dev_a, dev_b);
    cudaProfilerStop();

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}