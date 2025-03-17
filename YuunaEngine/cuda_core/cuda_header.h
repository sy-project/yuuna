#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#ifdef __INTELLISENSE__

#define __CUDACC__

#define __global__ 
#define __host__ 
#define __device__ 
#define __device_builtin__
#define __device_builtin_texture_type__
#define __device_builtin_surface_type__
#define __cudart_builtin__
#define __constant__ 
#define __shared__ 
#define __restrict__
#define __noinline__
#define __forceinline__
#define __managed__

//KERNEL_ARG2(grid, block) : <<< grid, block >>>
#define KERNEL_ARG2(grid, block)
//KERNEL_ARG3(grid, block, sh_mem) : <<< grid, block, sh_mem >>>
#define KERNEL_ARG3(grid, block, sh_mem)
//KERNEL_ARG4(grid, block, sh_mem, stream) : <<< grid, block, sh_mem, stream >>>
#define KERNEL_ARG4(grid, block, sh_mem, stream)

#else

#define KERNEL_ARG2(grid, block) <<< grid, block >>>
#define KERNEL_ARG3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARG4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>

#endif