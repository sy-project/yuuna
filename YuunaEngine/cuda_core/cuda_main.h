#pragma once

#ifdef COMPILE_DLL
#define CUDA_DLL __declspec(dllexport)
#else
#define CUDA_DLL __declspec(dllimport)
#endif // COMPILE_DLL

#include "cuda_header.h"
#include <windows.h>
#include <stdio.h>

extern "C"
{
	namespace cuda_dll
	{
		namespace math
		{
			CUDA_DLL void CudaMath_test(int* c, const int* a, const int* b, unsigned int size);
		}
		namespace physics
		{
			namespace SoundTracing
			{
				CUDA_DLL void* GetInstance();
			}
		}
	}
}