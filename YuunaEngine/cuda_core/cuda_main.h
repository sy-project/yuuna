#pragma once

#ifdef COMPILE_DLL
#define CUDA_DLL __declspec(dllexport)
#else
#define CUDA_DLL __declspec(dllimport)
#endif // COMPILE_DLL

#include "cuda_header.h"
#include <windows.h>
#include <iostream>
#include <stdio.h>

//extern "C"
//{
	namespace cuda_dll
	{
		extern "C" CUDA_DLL int Check_Cuda();
		namespace GDevice
		{
			extern "C" CUDA_DLL void DeviceInit(const int WIDTH, const int HEIGHT);
			extern "C" CUDA_DLL void DeviceRender(const int WIDTH, const int HEIGHT);
			extern "C"  CUDA_DLL void DeviceCopyFramebuffer(uint8_t* h_framebuffer, const int WIDTH, const int HEIGHT);
			extern "C" CUDA_DLL void DevicedisplayFramebuffer(uint8_t* framebuffer, const int x, const int y, const int WIDTH, const int HEIGHT);
			extern "C" CUDA_DLL void DeviceDelete();
		}
		namespace math
		{
			extern "C" CUDA_DLL void CudaMath_test(int* c, const int* a, const int* b, unsigned int size);
		}
		namespace physics
		{
			namespace SoundTracing
			{
				extern "C" CUDA_DLL void* GetInstance();
			}
		}
	}
//}