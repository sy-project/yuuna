#pragma once

#ifdef COMPILE_DLL
#define CUDA_DLL __declspec(dllexport)
#else
#define CUDA_DLL __declspec(dllimport)
#endif // COMPILE_DLL

#include "all_type.h"
#include <windows.h>
#include <iostream>
#include <stdio.h>

//extern "C"
//{
	namespace SYCUDA
	{
		extern "C" CUDA_DLL int Check_Cuda();
		namespace GDevice
		{
			extern "C" CUDA_DLL void DeviceInit(std::string winName, const int WIDTH, const int HEIGHT);
			extern "C" CUDA_DLL void DeviceRender(uint8_t* h_framebuffer, const int WIDTH = 0, const int HEIGHT = 0);
			extern "C" CUDA_DLL void DeviceDelete();

			extern "C" CUDA_DLL void DeviceUpdate2DVertex(Vector::Vector2D v0, Vector::Vector2D v1, Vector::Vector2D v2);
			extern "C" CUDA_DLL void DeviceUpdate3DVertex(int objId, Vector::Vector3D _vec);
			extern "C" CUDA_DLL void DeviceUpdate4DVertex(int objId, Vector::Vector4D _vec);
		}
		namespace math
		{
			extern "C" CUDA_DLL void CudaMath_test(int* c, const int* a, const int* b, unsigned int size);
		}
		namespace physics
		{
			namespace SoundTracing
			{
				extern "C" CUDA_DLL void* GetSTInstance();
			}
		}
	}
//}