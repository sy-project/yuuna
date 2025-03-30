#pragma once

#ifdef COMPILE_DLL
#define CUDA_DLL __declspec(dllexport)
#else
#define CUDA_DLL __declspec(dllimport)
#endif // COMPILE_DLL

#include <Types.h>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#elif __linux__
#endif
#include <iostream>
#include <stdio.h>

//extern "C"
//{
	namespace SYCUDA
	{
		extern "C" CUDA_DLL int Check_Cuda();
		namespace GDevice
		{
			extern "C" CUDA_DLL void DeviceInit(std::string winName, int X, int Y, const int WIDTH, const int HEIGHT, unsigned long dwStyle);
			extern "C" CUDA_DLL void DeviceImport(std::string winName, int X, int Y, const int WIDTH, const int HEIGHT);
			extern "C" CUDA_DLL void DeviceRender(uint8_t* h_framebuffer, const int WIDTH = 0, const int HEIGHT = 0);
			extern "C" CUDA_DLL void DeviceDelete();

			extern "C" CUDA_DLL void DeviceInput2DVertex(int objId, Vertex2D v0, Vertex2D v1, Vertex2D v2);
			extern "C" CUDA_DLL void DeviceInput2DImage(int objId, int texId, std::string _path);
			extern "C" CUDA_DLL void DeviceInput2DImageFromEnc(int objId, int texId, std::vector<char> encData = std::vector<char>());
			extern "C" CUDA_DLL void DeviceUpdate2DVertexPos(int objId, Vector::Vector2D v);
			extern "C" CUDA_DLL void DeviceUpdate2DVertexRot(int objId, float val);
			extern "C" CUDA_DLL void DeviceUpdate2DVertexRotWCenter(int objId, float val, Vector::Vector2D center);

			extern "C" CUDA_DLL void DeviceInput3DVertex(int objId, Vertex3D v0, Vertex3D v1, Vertex3D v2);
			extern "C" CUDA_DLL void DeviceInput3DTriangle(unsigned int objId, std::vector<Triangle3D> v);
			extern "C" CUDA_DLL void DeviceInput3DModel(int objId, std::vector<Vertex3D> v);
			extern "C" CUDA_DLL void DeviceInput3DTex(int objId, int texId, uint8_t * img, int width, int height);
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