#pragma once
#include "CudaTypes.h"

#ifdef COMPILE_DLL
#define CUDA_DLL __declspec(dllexport)
#else
#define CUDA_DLL __declspec(dllimport)
#endif // COMPILE_DLL

#include <Types.h>
#include "CudaBVH.h"
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
		extern "C" CUDA_DLL int Check_Cuda(sWindow& _window);
		namespace GDevice
		{
			extern "C" CUDA_DLL void DeviceInit(sWindow& _window);
			extern "C" CUDA_DLL void DeviceImport(sWindow& _window);
			extern "C" CUDA_DLL void DeviceRender(sWindow& _window, int _split = 1);
			extern "C" CUDA_DLL void DeviceDelete();

			extern "C" CUDA_DLL void DeviceInput2DVertex(sWindow& _window, int objId, Vertex2D v0, Vertex2D v1, Vertex2D v2);
			extern "C" CUDA_DLL void DeviceInput2DImage(sWindow& _window, int objId, int texId, std::string _path);
			extern "C" CUDA_DLL void DeviceInput2DImageFromEnc(sWindow& _window, int objId, int texId, std::vector<char> encData = std::vector<char>());
			extern "C" CUDA_DLL void DeviceUpdate2DVertexPos(sWindow& _window, int objId, Vector::Vector2D v);
			extern "C" CUDA_DLL void DeviceUpdate2DVertexRot(sWindow& _window, int objId, float val);
			extern "C" CUDA_DLL void DeviceUpdate2DVertexRotWCenter(sWindow& _window, int objId, float val, Vector::Vector2D center);

			extern "C" CUDA_DLL void DeviceInput3DVertex(int objId, Vertex3D v0, Vertex3D v1, Vertex3D v2);
			extern "C" CUDA_DLL void DeviceInput3DTriangle(unsigned int objId, std::vector<Triangle3D> v);
			extern "C" CUDA_DLL void DeviceInput3DModel(int objId, std::vector<Vertex3D> v);

			extern "C" CUDA_DLL void DeviceInput3DTex(int objId, int texId, uint8_t * img, int width, int height);
			extern "C" CUDA_DLL void DeviceUpdate3DVertex(int objId, Vector::Vector3D _vec);
			extern "C" CUDA_DLL void DeviceUpdate4DVertex(int objId, Vector::Vector4D _vec);


			extern "C" CUDA_DLL bool DeviceBuildBVH(unsigned int objId);
			extern "C" CUDA_DLL bool DeviceBuildAllBVH();
			extern "C" CUDA_DLL bool DeviceCreateTemporaryBVH(const std::vector<Triangle3D>& triangles, unsigned int* objId);
			extern "C" CUDA_DLL bool DevicePinBVH(unsigned int sourceId, unsigned int* aliasId);
			extern "C" CUDA_DLL bool DeviceGetBVHStats(unsigned int objId, BVHStats& stats);

			extern "C" CUDA_DLL bool DeviceCopyBVH(unsigned int objId, std::vector<BVHNode>& nodes,
				std::vector<int>& indices, std::vector<Triangle3D>& triangles, bool fromDevice = false);
			extern "C" CUDA_DLL void DeviceRemoveBVH(unsigned int objId);
			extern "C" CUDA_DLL void DeviceClearBVH();
			extern "C" CUDA_DLL const char* DeviceGetBVHError();


			extern "C" CUDA_DLL bool DeviceTraceBVHRays(unsigned int objId, const std::vector<BVHRay>& rays, std::vector<BVHHit>& hits);

			extern "C" CUDA_DLL bool DeviceRenderBVHFrame(unsigned int objId, const Camera& camera, int width, int height,
				uint8_t* framebuffer, const BVHDebugOptions& debug);
			extern "C" CUDA_DLL bool DeviceRender3D(sWindow& window, unsigned int objId, const Camera& camera, const BVHDebugOptions& debug);


			extern "C" CUDA_DLL bool DeviceRenderSceneFrame(const std::vector<SceneInstance>& instances,
				const std::vector<PointLight>& lights, const Camera& camera, int width, int height, uint8_t* framebuffer, const BVHDebugOptions& debug);
			extern "C" CUDA_DLL bool DeviceRenderScene(sWindow& window, const std::vector<SceneInstance>& instances,
				const std::vector<PointLight>& lights, const Camera& camera, const BVHDebugOptions& debug);
			extern "C" CUDA_DLL const char* DeviceGetBVHRenderError();
			extern "C" CUDA_DLL void DeviceReleaseBVHRenderer();
			extern "C" CUDA_DLL const char* DeviceGetRenderBackend();
			extern "C" CUDA_DLL bool DeviceSelectRenderBackend(const char* mode);
			extern "C" CUDA_DLL bool DeviceGetRenderMemory(RenderMemory& memory);
			extern "C" CUDA_DLL bool DevicePrepareRenderBackend(int waitMilliseconds);
            CUDA_DLL RenderStats DeviceGetRenderStats();



            extern "C" CUDA_DLL bool DeviceCopyRayDepths(std::vector<std::uint16_t>& depths, int& width, int& height);


			extern "C" CUDA_DLL bool DeviceInput3DMaterials(unsigned int objId, const std::vector<BVHMaterial>& materials);


			extern "C" CUDA_DLL bool DeviceInput3DSpectralMaterials(unsigned int objId,
				const std::vector<SpectralMaterial>& materials, const SpectralObserver& observer);
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
