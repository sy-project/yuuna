#include "cuda_main.h"
#include "cuda_header.h"
#include "SoundTracingPlugin.h"
#include "cuda_math.cuh"
#include "CudaGDevice.h"
#include "CudaGCamera.h"
#include "cCudaMainRuntimeClass.h"

BOOL APIENTRY DllMain(HMODULE hModule,
    DWORD  ul_reason_for_call,
    LPVOID lpReserved
)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

extern "C" int SYCUDA::Check_Cuda(sWindow& _window)
{
    if (!cCudaMainRuntimeClass::Get()->CheckCUDAUsage(_window)) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return -1;
    }
    return 0;
}

extern "C" void SYCUDA::GDevice::DeviceInit(sWindow& _window)
{
    cCudaMainRuntimeClass::Get()->Init(_window);
}

extern "C" void SYCUDA::GDevice::DeviceImport(sWindow& _window)
{
    cCudaMainRuntimeClass::Get()->ImportWindow(_window);
}

extern "C" void SYCUDA::GDevice::DeviceRender(sWindow& _window)
{
    cCudaMainRuntimeClass::Get()->Rendering(_window);
}

extern "C" void SYCUDA::GDevice::DeviceDelete()
{
    cCudaMainRuntimeClass::Delete();
}

extern "C"  void SYCUDA::GDevice::DeviceInput2DVertex(sWindow& _window, int objId, Vertex2D v0, Vertex2D v1, Vertex2D v2)
{
    cCudaMainRuntimeClass::Get()->Import2DVertex(_window, objId, v0, v1, v2);
}

extern "C" void SYCUDA::GDevice::DeviceInput2DImage(sWindow& _window, int objId, int texId, std::string _path)
{
    cCudaMainRuntimeClass::Get()->ImportImageFor2DV(_window, objId, texId, _path, std::vector<char>());
}

extern "C" void SYCUDA::GDevice::DeviceInput2DImageFromEnc(sWindow& _window, int objId, int texId, std::vector<char> encData)
{
    cCudaMainRuntimeClass::Get()->ImportImageFor2DV(_window, objId, texId, "", encData);
}

extern "C" void SYCUDA::GDevice::DeviceUpdate2DVertexPos(sWindow& _window, int objId, Vector::Vector2D v)
{
    cCudaMainRuntimeClass::Get()->Update2DVertexPos(_window, objId, v);
}

extern "C" void SYCUDA::GDevice::DeviceUpdate2DVertexRot(sWindow& _window, int objId, float val)
{
    cCudaMainRuntimeClass::Get()->Update2DVertexRot(_window, objId, val);
}

extern "C" void SYCUDA::GDevice::DeviceUpdate2DVertexRotWCenter(sWindow& _window, int objId, float val, Vector::Vector2D center)
{
    cCudaMainRuntimeClass::Get()->Update2DVertexRot(_window, objId, val, center);
}

extern "C" void SYCUDA::GDevice::DeviceInput3DVertex(int objId, Vertex3D v0, Vertex3D v1, Vertex3D v2)
{
    Input3DVertex(objId, v0, v1, v2);
}
extern "C" void SYCUDA::GDevice::DeviceInput3DTriangle(unsigned int objId, std::vector<Triangle3D> v)
{
    Input3DModelFromTriangle(objId, v);
}
extern "C" void SYCUDA::GDevice::DeviceInput3DModel(int objId, std::vector<Vertex3D> v)
{
    Input3DModel(objId, v);
}
extern "C" void SYCUDA::GDevice::DeviceInput3DTex(int objId, int texId, uint8_t* img, int width, int height)
{
    Input3DTexture(objId, texId, img, width, height);
}
void SYCUDA::GDevice::DeviceUpdate3DVertex(int objId, Vector::Vector3D _vec)
{
    Update3DVertex(objId, _vec);
}

extern "C"  void SYCUDA::GDevice::DeviceUpdate4DVertex(int objId, Vector::Vector4D _vec)
{
    Update4DVertex(objId, _vec);
}

extern "C" void SYCUDA::math::CudaMath_test(int* c, const int* a, const int* b, unsigned int size)
{
//    Math_Test(c, a, b, size);
}

extern "C" void* SYCUDA::physics::SoundTracing::GetSTInstance()
{
    return SoundTracer::Get();
}
