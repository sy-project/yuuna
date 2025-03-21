#include "cuda_main.h"
#include "cuda_header.h"
#include "SoundTracingPlugin.h"
#include "cuda_math.cuh"
#include "CudaGDevice.h"

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

extern "C" int SYCUDA::Check_Cuda()
{
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return -1;
    }
    return 0;
}

extern "C" void SYCUDA::GDevice::DeviceInit(std::string winName, int X, int Y, const int WIDTH, const int HEIGHT, unsigned long dwStyle)
{
    initFramebufferCudaDevice(WIDTH, HEIGHT);

#ifdef _WIN32
    createWindow(winName, X, Y, WIDTH, HEIGHT, dwStyle);
#elif __linux__
#endif
}

extern "C" void SYCUDA::GDevice::DeviceImport(std::string winName, int X, int Y, const int WIDTH, const int HEIGHT)
{
#ifdef _WIN32
    ImportWindow(winName, X, Y, WIDTH, HEIGHT);
#elif __linux__
#endif
}

//extern "C" void cuda_dll::GDevice::DeviceRender(const int WIDTH, const int HEIGHT)
//{
//    renderCudaDevice(WIDTH, HEIGHT);
//}

extern "C" void SYCUDA::GDevice::DeviceRender(uint8_t* h_framebuffer, const int WIDTH, const int HEIGHT)
{
    renderCudaDevice(WIDTH, HEIGHT);
    copyFramebufferToCPUCudaDevice(h_framebuffer, WIDTH, HEIGHT);
    displayFramebufferCudaDevice(h_framebuffer, 0, 0, WIDTH, HEIGHT);
}
//
//extern "C" void cuda_dll::GDevice::DevicedisplayFramebuffer(uint8_t* framebuffer, const int x, const int y, const int WIDTH, const int HEIGHT)
//{
//    displayFramebufferCudaDevice(framebuffer, x, y, WIDTH, HEIGHT);
//}

extern "C" void SYCUDA::GDevice::DeviceDelete()
{
    DeleteCudaDevice();
}

extern "C"  void SYCUDA::GDevice::DeviceInput2DVertex(int objId, Vertex2D v0, Vertex2D v1, Vertex2D v2)
{
    Input2DVertex(objId, v0, v1, v2);
}

extern "C" void SYCUDA::GDevice::DeviceInput2DImage(int objId, std::string _path)
{
    Input2DImage(objId, _path, std::vector<char>());
}

extern "C" void SYCUDA::GDevice::DeviceInput2DImageFromEnc(int objId, std::vector<char> encData)
{
    Input2DImage(objId, "", encData);
}

extern "C" void SYCUDA::GDevice::DeviceUpdate2DVertexPos(int objId, Vector::Vector2D v)
{
    Update2DVertexPos(objId, v);
}

extern "C" void SYCUDA::GDevice::DeviceUpdate2DVertexRot(int objId, float val)
{
    Update2DVertexRot(objId, val);
}

extern "C" void SYCUDA::GDevice::DeviceUpdate2DVertexRotWCenter(int objId, float val, Vector::Vector2D center)
{
    Update2DVertexRot(objId, val, center);
}

extern "C" void SYCUDA::GDevice::DeviceInput3DVertex(int objId, Vertex3D v0, Vertex3D v1, Vertex3D v2)
{
    Input3DVertex(objId, v0, v1, v2);
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

//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}
