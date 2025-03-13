#include "cuda_main.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SoundTracingPlugin.h"
#include "cuda_math.h"
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

extern "C" int cuda_dll::Check_Cuda()
{
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return -1;
    }
    return 0;
}

extern "C" void cuda_dll::GDevice::DeviceInit(const int WIDTH, const int HEIGHT)
{
    initFramebufferCudaDevice(WIDTH, HEIGHT);
}

extern "C" void cuda_dll::GDevice::DeviceRender(const int WIDTH, const int HEIGHT)
{
    renderCudaDevice(WIDTH, HEIGHT);
}

extern "C" void cuda_dll::GDevice::DeviceCopyFramebuffer(uint8_t* h_framebuffer, const int WIDTH, const int HEIGHT)
{
    copyFramebufferToCPUCudaDevice(h_framebuffer, WIDTH, HEIGHT);
}

extern "C" void cuda_dll::GDevice::DevicedisplayFramebuffer(uint8_t* framebuffer, const int x, const int y, const int WIDTH, const int HEIGHT)
{
    displayFramebufferCudaDevice(framebuffer, x, y, WIDTH, HEIGHT);
}

extern "C" void cuda_dll::GDevice::DeviceDelete()
{
    DeleteCudaDevice();
}

extern "C" void cuda_dll::math::CudaMath_test(int* c, const int* a, const int* b, unsigned int size)
{
    Math_Test(c, a, b, size);
}

extern "C" void* cuda_dll::physics::SoundTracing::GetInstance()
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
