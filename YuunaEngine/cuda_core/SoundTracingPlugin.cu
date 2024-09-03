#include "cuda_main.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SoundTracingPlugin.h"
#include "cuda_math.h"

void SoundTracer::Init()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };
    cudaError_t cudaStatus = Math_Test(c, a, b, arraySize);
}

SoundTracer::SoundTracer()
{
    Init();
}

SoundTracer::~SoundTracer()
{
}
