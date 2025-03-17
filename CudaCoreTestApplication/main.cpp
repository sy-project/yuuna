#include <cuda_main.h>
#include <iostream>

#pragma comment(lib, "SYCUDA.lib")

int main()
{
    std::string winName = "test";

    int WIDTH, HEIGHT;

    WIDTH = 800;
    HEIGHT = 600;

    SYCUDA::GDevice::DeviceInit(winName, WIDTH, HEIGHT);

    uint8_t* h_framebuffer = new uint8_t[WIDTH * HEIGHT * 4];
    SYCUDA::GDevice::DeviceUpdate2DVertex({ 30, 50 }, { 100, 50 }, { 100, 100 });

    while (true)
    {
        SYCUDA::GDevice::DeviceRender(h_framebuffer);
    }

    delete[] h_framebuffer;
    SYCUDA::GDevice::DeviceDelete();
    return 0;
}