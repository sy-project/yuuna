#include <cuda_main.h>
#include <iostream>

#pragma comment(lib, "cuda_core.lib")

int main()
{
    int X, Y, WIDTH, HEIGHT;

    X = 100;
    Y = 200;
    WIDTH = 800;
    HEIGHT = 600;

    cuda_dll::GDevice::DeviceInit(WIDTH, HEIGHT);
    cuda_dll::GDevice::DeviceRender(WIDTH, HEIGHT);

    uint8_t* h_framebuffer = new uint8_t[WIDTH * HEIGHT * 4];
    cuda_dll::GDevice::DeviceCopyFramebuffer(h_framebuffer, WIDTH, HEIGHT);

    while (true)
    {
        cuda_dll::GDevice::DevicedisplayFramebuffer(h_framebuffer, X, Y, WIDTH, HEIGHT);
    }

    delete[] h_framebuffer;
    cuda_dll::GDevice::DeviceDelete();
    return 0;
}