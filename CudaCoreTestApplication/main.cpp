#include <cuda_main.h>
#include <iostream>

#pragma comment(lib, "cuda_core.lib")

int main()
{
    std::string winName = "test";

    int WIDTH, HEIGHT;

    WIDTH = 800;
    HEIGHT = 600;

    cuda_dll::GDevice::DeviceInit(winName, WIDTH, HEIGHT);

    uint8_t* h_framebuffer = new uint8_t[WIDTH * HEIGHT * 4];

    while (true)
    {
        cuda_dll::GDevice::DeviceRender(h_framebuffer, WIDTH, HEIGHT);
    }

    delete[] h_framebuffer;
    cuda_dll::GDevice::DeviceDelete();
    return 0;
}