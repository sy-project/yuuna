#pragma once

void createWindow(std::string _name, const int x, const int y, const int WIDTH, const int HEIGHT);
void initFramebufferCudaDevice(const int WIDTH, const int HEIGHT);
void renderCudaDevice(const int WIDTH, const int HEIGHT);
void copyFramebufferToCPUCudaDevice(uint8_t* h_framebuffer, const int WIDTH, const int HEIGHT);
void DeleteCudaDevice();
#ifdef _WIN32
#define displayFramebufferCudaDevice(f, x, y, w, h) displayFramebufferCudaDevice2Windows(f, x, y, w, h)
void displayFramebufferCudaDevice2Windows(uint8_t* framebuffer, const int x, const int y, const int WIDTH, const int HEIGHT);
#elif __linux__
#define displayFramebufferCudaDevice(f, w, h) displayFramebufferCudaDevice2Linux(f, w, h)
void displayFramebufferCudaDevice2Linux(uint8_t* framebuffer, const int WIDTH, const int HEIGHT);
#endif
