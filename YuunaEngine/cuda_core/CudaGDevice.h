#pragma once
#include "all_type.h"

void createWindow(std::string _name, const int x, const int y, const int WIDTH, const int HEIGHT, unsigned long dwStyle);
void initFramebufferCudaDevice(const int WIDTH, const int HEIGHT);
void renderCudaDevice(const int WIDTH, const int HEIGHT);
void RenderVertexDevice(const int WIDTH, const int HEIGHT);
void copyFramebufferToCPUCudaDevice(uint8_t* h_framebuffer, const int WIDTH, const int HEIGHT);
void DeleteCudaDevice();
void Update2DVertex(Vector::Vector2D v0, Vector::Vector2D v1, Vector::Vector2D v2);
void Update3DVertex(int objId, Vector::Vector3D _vec);
void Update4DVertex(int objId, Vector::Vector4D _vec);
#ifdef _WIN32
#define displayFramebufferCudaDevice(f, x, y, w, h) displayFramebufferCudaDevice2Windows(f, x, y, w, h)
void displayFramebufferCudaDevice2Windows(uint8_t* framebuffer, const int x, const int y, const int WIDTH, const int HEIGHT);
#elif __linux__
#define displayFramebufferCudaDevice(f, w, h) displayFramebufferCudaDevice2Linux(f, w, h)
void displayFramebufferCudaDevice2Linux(uint8_t* framebuffer, const int WIDTH, const int HEIGHT);
#else
#define displayFramebufferCudaDevice
#endif
