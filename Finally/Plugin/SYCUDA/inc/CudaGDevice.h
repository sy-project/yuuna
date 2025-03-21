#pragma once
#include <Types.h>

void createWindow(std::string _name, const int x, const int y, const int WIDTH, const int HEIGHT, unsigned long dwStyle);
void ImportWindow(std::string _name, const int x, const int y, const int WIDTH, const int HEIGHT);
void initFramebufferCudaDevice(const int WIDTH, const int HEIGHT);
void renderCudaDevice(const int WIDTH, const int HEIGHT);
void RenderVertexDevice(const int WIDTH, const int HEIGHT);
void copyFramebufferToCPUCudaDevice(uint8_t* h_framebuffer, const int WIDTH, const int HEIGHT);
void DeleteCudaDevice();
void Input2DVertex(unsigned int objId, Vertex2D v0, Vertex2D v1, Vertex2D v2);
void Input2DImage(unsigned int objId, std::string _path);
void Update2DVertexPos(unsigned int objId, Vector::Vector2D v);
void Update2DVertexRot(unsigned int objId, float _val);
void Update2DVertexRot(unsigned int objId, float _val, Vector::Vector2D center);
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
