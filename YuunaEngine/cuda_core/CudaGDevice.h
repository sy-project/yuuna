#pragma once
#include <Types.h>
#include <Singleton.h>
#ifdef _WIN32
#include <windows.h>
#elif __linux__
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/fb.h>
#include <sys/mman.h>
#include <cstring>
#endif

class CudaGDevice
{
public:
	CudaGDevice();
	~CudaGDevice();

public:
	int CheckCUDA();
	void CreateCudaWindow(std::string _name, const int x, const int y, const int WIDTH, const int HEIGHT, unsigned long dwStyle);
	void ImportCudaWindow(std::string _name, const int x, const int y, const int WIDTH, const int HEIGHT);

#ifdef _WIN32
	void displayFramebufferCudaDevice2Windows(uint8_t* framebuffer, const int x, const int y, const int WIDTH, const int HEIGHT);
#elif __linux__
	void displayFramebufferCudaDevice2Linux(uint8_t* framebuffer, const int WIDTH, const int HEIGHT);
#endif
private:
	HWND mhwnd;
	HDC mhdc, mmemDC;
	HBITMAP mhBitmap;
	BITMAPINFO mbmi;
	MSG mmsg;
	int _X, _Y;
	int _W, _H;

};

void Input3DVertex(unsigned int objId, Vertex3D v0, Vertex3D v1, Vertex3D v2);
void Input3DModel(unsigned int objId, std::vector<Vertex3D> v);
void Input3DModelFromTriangle(unsigned int objId, std::vector<Triangle3D> v);
void Input3DTexture(unsigned int objId, int texId, uint8_t* img, int width, int height);
void Update3DVertex(int objId, Vector::Vector3D _vec);
void Update4DVertex(int objId, Vector::Vector4D _vec);
//#ifdef _WIN32
//#define displayFramebufferCudaDevice(f, x, y, w, h) CudaGDevice::Get()->displayFramebufferCudaDevice2Windows(f, x, y, w, h);
//#elif __linux__
//#define displayFramebufferCudaDevice(f, w, h) CudaGDevice::Get()->displayFramebufferCudaDevice2Linux(f, w, h)
//#else
//#define displayFramebufferCudaDevice
//#endif
