#include "cCudaMainRuntimeClass.h"
#include "CudaGDevice.h"
#include "CudaGCamera.h"

bool cCudaMainRuntimeClass::CheckCUDAUsage(sWindow& _window)
{
	if (pDevice->CheckCUDA() == 0)
	{
		_window.CUDA_FLAG ^= eCUDA::READ_CUDA;
		return true;
	}
	return false;
}

void cCudaMainRuntimeClass::Init(sWindow& _window)
{
	if (_window.framebuffer == 0)
	{
		uint8_t* h_framebuffer = new uint8_t[_window.size.x * _window.size.y * 4];
		_window.framebuffer = h_framebuffer;
	}
	pCamera->SetFOV(90.0f);
	pCamera->SetWH(_window.size.x, _window.size.y);
	pCamera->SetPosition(0, 0, 5);
	pCamera->SetForward(0, 0, -1);
	pCamera->SetUp(0, 1, 0);
	pCamera->SetPosition(0.0f, 13.0f, 1.0f);
	pCamera->SetForward(0.0f, 0.0f, -1.0f);
	pCamera->SetFrameBuffer();
#ifdef _WIN32
	pDevice->CreateCudaWindow(_window.winName, _window.pos.x, _window.pos.y, _window.size.x, _window.size.y, _window.dwStyle);
#elif __linux__
#endif
	pCamera->SetMultiRendering(_window.framebuffer, 1);
}

void cCudaMainRuntimeClass::ImportWindow(sWindow& _window)
{
#ifdef _WIN32
	pDevice->ImportCudaWindow(_window.winName, _window.pos.x, _window.pos.y, _window.size.x, _window.size.y);
#elif __linux__
#endif
}

void cCudaMainRuntimeClass::Rendering(sWindow& _window, int _split)
{
	if (!(_window.CUDA_FLAG & eCUDA::READ_CUDA))
		return;
	pCamera->Render(_window.framebuffer, _split);
	pDevice->displayFramebufferCudaDevice2Windows(_window.framebuffer, _window.pos.x, _window.pos.y, _window.size.x, _window.size.y);
}

void cCudaMainRuntimeClass::Import2DVertex(sWindow& _window, int objId, Vertex2D v0, Vertex2D v1, Vertex2D v2)
{
	if (!(_window.CUDA_FLAG & eCUDA::READ_CUDA))
		return;
	pCamera->Input2DVertexToCam(objId, v0, v1, v2);
}

void cCudaMainRuntimeClass::ImportImageFor2DV(sWindow& _window, unsigned int objId, unsigned int texId, std::string _path, std::vector<char> encData)
{
	if (!(_window.CUDA_FLAG & eCUDA::READ_CUDA))
		return;
	pCamera->Input2DImageTo2DVertex(objId, texId, _path, encData);
}

void cCudaMainRuntimeClass::Update2DVertexPos(sWindow& _window, unsigned int objId, Vector::Vector2D v)
{
	if (!(_window.CUDA_FLAG & eCUDA::READ_CUDA))
		return;
	pVCont->Update2DVertexPos(objId, v);
}

void cCudaMainRuntimeClass::Update2DVertexRot(sWindow& _window, unsigned int objId, float _val)
{
	if (!(_window.CUDA_FLAG & eCUDA::READ_CUDA))
		return;
	pVCont->Update2DVertexRot(objId, _val);
}

void cCudaMainRuntimeClass::Update2DVertexRot(sWindow& _window, unsigned int objId, float _val, Vector::Vector2D center)
{
	if (!(_window.CUDA_FLAG & eCUDA::READ_CUDA))
		return;
	pVCont->Update2DVertexRot(objId, _val, center);
}

void cCudaMainRuntimeClass::buildBVH(sWindow& _window)
{
	if (!(_window.CUDA_FLAG & eCUDA::READ_CUDA))
		return;
	int objId = 0;
	while (true)
	{
		if (!pVCont->BuildBVH(objId))
			break;
		if (!pVCont->WorldBuildBVH())
			break;
		objId++;
	}
}

cCudaMainRuntimeClass::cCudaMainRuntimeClass()
{
	pCamera = std::make_unique<CudaGCamera>();
	pVCont = std::make_unique<CudaVertexController>();
	pDevice = std::make_unique<CudaGDevice>();
}

cCudaMainRuntimeClass::~cCudaMainRuntimeClass()
{
	pCamera.release();
	pVCont.release();
	pDevice.release();
}
