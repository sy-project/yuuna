#pragma once
#include <Singleton.h>
#include <Types.h>
#include "CudaGCamera.h"
#include "CudaGDevice.h"

enum eCUDA
{
	READ_CUDA		= 1 << 0,
	RENDER_BG		= 1 << 1,
	USE_BVH			= 1 << 2,
	USE_KDTREE		= 1 << 3,
};

class cCudaMainRuntimeClass : public Singleton<cCudaMainRuntimeClass>
{
	template<typename T>
	friend class Singleton;
public:
	bool CheckCUDAUsage(sWindow& _window);
	void Init(sWindow& _window);
	void ImportWindow(sWindow& _window);
	void Rendering(sWindow& _window, int _split = 1);

public: //2D
	void Import2DVertex(sWindow& _window, int objId, Vertex2D v0, Vertex2D v1, Vertex2D v2);
	void ImportImageFor2DV(sWindow& _window, unsigned int objId, unsigned int texId, std::string _path, std::vector<char> encData);
	
	void Update2DVertexPos(sWindow& _window, unsigned int objId, Vector::Vector2D v);
	void Update2DVertexRot(sWindow& _window, unsigned int objId, float _val);
	void Update2DVertexRot(sWindow& _window, unsigned int objId, float _val, Vector::Vector2D center);

private: //3D

private: //BVH
	void buildBVH(sWindow& _window);

private: //CUDA MAIN
	std::unique_ptr<CudaGCamera> pCamera;
	std::unique_ptr<CudaVertexController> pVCont;
	std::unique_ptr<CudaGDevice> pDevice;

private:
	cCudaMainRuntimeClass();
	~cCudaMainRuntimeClass();
};

