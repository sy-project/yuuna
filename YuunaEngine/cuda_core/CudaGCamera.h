#pragma once
#include <Types.h>
#include <thread>

class CudaGCamera
{
public:
	CudaGCamera();
	~CudaGCamera();
public: //GetSet
	Camera GetCamera() { return cam; }

	void SetFrameBuffer();

	void SetWH(int W, int H);

	void SetFOV(float fov = 90.0f);

	void SetPosition(float x, float y, float z);
	void SetPosition(Vector::Vector3D pos);

	void SetRotation(float x, float y, float z);
	void SetRotation(Vector::Vector3D rot);

	void SetForward(float x, float y, float z);
	void SetForward(Vector::Vector3D vec);

	void SetUp(float x, float y, float z);
	void SetUp(Vector::Vector3D vec);

	void SetMultiRendering(uint8_t* framebuffer, int _split = 1);

public: // Update Render
	void MoveForward(float val);
	void Move(Vector::Vector3D vec);
	void Render(uint8_t* framebuffer, int _split = 1);

public:
	void Input2DVertexToCam(unsigned int objId, Vertex2D v0, Vertex2D v1, Vertex2D v2);
	void Input2DImageTo2DVertex(unsigned int objId, unsigned int texId, std::string _path, std::vector<char> encData);

private:
	std::vector<std::thread> worker;
	Camera cam;
	int _W, _H;
	uint8_t* _framebuffer;
	int m_isplit;

private: //2D

private: //3D

};

class CudaVertexController
{
public:
	CudaVertexController();
	~CudaVertexController();
public:
	void Update2DVertexPos(unsigned int objId, Vector::Vector2D v);
	void Update2DVertexRot(unsigned int objId, float _val);
	void Update2DVertexRot(unsigned int objId, float _val, Vector::Vector2D center);

	bool BuildBVH(unsigned int objId);
	bool WorldBuildBVH();

private:

};