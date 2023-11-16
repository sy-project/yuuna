#pragma once
#define WIN32_LEAN_AND_MEAN
#define _CRT_SECURE_NO_WARNINGS
#include <framework.h>
#include <d3d11.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <time.h>

#include <d3d11.h>
#include <DirectXMath.h>
#include <d3dcompiler.h>
#include <DirectXCollision.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")

#include <d2d1_2.h>
#include <dwrite.h>

#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dwrite.lib")

#include <DirectXTex.h>

#pragma comment(lib, "DirectXTex.lib")

#define LERP(s, e, t) s + (e - s) * t
#define MAX_BONE 500
#define MAX_FRAME_KEY 600
#define MAX_INSTANCE 400

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720
#define FULLSCREEN false
#define WINNAME	L"YuunaEngine"

#ifdef NDEBUG
#define V(hr) hr
#else
#define V(hr) assert(SUCCEEDED(hr))
#endif

using namespace DirectX;
using namespace std;
using namespace DirectX::TriangleTests;

typedef XMFLOAT4 Float4;
typedef XMFLOAT3 Float3;
typedef XMFLOAT2 Float2;
typedef XMVECTOR Vector4;
typedef XMMATRIX Matrix;
typedef XMFLOAT4X4 Float4x4;

const XMVECTORF32 kRight = { 1,0,0 };
const XMVECTORF32 kUp = { 0,1,0 };
const XMVECTORF32 kForward = { 0,0,1 };

#include <imgui.h>
#include <backends/imgui_impl_win32.h>
#include <backends/imgui_impl_dx11.h>

#include "Utility.h"
#include "cControl.h"
#include "cTimer.h"

#include "Device.h"
#include "ImGuiManager.h"
#include "cSystem.h"

//Shader
#include "Shader.h"
#include "VertexShader.h"
#include "HullShader.h"
#include "DomainShader.h"
#include "GeometryShader.h"
#include "PixelShader.h"
#include "ComputeShader.h"

#include "VertexBuffer.h"
#include "IndexBuffer.h"
#include "ConstBuffer.h"
#include "VertexLayouts.h"
#include "GlobalBuffers.h"

#include "Mesh.h"
#include "Texture.h"
#include "Material.h"

#include "Transform.h"
#include "cVector3.h"
#include "Math.h"

#include "Collider.h"
#include "BoxCollider.h"
#include "SphereCollider.h"
#include "CapsuleCollider.h"

#include "SamplerState.h"

#include "Camera.h"
#include "Environment.h"

extern HWND g_hWnd;