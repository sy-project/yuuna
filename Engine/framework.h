#pragma once
#define WIN32_LEAN_AND_MEAN
#define _CRT_SECURE_NO_WARNINGS

#define WIN_NAME L"SY_Engine"
#define WIN_X 1600
#define WIN_Y 900

#ifdef NDEBUG
#define V(hr) hr
#else
#define V(hr) assert(SUCCEEDED(hr))
#endif

#define FULLSCREEN false

//STL
#include <vector>
#include <list>
#include <map>
#include <unordered_map>

//WIN
#include <Windows.h>
#include <assert.h>
#include <string>
#include <functional>
#include <iostream>
#include <io.h>
#include <urlmon.h>
#include <shellapi.h>
#include <stdexcept>
#include <windowsx.h>
#include <../utf8.h>

#pragma comment(lib, "urlmon.lib")

//DX
#include <d3d11.h>
#include <DirectXMath.h>
#include <d3dcompiler.h>
#include <DirectXCollision.h>
#include <wrl.h>
#include <DirectXTex.h>
#include <DirectXTex.inl>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "DirectXTex.lib")


#include <d2d1_2.h>
#include <dwrite.h>

#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dwrite.lib")
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>
#include <backends/imgui_impl_win32.h>
#include <backends/imgui_impl_dx11.h>

#include <assimp/types.h>
#include <Assimp/Importer.hpp>
#include <Assimp/scene.h>
#include <Assimp/postprocess.h>

#include <fbxsdk.h>

#ifdef _DEBUG
#pragma comment(lib, "Debug/assimp-vc143-mtd.lib")
#pragma comment(lib, "Debug/SYCUDA.lib")
#pragma comment(lib, "Debug/Core.lib")
#pragma comment(lib, "debug/libfbxsdk.lib")
#else
#pragma comment(lib, "Release/assimp-vc143-mt.lib")
#pragma comment(lib, "Release/SYCUDA.lib")
#pragma comment(lib, "Release/Core.lib")
#pragma comment(lib, "release/libfbxsdk.lib")
#endif

using namespace DirectX;
using namespace std;
using namespace DirectX::TriangleTests;
using namespace Microsoft::WRL;

typedef XMFLOAT4 Float4;
typedef XMFLOAT3 Float3;
typedef XMFLOAT2 Float2;
typedef XMVECTOR Vector4;
typedef XMMATRIX Matrix;
typedef XMFLOAT4X4 Float4x4;

#include "cuda_main.h"
#include "pch.h"

#include "Singleton.h"
#include "cImGuiManager.h"
//#include "cModel.h"
//#include "cModelManager.h"
#include "cNetworkManager.h"
#include "cDevice.h"


#include "cProgram.h"

extern HWND g_hWnd;