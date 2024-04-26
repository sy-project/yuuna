#include "framework.h"
#include "cEngine.h"
#include "ModelLoader.h"
#include "SafeRelease.hpp"

cEngine::cEngine()
{
	m_pxShader = "PixelShader.hlsl";
	m_vxShader = "VertexShader.hlsl";
}

cEngine::~cEngine()
{
}

void cEngine::Update()
{
}

void cEngine::PreRender()
{
}

void cEngine::Render()
{
}

void cEngine::PostRender()
{
}

