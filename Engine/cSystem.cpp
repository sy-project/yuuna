#include "framework.h"
#include "cSystem.h"

cSystem::cSystem()
{
	Create();
}

cSystem::~cSystem()
{
	Delete();
}

void cSystem::Update()
{
	engine->Update();
}

void cSystem::PreRender()
{
	engine->PreRender();
}

void cSystem::Render()
{
	engine->Render();
}

void cSystem::PostRender()
{
	cImGuiManager::Get()->NewFrame();
	engine->PostRender();
	cImGuiManager::Get()->Render();
}

void cSystem::Create()
{
	cDevice::Get();
	engine = new cEngine();
	cImGuiManager::Get();
}

void cSystem::Delete()
{
	delete engine;
	cDevice::Delete();
}
