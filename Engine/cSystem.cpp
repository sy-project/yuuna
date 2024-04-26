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
	ImGui_ImplDX11_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();
	engine->PostRender();
	ImGui::Render();
	ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
}

void cSystem::Create()
{
	cDevice::Get();

	ImGui::CreateContext();
	ImGui::StyleColorsDark();

	ImGui_ImplWin32_Init(g_hWnd);
	ImGui_ImplDX11_Init(cDevice::Get()->GetDevice(), cDevice::Get()->GetDeviceContext());
	engine = new cEngine();
}

void cSystem::Delete()
{
	delete engine;
	cDevice::Delete();
}
