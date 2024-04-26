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
}

void cSystem::PreRender()
{
}

void cSystem::Render()
{
}

void cSystem::PostRender()
{
	ImGui_ImplDX11_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();


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
}

void cSystem::Delete()
{
	cDevice::Delete();
}
