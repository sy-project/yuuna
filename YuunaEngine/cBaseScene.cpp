#include "header.h"

cBaseScene::~cBaseScene()
{
}

void cBaseScene::Init()
{
    Shader::AddVS(L"VertexShader", "main");
    Shader::AddPS(L"PixelShader", "main");
    ModelImporter::Get();
    ModelImporter::Get()->Load("Model/sponza/sponza.obj");
}

bool cBaseScene::Update()
{
	return true;
}

void cBaseScene::PreRender()
{
}

void cBaseScene::Render()
{
    ModelImporter::Get()->Draw();
}

void cBaseScene::PostRender()
{
    if (ImGuiManager::Get()->OpenImGuiWindow("GameWindow"))
    {
        ImGui::End();
    }
}
