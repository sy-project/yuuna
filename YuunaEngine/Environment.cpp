#include "header.h"

Environment::Environment()
{
	CreatePerspective();

	samplerState = new SamplerState();
	samplerState->SetState();

	mainCamera = new Camera();
	mainCamera->position = { 0, 5, -5 };	

	lightBuffer = new LightBuffer();
	lightBuffer->Add();
}

Environment::~Environment()
{
	delete projectionBuffer;
	delete lightBuffer;

	delete samplerState;	
}

void Environment::PostRender()
{
	mainCamera->PostRender();
	ImGui::Begin("Test");
	ImGui::Text("LightInfo");
	ImGui::ColorEdit4("AmbientColor", (float*)&lightBuffer->data.ambient);
	ImGui::ColorEdit4("AmbientCeilColor", (float*)&lightBuffer->data.ambientCeil);
	
	if (ImGui::Button("AddLight"))
	{
		lightBuffer->Add();
	}
	for (UINT i = 0; i < lightBuffer->data.lightCount; i++)
	{
		string name = "Light " + to_string(i);
		if (ImGui::BeginMenu(name.c_str()))
		{
			ImGui::Checkbox("Active", (bool*)&lightBuffer->data.lights[i].active);
			ImGui::SliderInt("Type", (int*)&lightBuffer->data.lights[i].type, 0, 3);
			ImGui::SliderFloat3("Direction", (float*)&lightBuffer->data.lights[i].direction, -1, 1);
			ImGui::SliderFloat3("LightPosition", (float*)&lightBuffer->data.lights[i].position, -100, 100);
			ImGui::SliderFloat("LightRange", (float*)&lightBuffer->data.lights[i].range, 0, 100);
			ImGui::ColorEdit4("LightColor", (float*)&lightBuffer->data.lights[i].color);			
			ImGui::SliderFloat("LightInner", (float*)&lightBuffer->data.lights[i].inner, 0, 90);
			ImGui::SliderFloat("LightOuter", (float*)&lightBuffer->data.lights[i].outer, 0, 180);
			ImGui::SliderFloat("LightLength", (float*)&lightBuffer->data.lights[i].length, 0, 180);			

			ImGui::EndMenu();
		}		
	}

	ImGui::End();
}

void Environment::Set()
{
	SetViewport();
	SetProjection();
	mainCamera->SetVS(1);
	lightBuffer->SetPSBuffer(0);
}

void Environment::SetProjection()
{	
	projectionBuffer->SetVSBuffer(2);
}

void Environment::SetViewport(UINT width, UINT height)
{
	viewport.Width = WINDOW_WIDTH;
	viewport.Height = WINDOW_HEIGHT;
	viewport.MinDepth = 0.0f;
	viewport.MaxDepth = 1.0f;
	viewport.TopLeftX = 0;
	viewport.TopLeftY = 0;

	Device::Get()->GetDeviceContext()->RSSetViewports(1, &viewport);
}

void Environment::CreatePerspective()
{	
	projection = XMMatrixPerspectiveFovLH(XM_PIDIV4,
		WINDOW_WIDTH / (float)WINDOW_HEIGHT, 0.1f, 1000.0f);

	projectionBuffer = new MatrixBuffer();
		
	projectionBuffer->Set(projection);
}
