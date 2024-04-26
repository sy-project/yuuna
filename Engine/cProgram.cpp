#include "framework.h"
#include "cSystem.h"

cSystem* m_system;

cProgram::cProgram()
{
	Init();
}

cProgram::~cProgram()
{
	Delete();
}

bool cProgram::Init()
{
	m_system = new cSystem();
	return true;
}

bool cProgram::Update()
{
	MSG msg;
	bool done = false, result;
	ZeroMemory(&msg, sizeof(MSG));

	if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
	if (msg.message == WM_QUIT)
	{
		done = true;
	}
	else
	{
		m_system->Update();
		result = true;
		if (!result)
		{
			done = true;
		}
	}
	if (!done)
		return result;
	else
		return false;
}

void cProgram::Render()
{
	m_system->PreRender();
	cDevice::Get()->Clear();
	cDevice::Get()->SetRenderTarget();
	m_system->Render();
	m_system->PostRender();
	cDevice::Get()->Present();
}

void cProgram::Delete()
{
	delete m_system;
}


