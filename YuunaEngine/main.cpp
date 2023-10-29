#include "header.h"

HWND g_hWnd;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR pScmdline, int iCmdshow)
{
	cSystem::Get()->Init(WINDOW_WIDTH, WINDOW_HEIGHT, FULLSCREEN);
	while (cSystem::Get()->Update())
		cSystem::Get()->Render();
	Device::Delete();
	cSystem::Delete();
	return 0;
}