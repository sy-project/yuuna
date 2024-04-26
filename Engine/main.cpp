#include "framework.h"

HWND g_hWnd;

LRESULT CALLBACK MessageHandler(HWND, UINT, WPARAM, LPARAM);
static LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

void Init()
{
	WNDCLASSEX wc = {};
	DEVMODE dmScreenSettings;
	int posX, posY;

	wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
	wc.lpfnWndProc = WndProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = GetModuleHandle(NULL);
	wc.hIcon = LoadIcon(NULL, IDI_WINLOGO);
	wc.hIconSm = wc.hIcon;
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wc.lpszMenuName = NULL;
	wc.lpszClassName = WIN_NAME;
	wc.cbSize = sizeof(WNDCLASSEX);

	RegisterClassEx(&wc);

	if (FULLSCREEN)
	{
		memset(&dmScreenSettings, 0, sizeof(dmScreenSettings));
		dmScreenSettings.dmSize = sizeof(dmScreenSettings);
		dmScreenSettings.dmPelsWidth = (unsigned long)WIN_X;
		dmScreenSettings.dmPelsHeight = (unsigned long)WIN_Y;
		dmScreenSettings.dmBitsPerPel = 32;
		dmScreenSettings.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT;

		ChangeDisplaySettings(&dmScreenSettings, CDS_FULLSCREEN);

		posX = posY = 0;
	}
	else
	{
		posX = (GetSystemMetrics(SM_CXSCREEN) - WIN_X) / 2;
		posY = (GetSystemMetrics(SM_CYSCREEN) - WIN_Y) / 2;
	}

	g_hWnd = CreateWindowEx(0, WIN_NAME, WIN_NAME,
		WS_OVERLAPPEDWINDOW,
		posX, posY, WIN_X, WIN_Y, NULL, NULL, GetModuleHandle(NULL), NULL);
	DWORD test = (GetLastError());
	ShowWindow(g_hWnd, SW_SHOW);
	SetForegroundWindow(g_hWnd);
	SetFocus(g_hWnd);

	//ShowCursor(false);

	return;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR pScmdline, int iCmdshow)
{
	Init();
	while (cProgram::Get()->Update())
	{
		cProgram::Get()->Render();
	}

	cProgram::Get()->Delete();

	ShowCursor(true);

	if (FULLSCREEN)
	{
		ChangeDisplaySettings(NULL, 0);
	}

	DestroyWindow(g_hWnd);
	g_hWnd = NULL;

	UnregisterClass(WIN_NAME, GetModuleHandle(NULL));

	return 0;
}

LRESULT CALLBACK MessageHandler(HWND hwnd, UINT umsg, WPARAM wparam, LPARAM lparam)
{
	switch (umsg)
	{
	case WM_INITDIALOG:
		return (INT_PTR)TRUE;

	case WM_COMMAND:
		if (LOWORD(wparam) == IDOK || LOWORD(wparam) == IDCANCEL)
		{
			EndDialog(hwnd, LOWORD(wparam));
			return (INT_PTR)TRUE;
		}
		break;
	default:
	{
		return DefWindowProc(hwnd, umsg, wparam, lparam);
	}
	}
	return (INT_PTR)FALSE;
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT umessage, WPARAM wparam, LPARAM lparam)
{
	if (ImGui_ImplWin32_WndProcHandler(hwnd, umessage, wparam, lparam))
		return true;

	switch (umessage)
	{
	case WM_MOUSEMOVE:
		//cControl::Get()->SetMouse(lparam);
		break;
	case WM_MOUSEWHEEL:
	{
		short value = (short)HIWORD(wparam);
		//cControl::Get()->SetWheel((float)value);
	}
	break;
	case WM_DESTROY:
	{
		PostQuitMessage(0);
		return 0;
	}
	case WM_CLOSE:
	{
		PostQuitMessage(0);
		return 0;
	}
	default:
	{
		return MessageHandler(hwnd, umessage, wparam, lparam);
	}
	}
}