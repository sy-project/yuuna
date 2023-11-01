#include "header.h"
#include "cSystem.h"

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
static LRESULT CALLBACK
WindowProc(HWND wnd, UINT msg, WPARAM wparam, LPARAM lparam)
{
    if (ImGui_ImplWin32_WndProcHandler(wnd, msg, wparam, lparam))
        return true;

    switch (msg)
    {
    case WM_MOUSEMOVE:
        cControl::Get()->SetMouse(lparam);
        break;
    case WM_MOUSEWHEEL:
    {
        short value = (short)HIWORD(wparam);
        cControl::Get()->SetWheel((float)value);
    }
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;

    case WM_SIZE:
        if (Device::Get()->GetSwapChain())
        {
            int width = LOWORD(lparam);
            int height = HIWORD(lparam);
            Device::Get()->CreateBackBuffer(width, height);
        }
        break;
    }


    return DefWindowProcW(wnd, msg, wparam, lparam);
}

bool cSystem::Init(int _x, int _y, bool _fullscreen)
{
	WNDCLASSEXW wc;
	RECT rect = { 0,0,_x,_y };
	DWORD style = WS_OVERLAPPED;
	DWORD exstyle = WS_EX_APPWINDOW;
	memset(&wc, 0, sizeof(wc));
	wc.style = CS_DBLCLKS;
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandleW(0);
    wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = WINNAME;
    wc.cbSize = sizeof(WNDCLASSEXW);
    RegisterClassExW(&wc);
    AdjustWindowRectEx(&rect, style, FALSE, exstyle);

    g_hWnd = CreateWindowExW(exstyle,
        wc.lpszClassName,
        WINNAME,
        style | WS_VISIBLE,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        rect.right - rect.left,
        rect.bottom - rect.top,
        NULL,
        NULL,
        wc.hInstance,
        NULL);

    if (!g_hWnd)
    {
        return false;
    }

    Device::Get()->CreateDeviceAndSwapChain(_x, _y, g_hWnd);
    Device::Get()->CreateBackBuffer(_x, _y);

    IMGUI_CHECKVERSION();

    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    ImGui::StyleColorsDark();

    ImGuiStyle& imgui_style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        imgui_style.WindowRounding = 0.0f;
        imgui_style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    ImGui_ImplWin32_Init(g_hWnd);
    ImGui_ImplDX11_Init(Device::Get()->GetDevice(), Device::Get()->GetDeviceContext());

    clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    show_demo_window = true;
	return true;
}

bool cSystem::Update()
{
    cControl::Get()->Update();
    cTimer::Get()->Update();
    cControl::Get()->SetWheel(0.0f);
    MSG msg;
    while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE))
    {
        if (msg.message == WM_QUIT)
            return false;
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
	return true;
}

void cSystem::PreRender()
{
    ImGui_ImplDX11_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    ImGui::ShowDemoWindow(&show_demo_window);
    ImGui::Render();
    Device::Get()->Clear();
}

void cSystem::Render()
{
    PreRender();

    PostRender();
}

void cSystem::PostRender()
{
    ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
    Device::Get()->Present();
}

cSystem::cSystem()
{
    cControl::Get();
    cTimer::Get();
}

cSystem::~cSystem()
{
    cControl::Delete();
    cTimer::Delete();
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
}
