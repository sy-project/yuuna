#include "header.h"
#include "cSystem.h"
#include "../nuklear/nuklear.h"
#include "nuklear_d3d11.h"

struct nk_context* ctx;
struct nk_colorf bg;

static LRESULT CALLBACK
WindowProc(HWND wnd, UINT msg, WPARAM wparam, LPARAM lparam)
{
    switch (msg)
    {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;

    case WM_SIZE:
        if (Device::Get()->GetSwapChain())
        {
            int width = LOWORD(lparam);
            int height = HIWORD(lparam);
            Device::Get()->CreateBackBuffer(width, height);
            nk_d3d11_resize(Device::Get()->GetDeviceContext(), width, height);
        }
        break;
    }

    if (nk_d3d11_handle_event(wnd, msg, wparam, lparam))
        return 0;

    return DefWindowProcW(wnd, msg, wparam, lparam);
}

bool cSystem::Init(int _x, int _y, bool _fullscreen)
{
	WNDCLASSEXW wc;
    auto e = GetLastError();
	RECT rect = { 0,0,_x,_y };
	DWORD style = WS_OVERLAPPED;
	DWORD exstyle = WS_EX_APPWINDOW;
	memset(&wc, 0, sizeof(wc));
    e = GetLastError();
	wc.style = CS_DBLCLKS;
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandleW(0);
    wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = WINNAME;
    wc.cbSize = sizeof(WNDCLASSEXW);
    e = GetLastError();
    RegisterClassExW(&wc);
    AdjustWindowRectEx(&rect, style, FALSE, exstyle);
    e = GetLastError();

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
    e = GetLastError();

    if (!g_hWnd)
    {
        return false;
    }

    Device::Get()->CreateDeviceAndSwapChain(_x, _y, g_hWnd);
    Device::Get()->CreateBackBuffer(_x, _y);
    ctx = nk_d3d11_init(Device::Get()->GetDevice(), _x, _y, MAX_VERTEX_BUFFER, MAX_INDEX_BUFFER);
    
    struct nk_font_atlas* atlas;
    nk_d3d11_font_stash_begin(&atlas);
    /*struct nk_font *droid = nk_font_atlas_add_from_file(atlas, "../../extra_font/DroidSans.ttf", 14, 0);*/
    /*struct nk_font *robot = nk_font_atlas_add_from_file(atlas, "../../extra_font/Roboto-Regular.ttf", 14, 0);*/
    /*struct nk_font *future = nk_font_atlas_add_from_file(atlas, "../../extra_font/kenvector_future_thin.ttf", 13, 0);*/
    /*struct nk_font *clean = nk_font_atlas_add_from_file(atlas, "../../extra_font/ProggyClean.ttf", 12, 0);*/
    /*struct nk_font *tiny = nk_font_atlas_add_from_file(atlas, "../../extra_font/ProggyTiny.ttf", 10, 0);*/
    /*struct nk_font *cousine = nk_font_atlas_add_from_file(atlas, "../../extra_font/Cousine-Regular.ttf", 13, 0);*/
    nk_d3d11_font_stash_end();

    /* style.c */
#ifdef INCLUDE_STYLE
/*set_style(ctx, THEME_WHITE);*/
/*set_style(ctx, THEME_RED);*/
/*set_style(ctx, THEME_BLUE);*/
/*set_style(ctx, THEME_DARK);*/
#endif
    bg.r = 0.10f, bg.g = 0.18f, bg.b = 0.24f, bg.a = 1.0f;
	return true;
}

bool cSystem::Update()
{
    MSG msg;
    nk_input_begin(ctx);
    while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE))
    {
        if (msg.message == WM_QUIT)
            return false;
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    nk_input_end(ctx);
	return true;
}

void cSystem::PreRender()
{
    Device::Get()->Clear();
}

void cSystem::Render()
{
    PreRender();

    PostRender();
}

void cSystem::PostRender()
{
    if (nk_begin(ctx, "Demo", nk_rect(50, 50, 230, 250),
        NK_WINDOW_BORDER | NK_WINDOW_MOVABLE | NK_WINDOW_SCALABLE |
        NK_WINDOW_MINIMIZABLE | NK_WINDOW_TITLE))
    {
        enum { EASY, HARD };
        static int op = EASY;
        static int property = 20;

        nk_layout_row_static(ctx, 30, 80, 1);
        if (nk_button_label(ctx, "button"))
            fprintf(stdout, "button pressed\n");
        nk_layout_row_dynamic(ctx, 30, 2);
        if (nk_option_label(ctx, "easy", op == EASY)) op = EASY;
        if (nk_option_label(ctx, "hard", op == HARD)) op = HARD;
        nk_layout_row_dynamic(ctx, 22, 1);
        nk_property_int(ctx, "Compression:", 0, &property, 100, 10, 1);

        nk_layout_row_dynamic(ctx, 20, 1);
        nk_label(ctx, "background:", NK_TEXT_LEFT);
        nk_layout_row_dynamic(ctx, 25, 1);
        if (nk_combo_begin_color(ctx, nk_rgb_cf(bg), nk_vec2(nk_widget_width(ctx), 400))) {
            nk_layout_row_dynamic(ctx, 120, 1);
            bg = nk_color_picker(ctx, bg, NK_RGBA);
            nk_layout_row_dynamic(ctx, 25, 1);
            bg.r = nk_propertyf(ctx, "#R:", 0, bg.r, 1.0f, 0.01f, 0.005f);
            bg.g = nk_propertyf(ctx, "#G:", 0, bg.g, 1.0f, 0.01f, 0.005f);
            bg.b = nk_propertyf(ctx, "#B:", 0, bg.b, 1.0f, 0.01f, 0.005f);
            bg.a = nk_propertyf(ctx, "#A:", 0, bg.a, 1.0f, 0.01f, 0.005f);
            nk_combo_end(ctx);
        }
    }
    nk_end(ctx);
    nk_d3d11_render(Device::Get()->GetDeviceContext(), NK_ANTI_ALIASING_ON);
    Device::Get()->Present();
}

cSystem::cSystem()
{
}

cSystem::~cSystem()
{
    nk_d3d11_shutdown();
}
