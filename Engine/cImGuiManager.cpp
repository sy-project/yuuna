#include "framework.h"

bool cImGuiManager::OpenImGuiWindow(std::string _name, int viewport_x, int viewport_y)
{
    IM_ASSERT(ImGui::GetCurrentContext() != NULL && "Missing Dear ImGui context. Refer to examples app!");
    const ImGuiViewport* main_viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(main_viewport->WorkPos.x + viewport_x, main_viewport->WorkPos.y + viewport_y), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(550, 680), ImGuiCond_FirstUseEver);
    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_MenuBar;

    if (!ImGui::Begin(_name.c_str()))
        return false;
    ImGui::IsWindowAppearing();
    return true;
}

void cImGuiManager::ShowCMakeLogWindow()
{
    static std::vector<std::string> log;
    static bool runCMake = false;

    cImGuiManager::Get()->OpenImGuiWindow("Build Tool Box");
    if (ImGui::Button("Build")) {
        log = Core::Cmake::RunCommand("cmake .. && cmake --build .");
        runCMake = true;
    }

    if (runCMake) {
        for (const auto& line : log) {
            ImGui::TextWrapped("%s", line.c_str());
        }
    }

    ImGui::End();
}

void cImGuiManager::ShowLibList()
{
    cImGuiManager::Get()->OpenImGuiWindow("Lib List");
    auto pluginloader = Core::Lib::GetLibContInstance();
    for (const auto& pluginPair : pluginloader->GetLoadedPlugins()) {
        const std::string& pluginName = pluginPair.first;
        const PluginInfo& pluginInfo = pluginPair.second;

        for (const auto& functionPair : pluginInfo.functions) {
            const std::string& functionName = functionPair.first;
            std::string displayText = pluginName + "/" + functionName;

            ImGui::Text("%s", displayText.c_str());
        }
    }
    ImGui::End();
}

void cImGuiManager::NewFrame()
{
    ImGui_ImplDX11_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
}

void cImGuiManager::Render()
{
    ImGui::Render();
    ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
}

cImGuiManager::cImGuiManager()
{
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
    ImGui_ImplDX11_Init(cDevice::Get()->GetDevice(), cDevice::Get()->GetDeviceContext());
}

cImGuiManager::~cImGuiManager()
{
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
}