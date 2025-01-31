#include "pch.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <windows.h>

bool cLibController::LoadPlugin(const std::string& pluginName, const std::string& dllPath)
{
    if (loadedPlugins.find(pluginName) != loadedPlugins.end()) {
        std::cerr << "Plugin already loaded: " << pluginName << std::endl;
        return false;
    }

    HMODULE handle = LoadLibraryA(dllPath.c_str());
    if (!handle) {
        std::cerr << "Failed to load plugin: " << dllPath << std::endl;
        return false;
    }

    using GetFunctionsFunc = std::vector<std::pair<std::string, FunctionPtr>>(*)();
    GetFunctionsFunc getFunctions = (GetFunctionsFunc)GetProcAddress(handle, "GetPluginFunctions");

    if (!getFunctions) {
        std::cerr << "Plugin does not provide GetPluginFunctions(): " << dllPath << std::endl;
        FreeLibrary(handle);
        return false;
    }

    std::vector<std::pair<std::string, FunctionPtr>> functionList = getFunctions();

    PluginInfo info;
    info.handle = handle;
    for (const auto& func : functionList) {
        info.functions[func.first] = func.second;
        std::cout << "Loaded function: " << pluginName << "/" << func.first << std::endl;
    }

    loadedPlugins[pluginName] = info;
    return true;
}

void cLibController::UnloadPlugin(const std::string& pluginName)
{
    if (loadedPlugins.find(pluginName) == loadedPlugins.end()) {
        return;
    }

    FreeLibrary(loadedPlugins[pluginName].handle);
    loadedPlugins.erase(pluginName);
    std::cout << "Unloaded plugin: " << pluginName << std::endl;
}

cLibController::cLibController()
{
}

cLibController::~cLibController()
{
}
