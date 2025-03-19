// dllmain.cpp : DLL 애플리케이션의 진입점을 정의합니다.
#include "pch.h"

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

int Core::Init()
{
    cLogger::Get();
    cLibController::Get();
    cEncryptManager::Get();
    cCmakeManager::Get();
    cJsonController::Get();
    cControl::Get();
    return 0;
}

int Core::End()
{
    cLogger::Delete();
    cLibController::Delete();
    cEncryptManager::Delete();
    cCmakeManager::Delete();
    cJsonController::Delete();
    cControl::Delete();
    return 0;
}

int Core::Log::WriteLog(std::string _str)
{
    cLogger::Get()->Logging(_str);
    return 0;
}

void Core::Log::SetJsonFilename(std::string& _filename)
{
    return cJsonController::Get()->SetFilename(_filename);
}

bool Core::Log::LoadJson()
{
    return cJsonController::Get()->Load();
}

bool Core::Log::SaveJson()
{
    return cJsonController::Get()->Save();
}

std::string Core::Log::GetValueFromJson(const std::string& key, const std::string& defaultValue)
{
    return cJsonController::Get()->GetValue(key, defaultValue);
}

void Core::Log::SetValueFromJson(const std::string& key, const std::string& value)
{
    return cJsonController::Get()->SetValue(key, value);
}

std::vector<std::string> Core::Cmake::RunCommand(const std::string& command)
{
    return cCmakeManager::Get()->RunCMakeCommand(command);
}

cLibController* Core::Lib::GetLibContInstance()
{
    return cLibController::Get();
}

bool Core::Lib::LoadPluginFromName(const std::string& pluginName, const std::string& dllPath)
{
    return cLibController::Get()->LoadPlugin(pluginName, dllPath);
}

void Core::Lib::UnloadAllPlugin(const std::string& pluginName)
{
    return cLibController::Get()->UnloadPlugin(pluginName);
}

std::string Core::Encrypt::EStringData(std::string _str)
{
    return cEncryptManager::Get()->EncryptData(_str);
}

std::string Core::Encrypt::EPlayerData(sPlayerDescription _data)
{
    return cEncryptManager::Get()->EncryptNetwork(_data);
}

std::string Core::Decrypt::DStringData(std::string _str)
{
    return cEncryptManager::Get()->DecryptData(_str);
}

sPlayerDescription Core::Decrypt::DPlayerData(std::string _data)
{
    return cEncryptManager::Get()->DecryptNetwork(_data);
}

void Core::CONTROL::UpdateInput()
{
    cControl::Get()->Update();
}

bool Core::CONTROL::KeyDown(UINT key)
{
    return cControl::Get()->Down(key);
}

bool Core::CONTROL::KeyUp(UINT key)
{
    return cControl::Get()->Up(key);
}

bool Core::CONTROL::KeyPress(UINT key)
{
    return cControl::Get()->Press(key);
}

Vector3D Core::CONTROL::MouseGet()
{
    return cControl::Get()->GetMouse();
}

void Core::CONTROL::MouseSet(LPARAM lParam, int WIN_X, int WIN_Y)
{
    return cControl::Get()->SetMouse(lParam, WIN_X, WIN_Y);
}

float Core::CONTROL::MouseWheelGet()
{
    return cControl::Get()->GetWheel();
}

void Core::CONTROL::MouseWheelSet(float value)
{
    return cControl::Get()->SetWheel(value);
}
