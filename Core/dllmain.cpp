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

int Core::Init(int _key = 1234)
{
    cLogger::Get();
    cLibController::Get();
    cEncryptManager::Get();
    cEncryptManager::Get()->SetKey(_key);
    cCmakeManager::Get();
    return 0;
}

int Core::End()
{
    cLogger::Delete();
    cLibController::Delete();
    cEncryptManager::Delete();
    cCmakeManager::Delete();
    return 0;
}

int Core::Log::WriteLog(std::string _str)
{
    cLogger::Get()->Logging(_str);
    return 0;
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