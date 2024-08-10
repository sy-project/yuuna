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
    return 0;
}

int Core::End()
{
    cLogger::Delete();
    return 0;
}

int Core::Log::WriteLog(std::string _str)
{
    cLogger::Get()->Logging(_str);
    return 0;
}