#pragma once

#define WIN32_LEAN_AND_MEAN             // 거의 사용되지 않는 내용을 Windows 헤더에서 제외합니다.
#pragma warning(disable: 4996)
#define _CRT_SECURE_NO_WARNINGS 1

#define CMAKEVERSION 3.11
// Windows 헤더 파일
#include <windows.h>
#include <iostream>
#include <assert.h>
#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <functional>
#include <io.h>
#include <urlmon.h>
#pragma comment(lib, "urlmon.lib")
#include "Singleton.h"
#include "Common.h"
#include "cLogger.h"
#include "cEncryptManager.h"
#include "cCmakeManager.h"