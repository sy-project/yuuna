#pragma once

#define WIN32_LEAN_AND_MEAN             // 거의 사용되지 않는 내용을 Windows 헤더에서 제외합니다.
#pragma warning(disable: 4996)
#define _CRT_SECURE_NO_WARNINGS 1

#define CMAKEVERSION 3.15

#define KEY "my_secret_key_123"  // 16 bytes AES 키
#define IV "0000000000000000"    // 16 bytes IV

// Windows 헤더 파일
#include <windows.h>
#include <iostream>
#include <fstream>
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
#include "cLibController.h"
#include "cEncryptManager.h"
#include "cCmakeManager.h"
#include "cJsonController.h"