#pragma once
#include "cuda_header.h"
#include <windows.h>
#include <stdio.h>

namespace cuda_dll
{
	extern "C" __declspec(dllexport) int test(int* c, const int* a, const int* b, unsigned int size);
	namespace math
	{

	}
	namespace physics
	{

	}
}