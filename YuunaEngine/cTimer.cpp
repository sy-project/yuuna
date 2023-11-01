#include "header.h"
#include "cTimer.h"

cTimer::cTimer()
	: frameRate(0), frameCount(0), timeElapsed(0), oneSecCount(0),
	runTime(0), lockFPS(0)
{
	//1초동안 CPU의 진동수를 반환하는 함수
	QueryPerformanceFrequency((LARGE_INTEGER*)&periodFrequency);

	//현재 CPU진동수
	QueryPerformanceCounter((LARGE_INTEGER*)&lastTime);

	timeScale = 1.0f / (float)periodFrequency;
}

cTimer::~cTimer()
{
}

void cTimer::Update()
{
	QueryPerformanceCounter((LARGE_INTEGER*)&curTime);
	timeElapsed = (float)(curTime - lastTime) * timeScale;

	if (lockFPS != 0.0f)
	{
		while (timeElapsed < (1.0f / lockFPS))
		{
			QueryPerformanceCounter((LARGE_INTEGER*)&curTime);
			timeElapsed = (float)(curTime - lastTime) * timeScale;
		}
	}

	lastTime = curTime;

	//FPS(Frame Per Second)
	frameCount++;
	oneSecCount += timeElapsed;

	if (oneSecCount >= 1.0f)
	{
		frameRate = frameCount;
		frameCount = 0;
		oneSecCount = 0.0f;
	}

	runTime += timeElapsed;
}
