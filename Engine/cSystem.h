#pragma once
#include "cEngine.h"
class cSystem
{
private:
	cEngine* engine;
public:
	cSystem();
	~cSystem();

	void Update();

	void PreRender();
	void Render();
	void PostRender();

	void Create();
	void Delete();
};

