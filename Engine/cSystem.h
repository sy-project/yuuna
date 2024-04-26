#pragma once
class cSystem
{
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

