#pragma once
class cEngine
{
private:
	string m_pxShader;
	string m_vxShader;

public:
	cEngine();
	~cEngine();

	void Update();
	void PreRender();
	void Render();
	void PostRender();
};

