#pragma once
class cSystem : public Singleton<cSystem>
{
	friend class Singleton;
public:
	bool Init(int _x, int _y, bool _fullscreen);
	bool Update();
	void PreRender();
	void Render();
	void PostRender();

private:

private:
	cSystem();
	~cSystem();
};

