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
	ImVec4 clear_color;
	bool show_demo_window;

private:
	cSystem();
	~cSystem();
};

