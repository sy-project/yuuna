#pragma once
class cSystem : public Singleton<cSystem>
{
public:
	bool Init(int _x, int _y, bool _fullscreen);
	bool Update();
	void PreRender();
	void Render();
	void PostRender();

private:
	struct nk_context* ctx;
	struct nk_colorf bg;

private:
	cSystem();
	~cSystem();
};

