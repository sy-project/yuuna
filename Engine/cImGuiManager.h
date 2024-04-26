#pragma once
class cImGuiManager : public Singleton<cImGuiManager>
{
	friend class Singleton;
public:
	bool OpenImGuiWindow(std::string _name, int viewport_x = 650, int viewport_y = 20);
	void NewFrame();
	void Render();

private:
	cImGuiManager();
	~cImGuiManager();
};