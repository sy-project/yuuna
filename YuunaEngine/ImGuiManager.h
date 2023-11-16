#pragma once
class ImGuiManager : public Singleton<ImGuiManager>
{
	friend class Singleton;
public:
	void OpenImGuiWindow(std::string _name);
	void NewFrame();
	void Render();

private:
	ImGuiManager();
	~ImGuiManager();
};

