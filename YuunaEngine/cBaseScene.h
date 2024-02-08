#pragma once
class cBaseScene : public cScene
{
public:
	~cBaseScene() override;

	void Init() override;
	bool Update() override;
	void PreRender() override;
	void Render() override;
	void PostRender() override;
};

