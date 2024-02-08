#pragma once
class cScene
{
public:
	cScene();
	virtual ~cScene() {}

	virtual void Init() {};
	virtual bool Update() { return true; };
	virtual void PreRender() {};
	virtual void Render() {};
	virtual void PostRender() {};

protected:

};

