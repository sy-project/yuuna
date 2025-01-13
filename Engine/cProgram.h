#pragma once
class cProgram : public Singleton<cProgram>
{
private:

private:
	friend class Singleton;

	cProgram();
	~cProgram();

public:
	bool Init();
	bool Update();
	void Render();
	void Delete();
};