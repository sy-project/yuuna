#pragma once
struct sLib
{
	std::string		s_Path;
	std::string		s_Name;
	std::string*	ps_Func;
};

class cLibController : public Singleton<cLibController>
{
	friend class Singleton;
public:
	void Importer(std::string _Path);
	void ReadHeader();
	void CheckFunc();

private:
	std::vector<sLib> mv_Libs;
	std::vector<std::string> mv_funcform;

private:
	cLibController();
	~cLibController();
};

