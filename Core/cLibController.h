#pragma once
struct sLib
{
	std::string		s_Path;
	std::string		s_Name;
	std::string*	ps_Func;
};

class cLibController : Singleton<cLibController>
{
	friend class Singleton;
public:
	void Importer(std::string _Path);
	void ReadHeader();

private:
	std::vector<sLib> mv_Libs;

private:
	cLibController();
	~cLibController();
};

