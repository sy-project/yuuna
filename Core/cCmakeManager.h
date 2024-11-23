#pragma once
class cCmakeManager : public Singleton<cCmakeManager>
{
	friend class Singleton;
public:
	void MakeCmakeFile(std::string _ProjectName);
	void RunCmake();

public:
	void SetPath(std::string _path) { m_path = _path; }

public:
	void IncludeLib(std::string _lib) { m_vlib.push_back(_lib); }

private:
	std::string m_path;
	std::string m_FileName;
	std::vector<std::string> m_vlib;

private:
	cCmakeManager();
	~cCmakeManager();
};

