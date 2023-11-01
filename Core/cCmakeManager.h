#pragma once
class cCmakeManager : Singleton<cCmakeManager>
{
	friend class Singleton;
public:

public:
	void SetPath(std::string _path) { m_path = _path; }

public:
	void IncludeLib(std::string _lib) { m_vlib.push_back(_lib); }

private:
	std::string m_path;
	std::vector<std::string> m_vlib;

private:
	cCmakeManager();
	~cCmakeManager();
};

