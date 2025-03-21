#pragma once
class cCmakeManager : public Singleton<cCmakeManager>
{
	template<typename T>
	friend class Singleton;
public:
	void MakeCmakeFile(std::string _ProjectName);
	static std::vector<std::string> RunCMakeCommand(const std::string& command);

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

