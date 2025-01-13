#include "pch.h"
#include <iostream>
#include <fstream>

using namespace std;

void cCmakeManager::MakeCmakeFile(std::string _ProjectName)
{
	ofstream ofs;
	string fileName = m_path + "/CMakeLists.txt";
	ofs.open(fileName);
	m_FileName = _ProjectName;

	string str;
	str += "cmake_minimum_required(VERSION ";
	str += to_string(CMAKEVERSION);
	str += ")\n\n";

	str += "project(";
	str += _ProjectName;
	str += ")\n";

	str += "set(CMAKE_CXX_STANDARD 17)\n";
	str += "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n";

	str += "file(GLOB Project_src\n";
	str += "     \"src/*.cpp\"";
	str += ")\n";

	str += "include_directories(./include)\n";

	str += "add_executable(";
	str += _ProjectName;
	str += " ${Project_src})\n\n";

	for(const auto& value : m_vlib)
	{
		str += "target_include_directories(";
		str += _ProjectName;
		str += " PRIVATE ./Lib";
		str += value;
		str += ")\n";
	}

	ofs.write(str.c_str(), str.size());
	ofs.close();
}

void cCmakeManager::RunCmake()
{
	string out = "cmake " + m_path;
	system(out.c_str());
	string temp = "";
	temp += m_path;
	temp += "/Makefile";
	for (;;)
	{
		if (_access(temp.c_str(), 0) != -1)
		{
			system("make");
			break;
		}
		else
			Sleep(50);
	}
	temp = m_path;
	temp += "/" + m_FileName + ".exe";
	for (;;)
	{
		if (_access(temp.c_str(), 0) != -1)
		{
			out = "start " + m_path;
			system(out.c_str());
			break;
		}
		else
			Sleep(50);
	}
}

cCmakeManager::cCmakeManager()
{
	m_path = "";
	m_vlib.clear();
}

cCmakeManager::~cCmakeManager()
{
	m_vlib.clear();
}
