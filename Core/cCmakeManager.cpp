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

std::vector<std::string> cCmakeManager::RunCMakeCommand(const std::string& command)
{
	std::vector<std::string> output;
	char buffer[256];

	FILE* pipe = _popen(command.c_str(), "r");

	if (!pipe) {
		output.push_back("Failed to run CMake command.");
		return output;
	}

	while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
		output.push_back(buffer);
	}

	_pclose(pipe);

	return output;
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
