#include "pch.h"
#include "cLibController.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <windows.h>

void cLibController::Importer(std::string _Path)
{
	struct _finddata_t fd;
	intptr_t handle;
	if ((handle = _findfirst(_Path.c_str(), &fd)) == -1L)
	{
		std::cout << "No file in directory!" << std::endl;
	}
	do
	{
		std::cout << fd.name << std::endl;
		std::string a(fd.name);
		sLib temp;
		temp.s_Path = _Path;
		int funcsize = 0;
		if (a.back() == 'h')
		{
			a.pop_back();
			if (a.back() == '.')
			{
				a.pop_back();
				temp.s_Name = a;
				std::ifstream ifs;
				std::string file = _Path + "/" + a + ".h";
				ifs.open(file);
				int offset;
				if (ifs.is_open())
				{
					for (;;)
					{
						std::string line;
						std::getline(ifs, line);
						if ((offset = line.find("void")) != std::string::npos)
						{
							funcsize++;
						}
						else if ((offset = line.find("int")) != std::string::npos)
						{
							funcsize++;
						}
						else if ((offset = line.find("char")) != std::string::npos)
						{
							funcsize++;
						}
						if (!ifs.eof())
							break;
					}
					temp.ps_Func = (std::string*)malloc(funcsize * sizeof(std::string));
					for (int i = 0; i < funcsize; i++)
					{
						temp.ps_Func[i] = "";
					}
					mv_Libs.push_back(temp);
					ifs.close();
				}
			}
		}
	} while (_findnext(handle, &fd) == 0);
	_findclose(handle);

}

void cLibController::ReadHeader()
{
}

cLibController::cLibController()
{
	mv_Libs.clear();
}

cLibController::~cLibController()
{
	mv_Libs.clear();
}
