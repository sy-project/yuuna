#include "pch.h"
#include <fstream>
#include <ctime>

void cLogger::Logging(std::string _str)
{
	std::ofstream f;
	time_t timer = time(NULL);
	struct tm* t;
	t = localtime(&timer);
	std::string logfileName = "";
	logfileName += std::to_string(t->tm_year + 1900) + "_";
	logfileName += std::to_string(t->tm_mon + 1) + "_";
	logfileName += std::to_string(t->tm_mday);
	logfileName += ".log";
	f.open(logfileName, std::ios::app);
	f << _str << std::endl;
	f.close();
}

cLogger::cLogger()
{
}

cLogger::~cLogger()
{
}
