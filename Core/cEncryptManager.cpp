#include "pch.h"

void cEncryptManager::Encrypt()
{
}

void cEncryptManager::Decrypt()
{
}

std::string cEncryptManager::EncryptData(std::string _data)
{
	std::string temp = "";
	for (int i = 0; i < _data.length(); i++)
	{
		temp += (_data.at(i) + key);
	}
	return temp;
}

std::string cEncryptManager::DecryptData(std::string _data)
{
	std::string temp = "";
	for (int i = 0; i < _data.length(); i++)
	{
		temp += (_data.at(i) - key);
	}
	return temp;
}

std::string cEncryptManager::EncryptNetwork(sPlayerDescription _data)
{
	std::string temp = "";
	return temp;
}

sPlayerDescription cEncryptManager::DecryptNetwork(std::string _data)
{
	return sPlayerDescription();
}

cEncryptManager::cEncryptManager()
{
	key = 1234;
}

cEncryptManager::~cEncryptManager()
{
}
