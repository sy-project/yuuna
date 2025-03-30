#include "pch.h"

void cModelManager::ImportModel(std::string _path, std::string _format)
{
	cModel* m = new cModel(m_vModelvecList.size(), _path, _format);
	m_vModelvecList.push_back(m);
}

std::vector<Vertex3D> cModelManager::GetV(int objId)
{
	return m_vModelvecList.at(objId)->vertex;
}

std::vector<Image2D> cModelManager::GetT(int objId)
{
	return m_vModelvecList.at(objId)->GetTextrue();
}

std::vector<Triangle3D> cModelManager::GetTri(int objId)
{
	return m_vModelvecList.at(objId)->triangle;
}

int cModelManager::GetObjId(std::string _name)
{
	for (unsigned int i = 0; i < m_vModelvecList.size(); i++)
	{
		if (m_vModelvecList.at(i)->name == _name)
			return i;
	}
	return -1;
}

std::string cModelManager::GetModelName(int objId)
{
	return m_vModelvecList.at(objId)->name;
}

cModelManager::cModelManager()
{
	m_vModelvecList.clear();
}

cModelManager::~cModelManager()
{
	m_vModelvecList.clear();
}
