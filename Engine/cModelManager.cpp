#include "framework.h"
#include "cModelManager.h"

bool cModelManager::importModel(string _path)
{
    cModel* model = new cModel(_path);
    int num = m_vModels.size();
    m_vModels.push_back(model);
    if(num == m_vModels.size() + 1)
        return true;
    return false;
}

bool cModelManager::removeModel(int _num)
{
    int num = m_vModels.size();
    delete m_vModels.at(_num);
    m_vModels.erase(m_vModels.begin() + num);
    if (num == m_vModels.size() - 1)
        return true;
    return false;
}

cModel* cModelManager::GetModel(int _num)
{
    return m_vModels.at(_num);
}

cModelManager::cModelManager()
{
    m_vModels.clear();
}

cModelManager::~cModelManager()
{
    for (auto data : m_vModels)
        delete data;
    m_vModels.clear();
}
