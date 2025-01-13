#pragma once
class cModelManager : public Singleton<cModelManager>
{
	friend class Singleton;
private:
	vector<cModel*> m_vModels;

public:
	bool importModel(string _path);
	bool removeModel(int _num);
	cModel* GetModel(int _num);
	//void DrawModel();

private:
	cModelManager();
	~cModelManager();
};

