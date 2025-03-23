#pragma once
class cModelManager : public Singleton<cModelManager>
{
	template<typename T>
	friend class Singleton;
public:
	void ImportModel(std::string _path, std::string _format = "");
	std::vector<Vertex3D> GetV(int objId);
	int GetObjId(std::string _name);

private:
	std::vector<cModel*> m_vModelvecList;

	cModelManager();
	~cModelManager();
};

