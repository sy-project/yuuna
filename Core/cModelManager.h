#pragma once
class cModelManager : public Singleton<cModelManager>
{
	template<typename T>
	friend class Singleton;
public:
	void ImportModel(std::string _path, std::string _format = "");
	std::vector<Vertex3D> GetV(int objId);
	std::vector<Image2D> GetT(int objId);
	std::vector<Triangle3D> GetTri(int objId);
	int GetObjId(std::string _name);
	std::string GetModelName(int objId);

private:
	std::vector<cModel*> m_vModelvecList;

	cModelManager();
	~cModelManager();
};

