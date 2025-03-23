#pragma once
class cModel
{
public:
	int mobjId;
	std::string name;
	std::vector<Vertex3D> vertex;

public:
	cModel();
	cModel(int objId, std::string _path, std::string _format = "");
	~cModel();
};

