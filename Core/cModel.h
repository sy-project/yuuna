#pragma once
class cModel
{
public:
	int mobjId;
	std::string name;
	std::vector<Vertex3D> vertex;
	std::vector<Image2D> texture;
	std::vector<Triangle3D> triangle;

public:
	std::vector<Image2D> GetTextrue();
	cModel();
	cModel(int objId, std::string _path, std::string _format = "fbx");
	~cModel();

private:
	Image2D image;
};

