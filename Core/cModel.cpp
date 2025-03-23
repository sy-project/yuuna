#include "pch.h"
#include "StrUtils.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#ifdef _DEBUG
#pragma comment(lib, "Debug/assimp-vc143-mtd.lib")
#else
#pragma comment(lib, "Release/assimp-vc143-mt.lib")
#endif
cModel::cModel()
{

}

cModel::cModel(int objId, std::string _path, std::string _format)
{
	Assimp::Importer imp;
	const aiScene* scene;
	if (getExtension(_path) == "enc")
	{
		std::vector<char> decryptedData = cEncryptManager::Get()->DecryptResourceFile(_path);
		scene = imp.ReadFileFromMemory(
			decryptedData.data(),
			decryptedData.size(),
			aiProcess_Triangulate |
			aiProcess_ConvertToLeftHanded,
			_format.c_str()
		);
	}
	else
	{
		scene = imp.ReadFile(_path, aiProcess_Triangulate |
			aiProcess_ConvertToLeftHanded);
	}

	if (scene == nullptr)
	{
		std::cout << "Assimp Error:" << imp.GetErrorString() << std::endl;
		return;
	}
	
	aiMesh* mesh = scene->mMeshes[0];

	for (unsigned int i = 0; i < mesh->mNumVertices; i++)
	{
		Vertex3D tempD;
		tempD.p = { mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z };
		tempD.uv = { mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y };
		vertex.push_back(tempD);
	}
	mobjId = objId;
	name = GetFileName(_path);
}

cModel::~cModel()
{

}
