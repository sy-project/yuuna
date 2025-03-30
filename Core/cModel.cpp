#include "pch.h"
#include "StrUtils.h"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifdef _DEBUG
#pragma comment(lib, "Debug/assimp-vc143-mtd.lib")
#else
#pragma comment(lib, "Release/assimp-vc143-mt.lib")
#endif

static const aiScene* scene;
//const aiScene* mscene;

std::vector<Image2D> cModel::GetTextrue()
{
	return texture;
}
cModel::cModel()
{

}

cModel::cModel(int objId, std::string _path, std::string _format)
{
	Assimp::Importer imp;
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
	for (unsigned int meshnum = 0; meshnum < scene->mNumMeshes; meshnum++)
	{
		aiMesh* mesh = scene->mMeshes[meshnum];

		for (unsigned int i = 0; i < mesh->mNumVertices; i++)
		{
			Vertex3D tempD;
			tempD.p = { mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z };
			tempD.uv = { mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y };
			vertex.push_back(tempD);
		}
		for (unsigned int i = 0; i < mesh->mNumFaces; i++)
		{
			if (mesh->mFaces[i].mNumIndices == 3)
			{
				Triangle3D tempD;
				tempD.objId = objId;
				tempD.texId = meshnum;
				tempD.v1 = vertex.at(mesh->mFaces[i].mIndices[0]);
				tempD.v2 = vertex.at(mesh->mFaces[i].mIndices[1]);
				tempD.v3 = vertex.at(mesh->mFaces[i].mIndices[2]);
				tempD.normal.x = mesh->mNormals->x;
				tempD.normal.y = mesh->mNormals->y;
				tempD.normal.z = mesh->mNormals->z;
				triangle.push_back(tempD);
			}
		}
		mobjId = objId;
		name = GetFileName(_path);

		image.img = nullptr;
		image.objId = 0;
		image.size = { 0,0 };
		if (!scene || scene->mNumMaterials == 0) return;

		aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

		aiString texPath;
		if (material->GetTexture(aiTextureType_DIFFUSE, 0, &texPath) != AI_SUCCESS)
			return;

		const char* path = texPath.C_Str();

		if (path[0] == '*') {
			int texIndex = atoi(path + 1);
			if (texIndex < 0 || texIndex >= (int)scene->mNumTextures) return;

			const aiTexture* tex = scene->mTextures[texIndex];
			if (!tex) return;

			int width, height, channels;
			uint8_t* imageData = nullptr;

			if (tex->mHeight == 0) {
				imageData = stbi_load_from_memory(
					reinterpret_cast<const stbi_uc*>(tex->pcData),
					tex->mWidth,
					&width, &height, &channels,
					STBI_rgb_alpha);
			}
			else {
				width = tex->mWidth;
				height = tex->mHeight;
				imageData = new uint8_t[width * height * 4];
				memcpy(imageData, tex->pcData, width * height * 4);
			}
			image.img = imageData;
			image.objId = mobjId;
			image.texId = meshnum;
			image.size.x = width;
			image.size.y = height;
			texture.push_back(image);

			return;
		}

		int width, height, channels;
		std::string tempp = GetFilePath(_path) + "/";
		tempp += path;
		uint8_t* imageData = stbi_load(tempp.c_str(), &width, &height, &channels, STBI_rgb_alpha);
		image.img = imageData;
		image.objId = mobjId;
		image.size.x = width;
		image.size.y = height;
		image.texId = meshnum;
		texture.push_back(image);
	}
}

cModel::~cModel()
{

}
