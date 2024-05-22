#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <vector>
#include <d3d11_1.h>
#include <DirectXMath.h>

#include <assimp\Importer.hpp>
#include <assimp\scene.h>
#include <assimp\postprocess.h>

#include "Mesh.h"
#include "TextureLoader.h"

using namespace DirectX;

class ModelLoader
{
public:
	ModelLoader();
	~ModelLoader();

	bool Load(ID3D11Device* dev, ID3D11DeviceContext* devcon, std::string filename);
	void Draw(ID3D11DeviceContext* devcon);
	void SetRotation(Vector4 vec);
	void SetPosition(Vector4 vec);

	void Close();
private:
	ID3D11Device* dev_;
	ID3D11DeviceContext* devcon_;
	std::vector<Mesh> meshes_;
	std::string directory_;
	std::vector<Texture> textures_loaded_;

	void processNode(aiNode* node, const aiScene* scene);
	Mesh processMesh(aiMesh* mesh, const aiScene* scene);
	std::vector<Texture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName, const aiScene* scene);
	ID3D11ShaderResourceView* loadEmbeddedTexture(const aiTexture* embeddedTexture);
};

#endif // !MODEL_LOADER_H