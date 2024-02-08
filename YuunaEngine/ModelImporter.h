#pragma once
class ModelImporter : public Singleton<ModelImporter>
{
	friend class Singleton;
	
public:

	bool Load(std::string filename);
	void Draw();

	void Close();
private:
	std::vector<Mesh> meshes_;
	std::string directory_;
	std::vector<_Texture> textures_loaded_;

	void processNode(aiNode* node, const aiScene* scene);
	Mesh processMesh(aiMesh* mesh, const aiScene* scene);
	std::vector<_Texture> loadMaterialTextures(aiMaterial* mat, aiTextureType type, std::string typeName, const aiScene* scene);
	ID3D11ShaderResourceView* loadEmbeddedTexture(const aiTexture* embeddedTexture);

private:
	ModelImporter();
	~ModelImporter();
};

